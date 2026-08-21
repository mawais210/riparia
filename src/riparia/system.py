"""Couples upstream hydrology, both parties' reservoirs, and simple channel
routing into a single basin run for one candidate package.

Units: all flow/volume quantities MCM (or MCM/month for the monthly series
used throughout); energy in GWh/year; travel time in days.
"""

# Import necessary libraries
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from riparia.config_schema import ReservoirConfig, ScenarioConfig
from riparia.issues import Issue, Package
from riparia.reservoir import Reservoir, simulate

# Custom Functions


@dataclass
class BasinOutputs:
    """Physical and derived-indicator results of one basin run.

    Outcome-derived terms (used by payoffs.py, never the package directly):
    firm_hydropower_gwh_per_year, storage_utilization_fraction,
    irrigation_reliability_fraction, low_season_volume_mcm, shortfall_fraction.
    """

    upstream_release: pd.Series
    upstream_storage: pd.Series
    tributary_inflow: pd.Series
    inflow_to_b: pd.Series
    delivered_to_b: pd.Series
    b_storage: pd.Series | None

    firm_hydropower_gwh_per_year: float
    storage_utilization_fraction: float
    irrigation_reliability_fraction: float
    low_season_volume_mcm: float
    shortfall_fraction: float


class FlowModel(Protocol):
    """Interface a surrogate of an external hydrological model (e.g. a
    trained emulator of CWatM or MESSAGEix output) could implement in place
    of the internal model, without callers changing."""

    def run(self, package: Package, trace: pd.Series) -> BasinOutputs: ...


@dataclass
class BasinConfig:
    tributary_inflow_fraction: float
    travel_time_days: float
    muskingum_x: float
    upstream_template: ReservoirConfig
    downstream_template: ReservoirConfig
    b_irrigation_annual_mcm: float
    b_irrigation_shape: dict[int, float]
    low_flow_months: list[int]
    issues_by_name: dict[str, Issue] = field(default_factory=dict)

    @classmethod
    def from_scenario(cls, cfg: ScenarioConfig, issues: list[Issue]) -> "BasinConfig":
        return cls(
            tributary_inflow_fraction=cfg.hydrology.tributary_inflow_fraction,
            travel_time_days=cfg.hydrology.travel_time_days,
            muskingum_x=cfg.hydrology.muskingum_x,
            upstream_template=cfg.basin.reservoirs.upstream,
            downstream_template=cfg.basin.reservoirs.downstream,
            b_irrigation_annual_mcm=cfg.basin.demand.b_irrigation_annual_mcm,
            b_irrigation_shape=cfg.basin.demand.b_irrigation_shape,
            low_flow_months=cfg.time.seasons[cfg.time.low_flow_season],
            issues_by_name={issue.name: issue for issue in issues},
        )


def route_muskingum(inflow: pd.Series, travel_time_days: float, x: float, days_per_step: float = 30.0) -> pd.Series:
    """Muskingum channel routing. `travel_time_days` = 0 reduces to identity
    (no lag/attenuation), so the interface is always exercised even when the
    default scenario configures zero lag.

    Args:
        inflow: pd.Series of inbound volumes per step (MCM/step).
        travel_time_days: reach travel time K, days.
        x: Muskingum weighting factor, 0-0.5.
        days_per_step: length of one series step in days (30 for monthly).

    Returns:
        pd.Series of routed outflow, same index as `inflow`, MCM/step.
    """
    if travel_time_days <= 0:
        return inflow.copy()

    k = travel_time_days
    dt = days_per_step
    denom = 2 * k * (1 - x) + dt
    c0 = (dt - 2 * k * x) / denom
    c1 = (dt + 2 * k * x) / denom
    c2 = (2 * k * (1 - x) - dt) / denom

    values = inflow.values
    out = np.zeros(len(values))
    out[0] = values[0]
    for t in range(1, len(values)):
        out[t] = c0 * values[t] + c1 * values[t - 1] + c2 * out[t - 1]
    out = np.clip(out, 0.0, None)
    return pd.Series(out, index=inflow.index)


def _demand_target_series(shape: dict[int, float], annual_mcm: float, index: pd.PeriodIndex) -> pd.Series:
    total = sum(shape.values())
    monthly_target = {m: (shape[m] / total) * annual_mcm for m in shape}
    return pd.Series([monthly_target[int(m)] for m in index.month], index=index)


def run_basin(config: BasinConfig, package: Package, flow_trace: pd.Series) -> BasinOutputs:
    """Simulate one basin run for a candidate package.

    Args:
        config: BasinConfig (built once per scenario via `from_scenario`).
        package: whole candidate agreement (must cover every configured issue).
        flow_trace: pd.Series of upstream inflow at Country A's dam site,
            MCM/month, indexed by a monthly pandas PeriodIndex.

    Returns:
        BasinOutputs with delivery series and outcome-derived indicators.
    """
    storage_params = package.level_params(config.issues_by_name, "upstream_storage_capacity")
    live_storage = float(storage_params["live_storage_mcm"])

    window_params = package.level_params(config.issues_by_name, "filling_window")
    fill_months = [int(m) for m in window_params["months"]]
    release_months = [m for m in range(1, 13) if m not in fill_months]

    release_params = package.level_params(config.issues_by_name, "min_release_guarantee")
    min_release = float(release_params["min_release_mcm_per_month"])

    upstream = Reservoir(
        name=config.upstream_template.name,
        live_storage=live_storage,
        min_release=min_release,
        fill_months=fill_months if live_storage > 0 else config.upstream_template.fill_months,
        release_months=release_months if live_storage > 0 else config.upstream_template.release_months,
        fill_fraction=config.upstream_template.fill_fraction,
        initial_storage=config.upstream_template.initial_storage_fraction * live_storage,
        spin_up_years=config.upstream_template.spin_up_years,
    )
    upstream_result = simulate(upstream, flow_trace)

    tributary_inflow = flow_trace * config.tributary_inflow_fraction
    routed_release = route_muskingum(upstream_result.release, config.travel_time_days, config.muskingum_x)
    inflow_to_b = routed_release + tributary_inflow

    b_storage = None
    if config.downstream_template.enabled:
        downstream = Reservoir(
            name=config.downstream_template.name,
            live_storage=float(config.downstream_template.live_storage_mcm or 0.0),
            min_release=config.downstream_template.min_release_mcm_per_month,
            fill_months=config.downstream_template.fill_months,
            release_months=config.downstream_template.release_months,
            fill_fraction=config.downstream_template.fill_fraction,
            initial_storage=config.downstream_template.initial_storage_fraction
            * float(config.downstream_template.live_storage_mcm or 0.0),
            spin_up_years=config.downstream_template.spin_up_years,
        )
        downstream_result = simulate(downstream, inflow_to_b)
        delivered_to_b = downstream_result.release
        b_storage = downstream_result.storage
    else:
        delivered_to_b = inflow_to_b

    demand_target = _demand_target_series(config.b_irrigation_shape, config.b_irrigation_annual_mcm, delivered_to_b.index)
    reliability_ratio = np.minimum(1.0, delivered_to_b / demand_target)
    shortfall_ratio = np.maximum(0.0, (demand_target - delivered_to_b) / demand_target)
    irrigation_reliability_fraction = float(reliability_ratio.mean())
    shortfall_fraction = float(shortfall_ratio.mean())

    low_season_mask = delivered_to_b.index.month.isin(config.low_flow_months)
    n_years = len(delivered_to_b) / 12.0
    low_season_volume_mcm = float(delivered_to_b[low_season_mask].sum() / n_years) if n_years > 0 else 0.0

    if live_storage > 0:
        head_factor = 0.3 + 0.7 * min(live_storage / 15000.0, 1.0)
        k_energy = 0.01  # illustrative GWh per MCM at full head; see LIMITATIONS.md
        firm_monthly_release = float(upstream_result.release.min())
        firm_hydropower_gwh_per_year = k_energy * head_factor * 12.0 * firm_monthly_release
        effective_capacity = config.upstream_template.fill_fraction * live_storage
        storage_utilization_fraction = float(upstream_result.storage.mean() / effective_capacity)
    else:
        firm_hydropower_gwh_per_year = 0.0
        storage_utilization_fraction = 0.0

    return BasinOutputs(
        upstream_release=upstream_result.release,
        upstream_storage=upstream_result.storage,
        tributary_inflow=tributary_inflow,
        inflow_to_b=inflow_to_b,
        delivered_to_b=delivered_to_b,
        b_storage=b_storage,
        firm_hydropower_gwh_per_year=firm_hydropower_gwh_per_year,
        storage_utilization_fraction=storage_utilization_fraction,
        irrigation_reliability_fraction=irrigation_reliability_fraction,
        low_season_volume_mcm=low_season_volume_mcm,
        shortfall_fraction=shortfall_fraction,
    )
