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

from riparia.config_schema import AgricultureConfig, ReservoirConfig, ScenarioConfig
from riparia.economics import CropProfile, agricultural_outcomes, water_requirement_mcm
from riparia.issues import Issue, Package
from riparia.reservoir import FloorFn, Reservoir, simulate

# Custom Functions


@dataclass
class BasinOutputs:
    """Physical and derived-indicator results of one basin run.

    Outcome-derived terms (used by payoffs.py, never the package directly):
    firm_hydropower_gwh_per_year, storage_utilization_fraction,
    irrigation_reliability_fraction, low_season_volume_mcm, shortfall_fraction,
    a_agricultural_income_musd, a_agricultural_labor_persondays,
    b_agricultural_income_musd, b_agricultural_labor_persondays.
    """

    upstream_release: pd.Series
    upstream_storage: pd.Series
    a_consumptive_diversion: pd.Series
    tributary_inflow: pd.Series
    inflow_to_b: pd.Series
    delivered_to_b: pd.Series
    b_storage: pd.Series | None

    firm_hydropower_gwh_per_year: float
    storage_utilization_fraction: float
    irrigation_reliability_fraction: float
    low_season_volume_mcm: float
    shortfall_fraction: float
    a_agricultural_income_musd: float
    a_agricultural_labor_persondays: float
    b_agricultural_income_musd: float
    b_agricultural_labor_persondays: float


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
    b_irrigation_shape: dict[int, float]
    a_consumptive_shape: dict[int, float]
    agriculture: AgricultureConfig
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
            b_irrigation_shape=cfg.basin.demand.b_irrigation_shape,
            a_consumptive_shape=cfg.basin.demand.a_consumptive_shape,
            agriculture=cfg.basin.agriculture,
            low_flow_months=cfg.time.seasons[cfg.time.low_flow_season],
            issues_by_name={issue.name: issue for issue in issues},
        )


def _crop_profile(cfg) -> CropProfile:
    return CropProfile(
        label=cfg.label,
        water_use_mcm_per_1000ha=cfg.water_use_mcm_per_1000ha,
        income_musd_per_1000ha=cfg.income_musd_per_1000ha,
        labor_persondays_per_1000ha=cfg.labor_persondays_per_1000ha,
    )


def _crop_profiles(agriculture: AgricultureConfig) -> dict[str, CropProfile]:
    return {name: _crop_profile(cfg) for name, cfg in agriculture.crops.items()}


def _make_floor_fn(mechanism_params: dict, live_storage: float, fill_fraction: float) -> FloorFn:
    """Build the per-step release-floor function for the package's
    `allocation_mechanism` issue level. See docs/METHODOLOGY.md, "The
    allocation-mechanism issue", for what each mechanism represents."""
    mechanism = mechanism_params["mechanism"]
    effective_capacity = fill_fraction * live_storage

    if mechanism == "fixed_volume":
        floor = float(mechanism_params["min_release_mcm_per_month"])
        return lambda inflow_i, storage_start, m: floor

    if mechanism == "percentage_of_flow":
        share = float(mechanism_params["release_share"])
        return lambda inflow_i, storage_start, m: share * inflow_i

    if mechanism == "zonal_formula":
        threshold = float(mechanism_params["threshold_mcm_per_month"])
        b_share = float(mechanism_params["surplus_share_to_b"])
        return lambda inflow_i, storage_start, m: min(inflow_i, threshold) + b_share * max(0.0, inflow_i - threshold)

    if mechanism == "adaptive_rule_curve":
        base_floor = float(mechanism_params["base_min_release_mcm_per_month"])
        sensitivity = float(mechanism_params["sensitivity"])
        target_fraction = {int(k): float(v) for k, v in mechanism_params["target_storage_fraction"].items()}

        def floor_fn(inflow_i: float, storage_start: float, m: int) -> float:
            if effective_capacity <= 0:
                return base_floor
            frac = storage_start / effective_capacity
            extra = sensitivity * max(0.0, frac - target_fraction[m]) * effective_capacity
            return base_floor + extra

        return floor_fn

    raise ValueError(f"unknown allocation mechanism {mechanism!r}")


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

    mechanism_params = package.level_params(config.issues_by_name, "allocation_mechanism")
    floor_fn = _make_floor_fn(mechanism_params, live_storage, config.upstream_template.fill_fraction)

    upstream = Reservoir(
        name=config.upstream_template.name,
        live_storage=live_storage,
        min_release=floor_fn,
        fill_months=fill_months if live_storage > 0 else config.upstream_template.fill_months,
        release_months=release_months if live_storage > 0 else config.upstream_template.release_months,
        fill_fraction=config.upstream_template.fill_fraction,
        initial_storage=config.upstream_template.initial_storage_fraction * live_storage,
        spin_up_years=config.upstream_template.spin_up_years,
    )
    upstream_result = simulate(upstream, flow_trace)

    crops = _crop_profiles(config.agriculture)
    a_agri_cfg = config.agriculture.parties["A"]
    a_annual_demand = water_requirement_mcm(a_agri_cfg.crop_areas_1000ha, crops)
    a_demand_target = _demand_target_series(config.a_consumptive_shape, a_annual_demand, upstream_result.release.index)
    a_consumptive_diversion = np.minimum(upstream_result.release, a_demand_target)
    net_release = upstream_result.release - a_consumptive_diversion

    tributary_inflow = flow_trace * config.tributary_inflow_fraction
    routed_release = route_muskingum(net_release, config.travel_time_days, config.muskingum_x)
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

    b_agri_cfg = config.agriculture.parties["B"]
    b_annual_demand = water_requirement_mcm(b_agri_cfg.crop_areas_1000ha, crops)
    demand_target = _demand_target_series(config.b_irrigation_shape, b_annual_demand, delivered_to_b.index)
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

    a_ag_outcome = agricultural_outcomes(
        water_available_mcm=float(a_consumptive_diversion.sum() / n_years) if n_years > 0 else 0.0,
        areas_1000ha=a_agri_cfg.crop_areas_1000ha,
        crops=crops,
    )
    b_ag_outcome = agricultural_outcomes(
        water_available_mcm=float(delivered_to_b.sum() / n_years) if n_years > 0 else 0.0,
        areas_1000ha=b_agri_cfg.crop_areas_1000ha,
        crops=crops,
    )

    return BasinOutputs(
        upstream_release=upstream_result.release,
        upstream_storage=upstream_result.storage,
        a_consumptive_diversion=a_consumptive_diversion,
        tributary_inflow=tributary_inflow,
        inflow_to_b=inflow_to_b,
        delivered_to_b=delivered_to_b,
        b_storage=b_storage,
        firm_hydropower_gwh_per_year=firm_hydropower_gwh_per_year,
        storage_utilization_fraction=storage_utilization_fraction,
        irrigation_reliability_fraction=irrigation_reliability_fraction,
        low_season_volume_mcm=low_season_volume_mcm,
        shortfall_fraction=shortfall_fraction,
        a_agricultural_income_musd=a_ag_outcome.income_musd,
        a_agricultural_labor_persondays=a_ag_outcome.labor_persondays,
        b_agricultural_income_musd=b_ag_outcome.income_musd,
        b_agricultural_labor_persondays=b_ag_outcome.labor_persondays,
    )
