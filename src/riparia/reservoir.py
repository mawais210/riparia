"""Single-reservoir seasonal operating rule and mass-balance simulation.

Units: storage and inflow/release volumes are in MCM per simulation step
(the model is written generically over "steps"; the default scenario uses
monthly steps, so MCM/month). All quantities are volumes, never rates.
"""

# Import necessary libraries
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Custom Functions


@dataclass
class Reservoir:
    """A single reservoir under a fill-season / release-season operating rule.

    Attributes:
        name: label.
        live_storage: nominal live storage capacity, MCM.
        min_release: floor release enforced in every step, MCM/step.
        fill_months: calendar months (1-12) in which the reservoir retains
            inflow above `min_release` rather than actively drawing down.
        release_months: calendar months (1-12) in which the reservoir draws
            down stored water on top of `min_release`. Must, together with
            `fill_months`, form one contiguous wrap-around partition of the
            12-month calendar (i.e. every month is in exactly one of the two).
        fill_fraction: usable fraction of `live_storage` (0-1]; the spill
            threshold is `fill_fraction * live_storage`, leaving the rest as
            an unused flood buffer.
        initial_storage: storage, MCM, at the very start of simulation
            (before any spin-up years are run).
        spin_up_years: number of extra warm-up annual cycles run (by
            repeating the first 12 steps of the supplied inflow) before the
            reported window, so the annual storage cycle has converged.
    """

    name: str
    live_storage: float
    min_release: float
    fill_months: list[int]
    release_months: list[int]
    fill_fraction: float
    initial_storage: float
    spin_up_years: int = 3


@dataclass
class SimulationResult:
    release: pd.Series
    storage: pd.Series
    storage_start: float


def _ordered_release_months(fill_months: list[int], release_months: list[int]) -> list[int]:
    """Chronological order of `release_months`, starting right after the last
    fill month and wrapping around the calendar. Assumes fill_months and
    release_months partition 1..12."""
    start = (max(fill_months) % 12) + 1
    calendar_order = [((start - 1 + i) % 12) + 1 for i in range(12)]
    release_set = set(release_months)
    return [m for m in calendar_order if m in release_set]


def simulate(reservoir: Reservoir, inflow: pd.Series, timestep_months: int = 1) -> SimulationResult:
    """Run the reservoir mass balance over `inflow`, after an internal spin-up.

    Args:
        reservoir: the Reservoir to simulate.
        inflow: pd.Series of inbound volumes (MCM/step) with a `.index` that
            has a `.month` accessor consistent with calendar months (e.g. a
            pandas PeriodIndex with freq="M"), covering an integer number of
            years starting at the first month of the water year used to
            build it.
        timestep_months: number of calendar months represented by one entry
            of `inflow` (kept generic; the default scenario uses 1).

    Returns:
        SimulationResult with `release` and `storage` (end-of-step storage)
        aligned to `inflow.index`, plus `storage_start` (storage immediately
        before the first returned step, i.e. after spin-up).
    """
    if timestep_months != 1:
        raise NotImplementedError("only monthly timesteps are currently supported")

    months = inflow.index.month

    if reservoir.live_storage <= 0:
        release = inflow.copy()
        storage = pd.Series(0.0, index=inflow.index)
        return SimulationResult(release=release, storage=storage, storage_start=0.0)

    order = _ordered_release_months(reservoir.fill_months, reservoir.release_months)
    remaining_lookup = {m: len(order) - i for i, m in enumerate(order)}
    fill_set = set(reservoir.fill_months)
    effective_capacity = reservoir.fill_fraction * reservoir.live_storage

    spin_up_steps = reservoir.spin_up_years * 12
    if spin_up_steps > 0:
        first_year_inflow = inflow.iloc[:12].values
        first_year_months = months[:12]
        spin_inflow_values = np.tile(first_year_inflow, reservoir.spin_up_years)
        spin_months = np.tile(first_year_months, reservoir.spin_up_years)
    else:
        spin_inflow_values = np.array([])
        spin_months = np.array([], dtype=int)

    all_inflow = np.concatenate([spin_inflow_values, inflow.values])
    all_months = np.concatenate([spin_months, months.values])

    n = len(all_inflow)
    release_out = np.zeros(n)
    storage_out = np.zeros(n)
    storage_prev = reservoir.initial_storage

    for i in range(n):
        m = int(all_months[i])
        storage_pre = storage_prev + all_inflow[i]
        if m in fill_set:
            release_i = reservoir.min_release
        else:
            remaining = remaining_lookup[m]
            release_i = reservoir.min_release + max(0.0, storage_pre - reservoir.min_release) / remaining
            release_i = max(release_i, reservoir.min_release)

        release_i = min(release_i, storage_pre)
        storage_i = storage_pre - release_i

        if storage_i > effective_capacity:
            spill = storage_i - effective_capacity
            release_i += spill
            storage_i = effective_capacity
        storage_i = max(storage_i, 0.0)

        release_out[i] = release_i
        storage_out[i] = storage_i
        storage_prev = storage_i

    storage_start = reservoir.initial_storage if spin_up_steps == 0 else storage_out[spin_up_steps - 1]
    release_recorded = release_out[spin_up_steps:]
    storage_recorded = storage_out[spin_up_steps:]

    release = pd.Series(release_recorded, index=inflow.index)
    storage = pd.Series(storage_recorded, index=inflow.index)
    return SimulationResult(release=release, storage=storage, storage_start=storage_start)
