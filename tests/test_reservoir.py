import numpy as np
import pandas as pd
import pytest

from riparia.reservoir import Reservoir, _ordered_release_months, simulate


def _monthly_inflow(n_years: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    shape = np.array([0.03, 0.028, 0.035, 0.055, 0.075, 0.11, 0.17, 0.18, 0.13, 0.075, 0.045, 0.033])
    base = shape / shape.sum() * 175_000
    values = np.tile(base, n_years) * rng.uniform(0.9, 1.1, size=12 * n_years)
    index = pd.period_range("2000-01", periods=12 * n_years, freq="M")
    return pd.Series(values, index=index)


def test_ordered_release_months_wraps_calendar():
    order = _ordered_release_months(fill_months=[7, 8], release_months=[1, 2, 3, 4, 5, 6, 9, 10, 11, 12])
    assert order == [9, 10, 11, 12, 1, 2, 3, 4, 5, 6]


def test_reservoir_mass_balance_conserves_volume():
    reservoir = Reservoir(
        name="test",
        live_storage=8000,
        min_release=300,
        fill_months=[7, 8],
        release_months=[1, 2, 3, 4, 5, 6, 9, 10, 11, 12],
        fill_fraction=0.9,
        initial_storage=4000,
        spin_up_years=5,
    )
    inflow = _monthly_inflow(n_years=6)
    result = simulate(reservoir, inflow, timestep_months=1)

    balance = result.storage_start + inflow.sum() - result.release.sum()
    assert balance == pytest.approx(result.storage.iloc[-1], abs=1e-6)


def test_reservoir_storage_bounded():
    reservoir = Reservoir(
        name="test",
        live_storage=8000,
        min_release=300,
        fill_months=[7, 8],
        release_months=[1, 2, 3, 4, 5, 6, 9, 10, 11, 12],
        fill_fraction=0.9,
        initial_storage=0,
        spin_up_years=3,
    )
    inflow = _monthly_inflow(n_years=4)
    result = simulate(reservoir, inflow, timestep_months=1)
    assert (result.storage >= -1e-9).all()
    assert (result.storage <= reservoir.live_storage + 1e-9).all()


def test_reservoir_release_never_negative():
    reservoir = Reservoir(
        name="test",
        live_storage=8000,
        min_release=300,
        fill_months=[7, 8],
        release_months=[1, 2, 3, 4, 5, 6, 9, 10, 11, 12],
        fill_fraction=0.9,
        initial_storage=1000,
        spin_up_years=2,
    )
    inflow = _monthly_inflow(n_years=3)
    result = simulate(reservoir, inflow, timestep_months=1)
    assert (result.release >= -1e-9).all()


def test_reservoir_converges_to_repeating_cycle_after_spin_up():
    reservoir = Reservoir(
        name="test",
        live_storage=8000,
        min_release=300,
        fill_months=[7, 8],
        release_months=[1, 2, 3, 4, 5, 6, 9, 10, 11, 12],
        fill_fraction=0.9,
        initial_storage=0,
        spin_up_years=8,
    )
    base_year = _monthly_inflow(n_years=1, seed=5)
    inflow = pd.Series(
        np.tile(base_year.values, 3),
        index=pd.period_range("2000-01", periods=36, freq="M"),
    )
    result = simulate(reservoir, inflow, timestep_months=1)
    year1 = result.storage.iloc[0:12].values
    year2 = result.storage.iloc[12:24].values
    assert np.allclose(year1, year2, atol=1.0)


def test_zero_capacity_reservoir_passes_inflow_through():
    reservoir = Reservoir(
        name="test",
        live_storage=0,
        min_release=0,
        fill_months=[7, 8],
        release_months=[1, 2, 3, 4, 5, 6, 9, 10, 11, 12],
        fill_fraction=0.9,
        initial_storage=0,
        spin_up_years=2,
    )
    inflow = _monthly_inflow(n_years=2)
    result = simulate(reservoir, inflow, timestep_months=1)
    pd.testing.assert_series_equal(result.release, inflow, check_names=False)
