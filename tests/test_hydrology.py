import numpy as np
import pandas as pd
import pytest

from riparia.hydrology import (
    aggregate,
    disaggregate_to_daily,
    generate_climatology,
    generate_ensemble,
)

SHAPE = {
    1: 0.030, 2: 0.028, 3: 0.035, 4: 0.055, 5: 0.075, 6: 0.110,
    7: 0.170, 8: 0.180, 9: 0.130, 10: 0.075, 11: 0.045, 12: 0.033,
}


def test_generate_climatology_sums_to_mean_flow():
    monthly = generate_climatology(SHAPE, mean_flow=175_000)
    assert monthly.sum() == pytest.approx(175_000)
    assert list(monthly.index) == list(range(1, 13))


def test_generate_climatology_rejects_missing_month():
    bad_shape = dict(SHAPE)
    del bad_shape[12]
    with pytest.raises(ValueError):
        generate_climatology(bad_shape, mean_flow=100)


def test_disaggregate_preserves_monthly_means():
    monthly = generate_climatology(SHAPE, mean_flow=175_000)
    daily = disaggregate_to_daily(monthly, year=2023, cv_low=0.10, cv_high=0.35, phi=0.6, seed=1)
    for m in range(1, 13):
        mask = daily.index.month == m
        n_days = mask.sum()
        got_mean = daily[mask].mean()
        want_mean = monthly[m] / n_days
        assert got_mean == pytest.approx(want_mean, rel=1e-9)


def test_disaggregate_is_deterministic_given_seed():
    monthly = generate_climatology(SHAPE, mean_flow=175_000)
    d1 = disaggregate_to_daily(monthly, year=2023, cv_low=0.10, cv_high=0.35, phi=0.6, seed=7)
    d2 = disaggregate_to_daily(monthly, year=2023, cv_low=0.10, cv_high=0.35, phi=0.6, seed=7)
    pd.testing.assert_series_equal(d1, d2)


def test_disaggregate_nonnegative():
    monthly = generate_climatology(SHAPE, mean_flow=175_000)
    daily = disaggregate_to_daily(monthly, year=2023, cv_low=0.10, cv_high=0.6, phi=0.8, seed=3)
    assert (daily >= 0).all()


def test_generate_ensemble_tags_and_scaling():
    monthly = generate_climatology(SHAPE, mean_flow=175_000)
    ens = generate_ensemble(monthly, n_years=200, drought_prob=0.2, wet_prob=0.2, seed=42)
    assert set(ens["tag"].unique()) <= {"dry", "normal", "wet"}
    fracs = ens.groupby("tag").apply(lambda g: len(g.index.get_level_values("year").unique())) / 200
    assert fracs["dry"] == pytest.approx(0.2, abs=0.07)
    assert fracs["wet"] == pytest.approx(0.2, abs=0.07)

    annual_totals = ens.groupby(["year", "tag"])["flow_mcm"].sum().reset_index()
    dry_mean = annual_totals.loc[annual_totals.tag == "dry", "flow_mcm"].mean()
    wet_mean = annual_totals.loc[annual_totals.tag == "wet", "flow_mcm"].mean()
    normal_mean = annual_totals.loc[annual_totals.tag == "normal", "flow_mcm"].mean()
    assert dry_mean < normal_mean < wet_mean


def test_generate_ensemble_deterministic():
    monthly = generate_climatology(SHAPE, mean_flow=175_000)
    e1 = generate_ensemble(monthly, n_years=10, drought_prob=0.2, wet_prob=0.2, seed=99)
    e2 = generate_ensemble(monthly, n_years=10, drought_prob=0.2, wet_prob=0.2, seed=99)
    pd.testing.assert_frame_equal(e1, e2)


def test_generate_ensemble_rejects_bad_probs():
    monthly = generate_climatology(SHAPE, mean_flow=175_000)
    with pytest.raises(ValueError):
        generate_ensemble(monthly, n_years=5, drought_prob=0.7, wet_prob=0.6, seed=1)


def test_aggregate_sums_windows():
    s = pd.Series(np.arange(1, 13))
    quarterly = aggregate(s, interval_months=3)
    assert len(quarterly) == 4
    assert quarterly.iloc[0] == sum(range(1, 4))
    assert quarterly.sum() == s.sum()


def test_aggregate_handles_partial_trailing_window():
    s = pd.Series(np.arange(1, 11))
    windows = aggregate(s, interval_months=3)
    assert len(windows) == 4
    assert windows.iloc[-1] == sum([10])
