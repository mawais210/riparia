import numpy as np
import pandas as pd
import pytest

from riparia.climate import BASELINE, ClimateTrend, apply_climate_trend, generate_climate_ensemble
from riparia.hydrology import generate_climatology

SHAPE = {
    1: 0.030, 2: 0.028, 3: 0.035, 4: 0.055, 5: 0.075, 6: 0.110,
    7: 0.170, 8: 0.180, 9: 0.130, 10: 0.075, 11: 0.045, 12: 0.033,
}


def test_baseline_trend_is_identity():
    monthly = generate_climatology(SHAPE, mean_flow=175_000)
    trended = apply_climate_trend(monthly, BASELINE)
    pd.testing.assert_series_equal(trended, monthly)


def test_mean_flow_multiplier_scales_annual_total():
    monthly = generate_climatology(SHAPE, mean_flow=175_000)
    trend = ClimateTrend(name="drier", mean_flow_multiplier=0.8, timing_shift_months=0.0)
    trended = apply_climate_trend(monthly, trend)
    assert trended.sum() == pytest.approx(175_000 * 0.8)


def test_timing_shift_preserves_annual_total_approximately():
    monthly = generate_climatology(SHAPE, mean_flow=175_000)
    trend = ClimateTrend(name="earlier_snowmelt", mean_flow_multiplier=1.0, timing_shift_months=-1.0)
    trended = apply_climate_trend(monthly, trend)
    assert trended.sum() == pytest.approx(175_000, rel=0.01)
    # peak month should shift earlier
    assert trended.idxmax() != monthly.idxmax() or trended[monthly.idxmax() - 1] > monthly[monthly.idxmax() - 1]


def test_generate_climate_ensemble_deterministic():
    monthly = generate_climatology(SHAPE, mean_flow=175_000)
    trend = ClimateTrend(name="t", mean_flow_multiplier=0.9, timing_shift_months=-0.5, variance_multiplier=1.2)
    e1 = generate_climate_ensemble(monthly, trend, n_years=10, drought_prob=0.2, wet_prob=0.2, seed=5)
    e2 = generate_climate_ensemble(monthly, trend, n_years=10, drought_prob=0.2, wet_prob=0.2, seed=5)
    pd.testing.assert_frame_equal(e1, e2)


def test_drought_sequence_injection_produces_consecutive_tagged_years():
    monthly = generate_climatology(SHAPE, mean_flow=175_000)
    ensemble = generate_climate_ensemble(
        monthly, BASELINE, n_years=40, drought_prob=0.15, wet_prob=0.15, seed=3,
        drought_sequence_years=3, drought_sequence_prob=0.5, drought_sequence_severity=0.6,
    )
    tags_by_year = ensemble.groupby("year")["tag"].first()
    assert (tags_by_year == "drought_sequence").any(), "expected at least one injected drought sequence"

    sequence_years = sorted(tags_by_year[tags_by_year == "drought_sequence"].index)
    # every run of drought_sequence years should be a block of length >= drought_sequence_years=3,
    # and consecutive integers
    runs = []
    current = [sequence_years[0]]
    for y in sequence_years[1:]:
        if y == current[-1] + 1:
            current.append(y)
        else:
            runs.append(current)
            current = [y]
    runs.append(current)
    for run in runs:
        assert len(run) % 3 == 0 or len(run) >= 3


def test_drought_sequence_years_have_reduced_flow():
    monthly = generate_climatology(SHAPE, mean_flow=175_000)
    ensemble = generate_climate_ensemble(
        monthly, BASELINE, n_years=30, drought_prob=0.1, wet_prob=0.1, seed=11,
        drought_sequence_years=3, drought_sequence_prob=0.8, drought_sequence_severity=0.5,
    )
    normal_mean = ensemble.loc[ensemble.tag == "normal", "flow_mcm"].mean()
    drought_seq_mean = ensemble.loc[ensemble.tag == "drought_sequence", "flow_mcm"].mean()
    assert drought_seq_mean < normal_mean
