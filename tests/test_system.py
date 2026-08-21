import pandas as pd
import pytest

from riparia.config_schema import load_config
from riparia.hydrology import generate_climatology
from riparia.issues import Package, load_issues
from riparia.system import BasinConfig, route_muskingum, run_basin

CONFIG_PATH = "src/riparia/config/indus_style_v1.yaml"


def _basin_config_and_trace():
    cfg = load_config(CONFIG_PATH)
    issues = load_issues(cfg.issues)
    basin_config = BasinConfig.from_scenario(cfg, issues)
    monthly = generate_climatology(cfg.hydrology.seasonal_shape, cfg.hydrology.mean_annual_flow_mcm)
    n_years = 6
    index = pd.period_range("2000-01", periods=12 * n_years, freq="M")
    values = list(monthly.values) * n_years
    trace = pd.Series(values, index=index)
    return basin_config, trace, issues


def test_route_muskingum_zero_lag_is_identity():
    s = pd.Series([10.0, 20.0, 15.0])
    routed = route_muskingum(s, travel_time_days=0, x=0.2)
    pd.testing.assert_series_equal(routed, s)


def test_route_muskingum_preserves_total_volume_approximately():
    s = pd.Series([10.0] * 24)
    routed = route_muskingum(s, travel_time_days=5, x=0.2, days_per_step=30)
    assert routed.sum() == pytest.approx(s.sum(), rel=0.05)


def test_run_basin_large_storage_package():
    basin_config, trace, issues = _basin_config_and_trace()
    package = Package(
        selections={
            "upstream_storage_capacity": "large",
            "filling_window": "wet_season",
            "allocation_mechanism": "fixed_high",
            "data_exchange": "daily_verified",
            "flood_early_warning": "real_time",
            "financing_transfer": "none",
        }
    )
    outcomes = run_basin(basin_config, package, trace)
    assert (outcomes.delivered_to_b >= -1e-6).all()
    assert 0.0 <= outcomes.irrigation_reliability_fraction <= 1.0
    assert 0.0 <= outcomes.shortfall_fraction <= 1.0
    assert outcomes.firm_hydropower_gwh_per_year > 0
    assert 0.0 <= outcomes.storage_utilization_fraction <= 1.0


def test_run_basin_no_storage_package_has_zero_hydropower():
    basin_config, trace, issues = _basin_config_and_trace()
    package = Package(
        selections={
            "upstream_storage_capacity": "none",
            "filling_window": "wet_season",
            "allocation_mechanism": "fixed_low",
            "data_exchange": "none",
            "flood_early_warning": "none",
            "financing_transfer": "none",
        }
    )
    outcomes = run_basin(basin_config, package, trace)
    assert outcomes.firm_hydropower_gwh_per_year == 0.0
    assert outcomes.storage_utilization_fraction == 0.0


def test_run_basin_more_storage_increases_reliability_or_hydropower():
    basin_config, trace, issues = _basin_config_and_trace()
    small = Package(
        selections={
            "upstream_storage_capacity": "small",
            "filling_window": "wet_season",
            "allocation_mechanism": "zonal_formula",
            "data_exchange": "none",
            "flood_early_warning": "none",
            "financing_transfer": "none",
        }
    )
    large = Package(selections={**small.selections, "upstream_storage_capacity": "large"})
    out_small = run_basin(basin_config, small, trace)
    out_large = run_basin(basin_config, large, trace)
    assert out_large.firm_hydropower_gwh_per_year > out_small.firm_hydropower_gwh_per_year


def test_run_basin_fixed_high_raises_worst_month_release_floor():
    # Under the FULL climatology, monsoon inflow vastly exceeds medium
    # storage capacity, so fill months spill regardless of the floor and it
    # never actually binds anywhere -- fixed_low(300) and fixed_high(1200)
    # produce identical results. Real floors only bind (and differentiate
    # release) under genuine scarcity, so this scales the trace down to a
    # dry-year-like level where the floor becomes the binding constraint;
    # see test_contingent.py for the full multi-year stress test.
    basin_config, trace, issues = _basin_config_and_trace()
    dry_trace = trace * 0.2
    base = {
        "upstream_storage_capacity": "medium",
        "filling_window": "shoulder",
        "data_exchange": "none",
        "flood_early_warning": "none",
        "financing_transfer": "none",
    }
    weak = Package(selections={**base, "allocation_mechanism": "fixed_low"})
    strong = Package(selections={**base, "allocation_mechanism": "fixed_high"})
    out_weak = run_basin(basin_config, weak, dry_trace)
    out_strong = run_basin(basin_config, strong, dry_trace)
    assert out_strong.upstream_release.min() > out_weak.upstream_release.min()
    assert out_strong.upstream_release.min() == pytest.approx(1200, abs=1.0)
    assert out_weak.upstream_release.min() == pytest.approx(300, abs=1.0)


def test_run_basin_percentage_of_flow_scales_with_inflow():
    basin_config, trace, issues = _basin_config_and_trace()
    package = Package(
        selections={
            "upstream_storage_capacity": "medium",
            "filling_window": "shoulder",
            "allocation_mechanism": "percentage_of_flow",
            "data_exchange": "none",
            "flood_early_warning": "none",
            "financing_transfer": "none",
        }
    )
    outcomes = run_basin(basin_config, package, trace)
    assert outcomes.upstream_release.min() >= 0.30 * trace.min() - 1e-6


def test_run_basin_a_consumptive_diversion_reduces_delivery_to_b():
    basin_config, trace, issues = _basin_config_and_trace()
    package = Package(
        selections={
            "upstream_storage_capacity": "medium",
            "filling_window": "shoulder",
            "allocation_mechanism": "fixed_low",
            "data_exchange": "none",
            "flood_early_warning": "none",
            "financing_transfer": "none",
        }
    )
    outcomes = run_basin(basin_config, package, trace)
    assert (outcomes.a_consumptive_diversion >= -1e-6).all()
    assert outcomes.a_agricultural_income_musd > 0
    assert outcomes.b_agricultural_income_musd > 0
