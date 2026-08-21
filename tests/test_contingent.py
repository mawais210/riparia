import pandas as pd
import pytest

from riparia.config_schema import load_config
from riparia.contingent import ContingentRule, compare_fixed_vs_contingent, evaluate_contingent
from riparia.hydrology import generate_climatology, generate_ensemble
from riparia.issues import Package, load_issues
from riparia.payoffs import batna, load_value_functions
from riparia.system import BasinConfig

CONFIG_PATH = "src/riparia/config/indus_style_v1.yaml"


def _fixture(n_years: int = 15):
    cfg = load_config(CONFIG_PATH)
    issues = load_issues(cfg.issues)
    basin_config = BasinConfig.from_scenario(cfg, issues)
    monthly = generate_climatology(cfg.hydrology.seasonal_shape, cfg.hydrology.mean_annual_flow_mcm)
    ensemble = generate_ensemble(
        monthly, n_years=n_years, drought_prob=0.3, wet_prob=0.2, seed=1, dry_scale=0.6, wet_scale=1.3
    )
    value_functions = load_value_functions(cfg)
    batnas = {p: batna(value_functions, p) for p in value_functions}

    low_release = Package(
        selections={
            "upstream_storage_capacity": "medium",
            "filling_window": "shoulder",
            "allocation_mechanism": "fixed_low",
            "data_exchange": "monthly",
            "flood_early_warning": "seasonal",
            "financing_transfer": "none",
        }
    )
    high_release = Package(selections={**low_release.selections, "allocation_mechanism": "fixed_high"})

    rule = ContingentRule(
        name="contingent",
        packages_by_state={"dry": high_release, "normal": low_release, "wet": low_release},
        basin_config=basin_config,
        value_functions=value_functions,
        batnas=batnas,
    )
    return rule, ensemble, low_release


def test_evaluate_contingent_covers_all_years_and_reports_percentiles():
    rule, ensemble, _ = _fixture()
    result = evaluate_contingent(rule, ensemble)
    n_years = len(ensemble.index.get_level_values("year").unique())
    assert len(result.raw) == n_years
    for party in rule.value_functions:
        assert set(result.percentiles[party].keys()) == {0.10, 0.25, 0.50, 0.75, 0.90}
        assert 0.0 <= result.frac_below_batna[party] <= 1.0


def test_compare_fixed_vs_contingent_reduces_batna_shortfall_in_dry_years():
    rule, ensemble, fixed_package = _fixture()
    comparison = compare_fixed_vs_contingent(fixed_package, rule, ensemble)
    fixed_eval, contingent_eval = comparison["fixed"], comparison["contingent"]
    # B benefits from the high-release guarantee kicking in during dry years.
    assert contingent_eval.frac_below_batna["B"] <= fixed_eval.frac_below_batna["B"] + 1e-9
