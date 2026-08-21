import pandas as pd
import pytest

from riparia.config_schema import load_config
from riparia.hydrology import generate_climatology
from riparia.issues import enumerate_packages, load_issues
from riparia.payoffs import ScoredPackage, batna, load_value_functions, score, score_all_packages, zopa
from riparia.system import BasinConfig

CONFIG_PATH = "src/riparia/config/indus_style_v1.yaml"


def _fixture(n_years: int = 3):
    cfg = load_config(CONFIG_PATH)
    issues = load_issues(cfg.issues)
    basin_config = BasinConfig.from_scenario(cfg, issues)
    monthly = generate_climatology(cfg.hydrology.seasonal_shape, cfg.hydrology.mean_annual_flow_mcm)
    index = pd.period_range("2000-01", periods=12 * n_years, freq="M")
    trace = pd.Series(list(monthly.values) * n_years, index=index)
    value_functions = load_value_functions(cfg)
    packages = enumerate_packages(issues, cfg.issues.forbidden_combinations)
    return cfg, issues, basin_config, trace, value_functions, packages


def test_weights_sum_to_one_for_each_party():
    cfg, *_ = _fixture()
    for party, vf in load_value_functions(cfg).items():
        assert sum(vf.weights.values()) == pytest.approx(1.0)


def test_score_bounded_0_100():
    cfg, issues, basin_config, trace, value_functions, packages = _fixture()
    sample = packages[:: max(1, len(packages) // 50)]
    scored = score_all_packages(sample, basin_config, trace, value_functions)
    for sp in scored:
        for party, s in sp.scores.items():
            assert 0.0 <= s <= 100.0


def test_batna_reads_from_config():
    cfg, *_ = _fixture()
    value_functions = load_value_functions(cfg)
    assert batna(value_functions, "A") == cfg.parties["A"].batna
    assert batna(value_functions, "B") == cfg.parties["B"].batna


def test_zopa_nonempty_for_default_scenario():
    cfg, issues, basin_config, trace, value_functions, packages = _fixture()
    scored = score_all_packages(packages, basin_config, trace, value_functions)
    batnas = {p: batna(value_functions, p) for p in value_functions}
    in_zopa = zopa(scored, batnas)
    assert len(in_zopa) > 0


def test_zopa_raises_when_batna_unreachable():
    cfg, issues, basin_config, trace, value_functions, packages = _fixture()
    sample = packages[:20]
    scored = score_all_packages(sample, basin_config, trace, value_functions)
    with pytest.raises(ValueError, match="ZOPA"):
        zopa(scored, {"A": 999.0, "B": 999.0})
