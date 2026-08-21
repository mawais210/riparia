import pandas as pd
import pytest

from riparia.config_schema import load_config
from riparia.frontier import (
    efficiency_loss,
    kalai_smorodinsky,
    nash_solution,
    pareto_frontier,
    post_settlement_search,
)
from riparia.hydrology import generate_climatology
from riparia.issues import enumerate_packages, load_issues
from riparia.payoffs import batna, load_value_functions, score_all_packages, zopa
from riparia.system import BasinConfig

CONFIG_PATH = "src/riparia/config/indus_style_v1.yaml"


def _scenario_fixture(n_years: int = 3):
    cfg = load_config(CONFIG_PATH)
    issues = load_issues(cfg.issues)
    basin_config = BasinConfig.from_scenario(cfg, issues)
    monthly = generate_climatology(cfg.hydrology.seasonal_shape, cfg.hydrology.mean_annual_flow_mcm)
    index = pd.period_range("2000-01", periods=12 * n_years, freq="M")
    trace = pd.Series(list(monthly.values) * n_years, index=index)
    value_functions = load_value_functions(cfg)
    packages = enumerate_packages(issues, cfg.issues.forbidden_combinations)
    scored = score_all_packages(packages, basin_config, trace, value_functions)
    batnas = {p: batna(value_functions, p) for p in value_functions}
    return scored, batnas


def test_frontier_points_are_non_dominated_brute_force_reduced_space():
    # Reduced, exhaustively checkable issue set: 4 x 4 = 16 packages built
    # directly as (score_A, score_B) pairs, brute-forced independently of
    # pareto_frontier's own internal logic.
    from riparia.payoffs import ScoredPackage
    from riparia.issues import Package

    rng_points = [(a, b) for a in [10, 30, 50, 70] for b in [10, 30, 50, 70]]
    scored = [
        ScoredPackage(package=Package(selections={"x": str(i)}), scores={"A": a, "B": b}, outcomes=None)
        for i, (a, b) in enumerate(rng_points)
    ]
    frontier = pareto_frontier(scored)

    def is_dominated(sp, all_sp):
        for other in all_sp:
            if other is sp:
                continue
            if other.scores["A"] >= sp.scores["A"] and other.scores["B"] >= sp.scores["B"]:
                if other.scores["A"] > sp.scores["A"] or other.scores["B"] > sp.scores["B"]:
                    return True
        return False

    for sp in frontier:
        assert not is_dominated(sp, scored)
    # every non-frontier point IS dominated (brute force confirms frontier is exactly right)
    frontier_ids = {id(sp) for sp in frontier}
    for sp in scored:
        if id(sp) not in frontier_ids:
            assert is_dominated(sp, scored)


def test_pss_empty_when_agreement_already_on_frontier():
    scored, batnas = _scenario_fixture()
    frontier = pareto_frontier(scored)
    agreement = frontier[len(frontier) // 2]
    pss = post_settlement_search(agreement, scored)
    assert pss == []


def test_pss_nonempty_for_dominated_agreement():
    scored, batnas = _scenario_fixture()
    frontier = pareto_frontier(scored)
    frontier_ids = {id(sp) for sp in frontier}
    dominated_candidates = [sp for sp in scored if id(sp) not in frontier_ids]
    assert dominated_candidates, "expected some dominated packages to exist"
    agreement = dominated_candidates[0]
    pss = post_settlement_search(agreement, scored)
    assert len(pss) > 0
    for sp in pss:
        assert sp.scores["A"] > agreement.scores["A"]
        assert sp.scores["B"] > agreement.scores["B"]


def test_default_scenario_zopa_and_frontier_size_smoke_test():
    scored, batnas = _scenario_fixture()
    in_zopa = zopa(scored, batnas)
    assert len(in_zopa) > 0, "default scenario has an empty ZOPA -- scenario design bug"
    frontier = pareto_frontier(in_zopa)
    distinct_points = {(round(sp.scores["A"], 3), round(sp.scores["B"], 3)) for sp in frontier}
    assert len(distinct_points) >= 15, (
        f"frontier has only {len(distinct_points)} distinct non-dominated points -- "
        "scenario is likely degenerate, rebalance issue weights"
    )


def test_efficiency_loss_zero_on_frontier_point():
    scored, batnas = _scenario_fixture()
    frontier = pareto_frontier(scored)
    best_joint = max(frontier, key=lambda sp: sp.scores["A"] + sp.scores["B"])
    loss = efficiency_loss(best_joint, frontier)
    assert loss.joint_score_shortfall == pytest.approx(0.0, abs=1e-9)


def test_nash_and_kalai_smorodinsky_clear_batna():
    scored, batnas = _scenario_fixture()
    frontier = pareto_frontier(zopa(scored, batnas))
    nash_pt = nash_solution(frontier, batnas)
    ks_pt = kalai_smorodinsky(frontier, batnas)
    assert nash_pt.scores["A"] >= batnas["A"] and nash_pt.scores["B"] >= batnas["B"]
    assert ks_pt.scores["A"] >= batnas["A"] and ks_pt.scores["B"] >= batnas["B"]
