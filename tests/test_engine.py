import pytest

from riparia.climate import BASELINE
from riparia.climate import generate_climate_ensemble
from riparia.config_schema import load_config
from riparia.contingent import ContingentRule
from riparia.engine import Exercise, ExerciseStatus, Phase
from riparia.issues import Package
from riparia.payoffs import batna

CONFIG_PATH = "src/riparia/config/generic_basin_v1.yaml"

FULL_PACKAGE = {
    "upstream_storage_capacity": "medium",
    "filling_window": "shoulder",
    "allocation_mechanism": "zonal_formula",
    "data_exchange": "weekly",
    "flood_early_warning": "seasonal",
    "financing_transfer": "none",
}


def _start_exercise() -> Exercise:
    cfg = load_config(CONFIG_PATH)
    return Exercise.start(cfg, n_years=3)


def test_start_creates_round_one_in_joint_fact_finding():
    ex = _start_exercise()
    assert len(ex.rounds) == 1
    assert ex.current_round.phase == Phase.JOINT_FACT_FINDING
    assert ex.status == ExerciseStatus.IN_PROGRESS
    assert len(ex.log.events) == 2  # round_started + exercise_started


def test_require_whole_packages_rejects_partial_offer():
    ex = _start_exercise()
    partial = Package(selections={"upstream_storage_capacity": "medium"})
    with pytest.raises(ValueError):
        ex.submit_offer("A", partial)


def test_require_whole_packages_false_allows_partial_offer():
    # A partial offer is recorded but can't be run through the physical
    # model (it doesn't specify every issue), so it's unscored.
    cfg = load_config(CONFIG_PATH)
    ex = Exercise.start(cfg, require_whole_packages=False)
    partial = Package(selections={"upstream_storage_capacity": "medium"})
    scores = ex.submit_offer("A", partial)
    assert scores is None
    assert ex.current_round.offers["A"].selections == partial.selections


def test_submit_offer_records_move_type_initial_then_classified():
    ex = _start_exercise()
    package_a = Package(selections=FULL_PACKAGE)
    ex.submit_offer("A", package_a)
    first_offer_event = [e for e in ex.log.events if e.event_type == "offer"][0]
    assert first_offer_event.move_type == "initial"

    package_b = Package(selections={**FULL_PACKAGE, "financing_transfer": "b_pays_a"})
    ex.submit_offer("B", package_b)
    second_offer_event = [e for e in ex.log.events if e.event_type == "offer"][1]
    assert second_offer_event.move_type in {"integrative", "distributive", "value_destroying"}


def test_single_negotiating_text_criticism_and_revision():
    ex = _start_exercise()
    ex.new_round(Phase.SINGLE_TEXT_REVISION)
    ex.submit_criticism("A", "too much storage for B's comfort")
    ex.submit_criticism("B", "release guarantee too weak")
    revised = Package(selections={**FULL_PACKAGE, "allocation_mechanism": "fixed_high"})
    ex.revise_text(revised)
    assert ex.current_text.selections == revised.selections
    assert ex.current_round.criticisms == {
        "A": "too much storage for B's comfort",
        "B": "release guarantee too weak",
    }


def test_single_negotiating_text_disabled_rejects_criticism():
    cfg = load_config(CONFIG_PATH)
    ex = Exercise.start(cfg, single_negotiating_text=False)
    with pytest.raises(ValueError):
        ex.submit_criticism("A", "no")


def test_settle_freezes_agreement_and_runs_frontier_analysis():
    ex = _start_exercise()
    package = Package(selections=FULL_PACKAGE)
    settlement = ex.settle(package)
    assert ex.status == ExerciseStatus.SETTLED
    assert ex.settlement is settlement
    assert settlement.package.selections == FULL_PACKAGE
    assert set(settlement.scores.keys()) == {"A", "B"}
    assert len(settlement.frontier) > 0
    assert settlement.efficiency_loss.joint_score_shortfall >= -1e-6
    assert settlement.nash_point is not None
    assert settlement.kalai_smorodinsky_point is not None
    settle_events = [e for e in ex.log.events if e.event_type == "settlement"]
    assert len(settle_events) == 1


def test_settle_with_contingent_comparison():
    cfg = load_config(CONFIG_PATH)
    ex = Exercise.start(cfg, n_years=3)
    from riparia.hydrology import generate_climatology

    monthly = generate_climatology(cfg.hydrology.seasonal_shape, cfg.hydrology.mean_annual_flow_mcm)
    ensemble = generate_climate_ensemble(monthly, BASELINE, n_years=10, drought_prob=0.2, wet_prob=0.2, seed=1)
    package = Package(selections=FULL_PACKAGE)
    batnas = {p: batna(ex.value_functions, p) for p in ex.value_functions}
    rule = ContingentRule(
        name="contingent",
        packages_by_state={"dry": package, "normal": package, "wet": package},
        basin_config=ex.basin_config,
        value_functions=ex.value_functions,
        batnas=batnas,
    )
    settlement = ex.settle(package, ensemble=ensemble, contingent_rule=rule)
    assert settlement.contingent_comparison is not None
    assert "fixed" in settlement.contingent_comparison and "contingent" in settlement.contingent_comparison


def test_offer_history_tracks_scores_and_zopa_status():
    ex = _start_exercise()
    a_opening = Package(selections={**FULL_PACKAGE, "allocation_mechanism": "fixed_low"})
    b_opening = Package(selections={**FULL_PACKAGE, "allocation_mechanism": "fixed_high"})
    ex.submit_offer("A", a_opening)
    ex.submit_offer("B", b_opening)
    history = ex.offer_history()
    assert len(history) == 2
    assert set(history["party"]) == {"A", "B"}
    assert {"score_A", "score_B", "in_zopa", "move_type", "round_number"} <= set(history.columns)
    assert history["move_type"].iloc[0] == "initial"


def test_offer_history_omits_partial_unscored_offers():
    cfg = load_config(CONFIG_PATH)
    ex = Exercise.start(cfg, require_whole_packages=False)
    ex.submit_offer("A", Package(selections={"upstream_storage_capacity": "medium"}))
    assert ex.offer_history().empty


def test_declare_impasse_ends_exercise_without_settlement():
    ex = _start_exercise()
    ex.submit_offer("A", Package(selections=FULL_PACKAGE))
    ex.declare_impasse("Country B walked away over the release guarantee.")
    assert ex.status == ExerciseStatus.IMPASSE
    assert ex.settlement is None
    impasse_events = [e for e in ex.log.events if e.event_type == "impasse"]
    assert len(impasse_events) == 1
    assert impasse_events[0].payload["reason"] == "Country B walked away over the release guarantee."


def test_no_further_moves_after_settlement():
    ex = _start_exercise()
    ex.settle(Package(selections=FULL_PACKAGE))
    with pytest.raises(ValueError):
        ex.submit_offer("A", Package(selections=FULL_PACKAGE))
    with pytest.raises(ValueError):
        ex.settle(Package(selections=FULL_PACKAGE))


def test_no_further_moves_after_impasse():
    ex = _start_exercise()
    ex.declare_impasse("no ZOPA found")
    with pytest.raises(ValueError):
        ex.submit_offer("A", Package(selections=FULL_PACKAGE))
    with pytest.raises(ValueError):
        ex.declare_impasse("again")
