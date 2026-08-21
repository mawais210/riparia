"""Scripted four-round negotiation exercise, run end to end.

    python examples/run_full_exercise.py

Round 1 (joint fact-finding) -> round 2 (package bidding, each party's
opening offer) -> round 3 (single negotiating text: criticism + facilitator
revision) -> settlement, with the automatic frontier/PSS/contingent-vs-fixed
analysis `Exercise.settle()` runs. Prints the agreed package, both parties'
scores, position relative to the frontier, efficiency loss, the top five
post-settlement-settlement packages missed, and the fixed-vs-contingent
dry-year stress comparison.
"""

# Import necessary libraries
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from riparia.climate import BASELINE, generate_climate_ensemble
from riparia.config_schema import load_config
from riparia.contingent import ContingentRule
from riparia.engine import Exercise, Phase
from riparia.hydrology import generate_climatology
from riparia.issues import Package
from riparia.payoffs import batna

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "src" / "riparia" / "config" / "indus_style_v1.yaml")

# Custom Functions


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    ex = Exercise.start(cfg, n_years=3)

    print(f"=== {cfg.name} ===")
    print(cfg.description.strip())
    print()

    # Round 1: joint fact-finding (already the exercise's opening round).
    print(f"--- Round {ex.current_round.number}: {ex.current_round.phase.value} ---")
    print("Both parties jointly simulate a baseline package (see app/facilitator.py's Simulate tab "
          "for the interactive version).")
    print()

    # Round 2: package bidding -- each party's opening offer.
    ex.new_round(Phase.PACKAGE_BIDDING)
    print(f"--- Round {ex.current_round.number}: {ex.current_round.phase.value} ---")
    a_opening = Package(selections={
        "upstream_storage_capacity": "large",
        "filling_window": "unrestricted",
        "allocation_mechanism": "fixed_low",
        "data_exchange": "none",
        "flood_early_warning": "none",
        "financing_transfer": "a_pays_b",
    })
    b_opening = Package(selections={
        "upstream_storage_capacity": "none",
        "filling_window": "wet_season",
        "allocation_mechanism": "fixed_high",
        "data_exchange": "daily_verified",
        "flood_early_warning": "real_time",
        "financing_transfer": "b_pays_a",
    })
    scores_a = ex.submit_offer("A", a_opening)
    scores_b = ex.submit_offer("B", b_opening)
    print(f"A's opening offer scores: A={scores_a['A']:.1f}, B={scores_a['B']:.1f}")
    print(f"B's opening offer scores: A={scores_b['A']:.1f}, B={scores_b['B']:.1f}")
    print("(As expected, each party's own opening offer scores far better for itself than the other.)")
    print()

    # Round 3: single negotiating text -- facilitator drafts, parties criticize, facilitator revises.
    ex.new_round(Phase.SINGLE_TEXT_REVISION)
    print(f"--- Round {ex.current_round.number}: {ex.current_round.phase.value} ---")
    draft = Package(selections={
        "upstream_storage_capacity": "medium",
        "filling_window": "shoulder",
        "allocation_mechanism": "zonal_formula",
        "data_exchange": "weekly",
        "flood_early_warning": "seasonal",
        "financing_transfer": "none",
    })
    ex.revise_text(draft, facilitator_notes="Facilitator's first draft, splitting the difference on storage/window.")
    ex.submit_criticism("A", "Storage still smaller than we'd like, but the mechanism protects our flexibility.")
    ex.submit_criticism("B", "Would prefer a stronger release guarantee than zonal_formula gives us.")
    revised = Package(selections={**draft.selections, "allocation_mechanism": "fixed_high", "financing_transfer": "b_pays_a"})
    ex.revise_text(revised, facilitator_notes="Strengthened B's release guarantee; B compensates A via financing.")
    print(f"Final single negotiating text: {revised.selections}")
    print()

    # Round 4: settlement, with automatic frontier / PSS / contingent-vs-fixed analysis.
    print("--- Settlement ---")
    monthly = generate_climatology(cfg.hydrology.seasonal_shape, cfg.hydrology.mean_annual_flow_mcm)
    ensemble = generate_climate_ensemble(monthly, BASELINE, n_years=30, drought_prob=0.2, wet_prob=0.2, seed=7)
    batnas = {p: batna(ex.value_functions, p) for p in ex.value_functions}
    fallback = Package(selections={**revised.selections, "allocation_mechanism": "fixed_low"})
    contingent_rule = ContingentRule(
        name="contingent",
        packages_by_state={"dry": revised, "normal": fallback, "wet": fallback},
        basin_config=ex.basin_config,
        value_functions=ex.value_functions,
        batnas=batnas,
    )
    settlement = ex.settle(revised, ensemble=ensemble, contingent_rule=contingent_rule)

    print(f"Agreed package: {settlement.package.selections}")
    print(f"Scores: A = {settlement.scores['A']:.1f}, B = {settlement.scores['B']:.1f} (0-100 scale)")
    print(f"BATNAs: A = {batnas['A']:.1f}, B = {batnas['B']:.1f}")
    print()

    print(f"Pareto frontier has {len(settlement.frontier)} distinct points.")
    print(f"Joint-score shortfall vs. best joint-score frontier point: {settlement.efficiency_loss.joint_score_shortfall:.1f}")
    print(f"Foregone gain at nearest frontier point: {settlement.efficiency_loss.foregone_gain}")
    print(f"Nash point: A={settlement.nash_point.scores['A']:.1f}, B={settlement.nash_point.scores['B']:.1f}")
    print(f"Kalai-Smorodinsky point: A={settlement.kalai_smorodinsky_point.scores['A']:.1f}, "
          f"B={settlement.kalai_smorodinsky_point.scores['B']:.1f}")
    print()

    print(f"Post-settlement settlement: {len(settlement.pss)} packages both parties would strictly prefer.")
    for sp in settlement.pss[:5]:
        print(f"  A={sp.scores['A']:.1f} B={sp.scores['B']:.1f}  {sp.package.selections}")
    if not settlement.pss:
        print("  (none -- the agreement was already on the Pareto frontier)")
    print()

    print("Fixed vs. contingent: fraction of ensemble years below BATNA")
    fixed_eval = settlement.contingent_comparison["fixed"]
    contingent_eval = settlement.contingent_comparison["contingent"]
    for party in ex.value_functions:
        print(f"  {party}: fixed = {fixed_eval.frac_below_batna[party]:.0%}, "
              f"contingent = {contingent_eval.frac_below_batna[party]:.0%}")

    print()
    print(f"Event log: {len(ex.log.events)} events across {len(ex.rounds)} rounds.")


if __name__ == "__main__":
    main()
