"""Streamlit facilitator application: one shared screen with five tabs
(Brief, Simulate, Negotiate, Debrief, Methodology). Run with:

    streamlit run app/facilitator.py

Simulate is the joint fact-finding surface -- both parties see the same
physical simulation, with continuous sliders (climate stress, each party's
own crop-mix) alongside the discrete negotiated issues, En-ROADS style: move
a slider, the hydrograph and outcome indicators update immediately. Party
value functions stay hidden until Debrief unlocks at settlement.

The sidebar's scenario picker lists every validated YAML file in
src/riparia/config/ -- this is not Indus-specific; adding another basin is
a matter of dropping in a new scenario file (see docs/METHODOLOGY.md,
section 8), not a code change.
"""

# Import necessary libraries
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from riparia.climate import BASELINE, ClimateTrend, apply_climate_trend, generate_climate_ensemble
from riparia.config_schema import load_config
from riparia.contingent import ContingentRule, compare_fixed_vs_contingent
from riparia.economics import CropProfile, agricultural_outcomes
from riparia.engine import Exercise, ExerciseStatus, Phase
from riparia.frontier import efficiency_loss, kalai_smorodinsky, nash_solution, pareto_frontier, post_settlement_search
from riparia.hydrology import generate_climatology
from riparia.issues import Package
from riparia.payoffs import batna, score
from riparia.system import run_basin

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "src" / "riparia" / "config"

# Custom Functions


def _fmt_number(n: float, decimals: int = 1) -> str:
    """Human-readable K/M/B suffix for a raw count (e.g. person-days,
    GWh), so it never renders as a bare seven-plus-digit number."""
    abs_n = abs(n)
    if abs_n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.{decimals}f}B"
    if abs_n >= 1_000_000:
        return f"{n / 1_000_000:.{decimals}f}M"
    if abs_n >= 1_000:
        return f"{n / 1_000:.{decimals}f}K"
    return f"{n:.{decimals}f}"


def _fmt_musd(musd: float) -> str:
    """Format a value already denominated in MUSD (millions of USD):
    switch to billions above $1000M instead of stacking another K/M/B
    suffix on top of the existing millions unit."""
    if abs(musd) >= 1000:
        return f"${musd / 1000:.2f}B"
    return f"${musd:.0f}M"


def _crop_profile(cfg) -> CropProfile:
    return CropProfile(
        label=cfg.label,
        water_use_mcm_per_1000ha=cfg.water_use_mcm_per_1000ha,
        income_musd_per_1000ha=cfg.income_musd_per_1000ha,
        labor_persondays_per_1000ha=cfg.labor_persondays_per_1000ha,
    )


def _crop_area_editor(ex, party: str, party_label: str, key_prefix: str) -> dict[str, float]:
    """Sliders for one party's planted area of every crop in the scenario's
    crop library, defaulting to the scenario config's areas. A domestic
    policy lever, not a negotiated issue -- see docs/METHODOLOGY.md."""
    ag_cfg = ex.config.basin.agriculture
    party_cfg = ag_cfg.parties[party]
    st.markdown(f"**{party_label}'s cropping pattern** (1000 ha per crop)")
    areas = {}
    cols = st.columns(len(ag_cfg.crops))
    for i, (crop_name, crop_cfg) in enumerate(ag_cfg.crops.items()):
        default_area = party_cfg.crop_areas_1000ha.get(crop_name, 0.0)
        max_area = max(default_area * 2.5, 100.0)
        areas[crop_name] = cols[i].slider(
            crop_name.capitalize(), 0.0, max_area, default_area, max_area / 100,
            key=f"{key_prefix}_{party}_{crop_name}",
        )
    return areas


@st.cache_data
def list_scenarios() -> dict[str, str]:
    """Every validated scenario YAML in CONFIG_DIR, keyed by its display
    name -> file path. New basins appear here automatically once their
    YAML file is added, no code change required."""
    scenarios = {}
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = load_config(str(path))
        scenarios[cfg.name] = str(path)
    return scenarios


@st.cache_resource
def _load_exercise(config_path: str) -> Exercise:
    cfg = load_config(config_path)
    return Exercise.start(cfg, n_years=3)


def _init_state() -> Exercise:
    scenarios = list_scenarios()
    scenario_name = st.sidebar.selectbox("Scenario / basin", list(scenarios.keys()), key="scenario_name")
    scenario_path = scenarios[scenario_name]

    if st.session_state.get("scenario_path") != scenario_path:
        st.session_state.scenario_path = scenario_path
        st.session_state.exercise = _load_exercise(scenario_path)
    elif "exercise" not in st.session_state:
        st.session_state.exercise = _load_exercise(scenario_path)

    return st.session_state.exercise


def _party_names(ex: Exercise) -> dict[str, str]:
    """Real country/party names, typed once in the sidebar, used everywhere
    the UI would otherwise say "Country A"/"Country B". The underlying data
    (Package selections, ValueFunction weights, scores) always keys on the
    fixed "A"/"B" party ids from the scenario config -- only display labels
    change. Defaults come from the scenario's `default_party_names` if it
    sets any (e.g. the Indus Basin scenario suggests "India"/"Pakistan");
    the widget key is scoped to the current scenario path so switching
    scenarios re-seeds fresh defaults instead of carrying over a stale name."""
    defaults = ex.config.default_party_names or {"A": "Country A", "B": "Country B"}
    scenario_path = st.session_state.get("scenario_path", "")
    st.sidebar.markdown("#### Party names")
    name_a = st.sidebar.text_input(
        "Upstream party name", value=defaults["A"], key=f"party_name_A__{scenario_path}"
    )
    name_b = st.sidebar.text_input(
        "Downstream party name", value=defaults["B"], key=f"party_name_B__{scenario_path}"
    )
    return {"A": name_a.strip() or defaults["A"], "B": name_b.strip() or defaults["B"]}


def _package_editor(issues, key_prefix: str, defaults: dict[str, str] | None = None) -> Package:
    defaults = defaults or {}
    cols = st.columns(2)
    selections = {}
    for i, issue in enumerate(issues):
        labels = issue.level_labels()
        default_label = defaults.get(issue.name, labels[0])
        default_idx = labels.index(default_label) if default_label in labels else 0
        selections[issue.name] = cols[i % 2].selectbox(
            issue.description or issue.name, labels, index=default_idx, key=f"{key_prefix}_{issue.name}"
        )
    return Package(selections=selections)


def render_brief(ex: Exercise, names: dict[str, str]) -> None:
    st.header(ex.config.name)
    st.write(ex.config.description)
    st.markdown(f"**Upstream party:** {names['A']} &nbsp;&nbsp; **Downstream party:** {names['B']}")
    st.markdown(f"**Current round:** {ex.current_round.number} &nbsp;&nbsp; **Phase:** {ex.current_round.phase.value}")
    st.markdown(f"**Status:** {ex.status.value}")
    st.markdown("#### The shared model")
    st.write(
        "Both parties see the same physical simulation of the river basin. "
        "Nothing about either party's private values is visible until settlement -- "
        "use the Simulate tab to jointly explore what a candidate package actually does "
        "to the river before anyone argues about whether it's a good deal."
    )
    st.markdown("#### Negotiated issues")
    for issue in ex.issues:
        st.markdown(f"- **{issue.name}**: {issue.description or ', '.join(issue.level_labels())}")

    if ex.config.factsheet.strip():
        st.markdown("#### Basin factsheet")
        st.caption("Real-world background, not modeled numerically in this exercise.")
        st.markdown(ex.config.factsheet)


def render_simulate(ex: Exercise, names: dict[str, str]) -> None:
    st.subheader("Joint fact-finding: simulate a candidate package")
    package = _package_editor(ex.issues, key_prefix="sim")

    st.markdown("#### Climate stress")
    mean_flow_mult = st.slider("Mean annual flow multiplier", 0.60, 1.10, 1.00, 0.01, key="sim_climate_mean")
    timing_shift = st.slider(
        "Seasonal timing shift, months (negative = earlier snowmelt)", -2.0, 2.0, 0.0, 0.1, key="sim_climate_shift"
    )
    trend = ClimateTrend(name="custom", mean_flow_multiplier=mean_flow_mult, timing_shift_months=timing_shift)

    st.markdown("#### Domestic cropping pattern *(each party's own choice -- not negotiated)*")
    ag_cfg = ex.config.basin.agriculture
    crops = {name: _crop_profile(cfg) for name, cfg in ag_cfg.crops.items()}
    a_areas = _crop_area_editor(ex, "A", names["A"], key_prefix="sim")
    b_areas = _crop_area_editor(ex, "B", names["B"], key_prefix="sim")

    monthly = generate_climatology(ex.config.hydrology.seasonal_shape, ex.config.hydrology.mean_annual_flow_mcm)
    trended = apply_climate_trend(monthly, trend)
    n_years = 3
    trace = pd.Series(list(trended.values) * n_years, index=pd.period_range("2000-01", periods=12 * n_years, freq="M"))

    outcomes = run_basin(ex.basin_config, package, trace)

    a_ag = agricultural_outcomes(float(outcomes.a_consumptive_diversion.sum() / n_years), a_areas, crops)
    b_ag = agricultural_outcomes(float(outcomes.delivered_to_b.sum() / n_years), b_areas, crops)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    outcomes.upstream_release.plot(ax=ax, label=f"{names['A']} release (gross, at the dam)")
    outcomes.delivered_to_b.plot(ax=ax, label=f"delivered to {names['B']}")
    outcomes.a_consumptive_diversion.plot(ax=ax, label=f"{names['A']}'s own consumptive draw", linestyle="--")
    ax.set_ylabel("MCM/month")
    ax.legend(fontsize=8)
    st.pyplot(fig)

    st.markdown("#### Outcome indicators")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{names['B']} irrigation reliability", f"{outcomes.irrigation_reliability_fraction:.0%}")
    c2.metric(f"{names['B']} shortfall", f"{outcomes.shortfall_fraction:.0%}")
    c3.metric(f"{names['A']} firm hydropower", f"{_fmt_number(outcomes.firm_hydropower_gwh_per_year)} GWh/yr")
    c4.metric(f"{names['A']} storage utilization", f"{outcomes.storage_utilization_fraction:.0%}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric(f"{names['A']} agricultural income", f"{_fmt_musd(a_ag.income_musd)}/yr")
    c6.metric(f"{names['A']} agricultural labor", f"{_fmt_number(a_ag.labor_persondays)} person-days/yr")
    c7.metric(f"{names['B']} agricultural income", f"{_fmt_musd(b_ag.income_musd)}/yr")
    c8.metric(f"{names['B']} agricultural labor", f"{_fmt_number(b_ag.labor_persondays)} person-days/yr")


def render_negotiate(ex: Exercise, names: dict[str, str]) -> None:
    st.subheader("Negotiate")

    if ex.status == ExerciseStatus.IMPASSE:
        reason = ex.current_round.facilitator_notes
        st.error(f"This exercise ended in impasse. Reason: {reason or '(none given)'}")
        st.write("See the Debrief tab for what was on the table when talks broke down.")
        return
    if ex.status == ExerciseStatus.SETTLED:
        st.success("This exercise has settled. See the Debrief tab.")
        return

    st.markdown(f"**Round {ex.current_round.number}** &nbsp;&nbsp; phase: `{ex.current_round.phase.value}`")
    batnas = ex.batnas()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"##### {names['A']}'s offer")
        package_a = _package_editor(ex.issues, key_prefix="offer_a")
        if st.button(f"Submit {names['A']}'s offer", key="submit_a"):
            scores = ex.submit_offer("A", package_a)
            st.session_state["last_scores_a"] = scores
    with col2:
        st.markdown(f"##### {names['B']}'s offer")
        package_b = _package_editor(ex.issues, key_prefix="offer_b")
        if st.button(f"Submit {names['B']}'s offer", key="submit_b"):
            scores = ex.submit_offer("B", package_b)
            st.session_state["last_scores_b"] = scores

    if ex.current_round.offers:
        st.markdown("##### This round's offers")
        for party, pkg in ex.current_round.offers.items():
            st.write(f"**{names[party]}**: {pkg.selections}")

    history = ex.offer_history()
    if not history.empty:
        st.markdown("##### Are the offers converging?")
        st.caption(
            "Round-over-round scores of every offer made so far. A cluster of points in the top-right "
            f"(both scores above BATNA: {names['A']} > {batnas['A']:.0f}, {names['B']} > {batnas['B']:.0f}) "
            "means the parties are inside the ZOPA; points spreading apart mean they're drifting toward impasse."
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        for party, color in (("A", "tab:blue"), ("B", "tab:orange")):
            sub = history[history["party"] == party]
            if not sub.empty:
                ax.scatter(sub["score_A"], sub["score_B"], color=color, label=f"{names[party]}'s offers")
                for _, row in sub.iterrows():
                    ax.annotate(str(int(row["round_number"])), (row["score_A"], row["score_B"]), fontsize=7)
        ax.axvline(batnas["A"], color="gray", linestyle="--", linewidth=1)
        ax.axhline(batnas["B"], color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(f"{names['A']} score")
        ax.set_ylabel(f"{names['B']} score")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        st.dataframe(history)

    st.divider()
    st.markdown("##### Single negotiating text")
    if ex.single_negotiating_text:
        if ex.current_text is not None:
            st.write(ex.current_text.selections)
        else:
            st.write("_No text drafted yet._")

        criticism_col1, criticism_col2 = st.columns(2)
        crit_a = criticism_col1.text_area(f"{names['A']}'s criticism of the current text", key="crit_a")
        crit_b = criticism_col2.text_area(f"{names['B']}'s criticism of the current text", key="crit_b")
        if st.button("Record criticisms", key="record_criticisms"):
            if ex.current_round.phase != Phase.SINGLE_TEXT_REVISION:
                ex.new_round(Phase.SINGLE_TEXT_REVISION)
            if crit_a:
                ex.submit_criticism("A", crit_a)
            if crit_b:
                ex.submit_criticism("B", crit_b)
            st.rerun()

        st.markdown("###### Facilitator revision")
        revised = _package_editor(
            ex.issues, key_prefix="revise", defaults=ex.current_text.selections if ex.current_text else None
        )
        notes = st.text_input("Facilitator notes on this revision", key="revise_notes")
        if st.button("Commit revised text", key="commit_revision"):
            ex.revise_text(revised, facilitator_notes=notes)
            st.rerun()
    else:
        st.write("Single negotiating text is disabled for this exercise (counter-offer mode).")

    st.divider()
    st.markdown("##### Settle")
    settle_default = ex.current_text.selections if ex.current_text else None
    settle_package = _package_editor(ex.issues, key_prefix="settle", defaults=settle_default)
    if st.button("Settle on this package", type="primary", key="settle_button"):
        monthly = generate_climatology(ex.config.hydrology.seasonal_shape, ex.config.hydrology.mean_annual_flow_mcm)
        ensemble = generate_climate_ensemble(monthly, BASELINE, n_years=25, drought_prob=0.2, wet_prob=0.2, seed=7)
        stronger = Package(selections={**settle_package.selections, "allocation_mechanism": "fixed_high"})
        contingent_rule = ContingentRule(
            name="contingent",
            packages_by_state={"dry": stronger, "normal": settle_package, "wet": settle_package},
            basin_config=ex.basin_config,
            value_functions=ex.value_functions,
            batnas=batnas,
        )
        ex.settle(settle_package, ensemble=ensemble, contingent_rule=contingent_rule)
        st.rerun()

    st.divider()
    st.markdown("##### Declare impasse")
    st.caption("Ends the exercise without agreement. What was on the table stays visible in the Debrief tab.")
    impasse_reason = st.text_input("Reason talks broke down", key="impasse_reason")
    if st.button("Declare impasse", key="declare_impasse_button"):
        ex.declare_impasse(impasse_reason or "No reason given.")
        st.rerun()


def render_debrief(ex: Exercise, names: dict[str, str]) -> None:
    if ex.status == ExerciseStatus.IN_PROGRESS:
        st.info("Debrief is locked until the parties settle or reach impasse (see the Negotiate tab).")
        return

    st.markdown("#### Both parties' value functions, revealed")
    for party, vf in ex.value_functions.items():
        with st.expander(f"{names[party]}'s weights"):
            st.write(vf.weights)

    if ex.status == ExerciseStatus.IMPASSE:
        reason = ex.current_round.facilitator_notes
        st.subheader("Debrief: impasse")
        st.error(f"Talks ended without agreement. Reason: {reason or '(none given)'}")
        history = ex.offer_history()
        if history.empty:
            st.write("No scored offers were made before the impasse.")
        else:
            st.markdown("#### What was on the table")
            st.caption(
                "Every scored offer either party made, and whether it cleared both BATNAs. "
                "This is the value that was available even though the parties walked away."
            )
            st.dataframe(history)
            in_zopa_offers = history[history["in_zopa"] == True]  # noqa: E712
            if not in_zopa_offers.empty:
                st.warning(
                    f"{len(in_zopa_offers)} offer(s) during the negotiation already cleared both BATNAs -- "
                    "a deal was available and not taken."
                )
        st.markdown("#### Event log")
        st.dataframe(ex.log.to_dataframe())
        return

    settlement = ex.settlement
    st.subheader("Debrief: settlement")
    st.markdown(f"**Agreed package:** {settlement.package.selections}")
    st.markdown(
        f"**Scores:** {names['A']} = {settlement.scores['A']:.1f}, "
        f"{names['B']} = {settlement.scores['B']:.1f} (0-100 scale)"
    )

    st.markdown("#### Efficiency loss")
    st.write(
        f"Joint-score shortfall vs. best joint-score frontier point: "
        f"**{settlement.efficiency_loss.joint_score_shortfall:.1f}**"
    )
    st.write(f"Foregone gain if moving to the nearest frontier point: {settlement.efficiency_loss.foregone_gain}")

    fig, ax = plt.subplots(figsize=(6, 5))
    all_scored = ex.all_scored_packages()
    ax.scatter([sp.scores["A"] for sp in all_scored], [sp.scores["B"] for sp in all_scored], s=4, alpha=0.1, color="gray")
    fa = [sp.scores["A"] for sp in settlement.frontier]
    fb = [sp.scores["B"] for sp in settlement.frontier]
    ax.plot(fa, fb, "o-", color="crimson", markersize=3, label="Pareto frontier")
    ax.scatter([settlement.scores["A"]], [settlement.scores["B"]], marker="X", s=150, color="black", label="Agreement", zorder=5)
    ax.scatter([settlement.nash_point.scores["A"]], [settlement.nash_point.scores["B"]], marker="*", s=200, color="gold", edgecolor="k", label="Nash", zorder=4)
    ax.scatter([settlement.kalai_smorodinsky_point.scores["A"]], [settlement.kalai_smorodinsky_point.scores["B"]], marker="D", s=80, color="green", edgecolor="k", label="Kalai-Smorodinsky", zorder=4)
    ax.set_xlabel(f"{names['A']} score")
    ax.set_ylabel(f"{names['B']} score")
    ax.legend(fontsize=8)
    st.pyplot(fig)

    st.markdown("#### Post-settlement settlement: top 5 packages both parties missed")
    if not settlement.pss:
        st.write("None -- the agreement was already on the Pareto frontier.")
    else:
        rows = [
            {"A": sp.scores["A"], "B": sp.scores["B"], **sp.package.selections}
            for sp in settlement.pss[:5]
        ]
        st.dataframe(pd.DataFrame(rows))

    if settlement.contingent_comparison is not None:
        st.markdown("#### Fixed vs. contingent: dry-year stress test")
        fixed_eval = settlement.contingent_comparison["fixed"]
        contingent_eval = settlement.contingent_comparison["contingent"]
        for party in ex.value_functions:
            st.write(
                f"**{names[party]}** -- fraction of ensemble years below BATNA: "
                f"fixed = {fixed_eval.frac_below_batna[party]:.0%}, "
                f"contingent (allocation_mechanism switches to fixed_high in dry years) = "
                f"{contingent_eval.frac_below_batna[party]:.0%}"
            )

    st.markdown("#### Event log")
    st.dataframe(ex.log.to_dataframe())


def render_methodology() -> None:
    st.subheader("Methodology")
    st.write(
        "The full negotiation-analytic and physical-model methodology, with citations -- "
        "readable here directly, or as `docs/METHODOLOGY.md` in the repository."
    )
    methodology_path = REPO_ROOT / "docs" / "METHODOLOGY.md"
    if methodology_path.exists():
        st.markdown(methodology_path.read_text())
    else:
        st.warning(f"{methodology_path} not found.")

    limitations_path = REPO_ROOT / "LIMITATIONS.md"
    if limitations_path.exists():
        with st.expander("Limitations (what's simplified, and why)"):
            st.markdown(limitations_path.read_text())


def main() -> None:
    st.set_page_config(page_title="riparia", layout="wide")
    ex = _init_state()
    names = _party_names(ex)
    st.title("riparia -- transboundary water negotiation exercise")
    tab_brief, tab_simulate, tab_negotiate, tab_debrief, tab_methodology = st.tabs(
        ["Brief", "Simulate", "Negotiate", "Debrief", "Methodology"]
    )
    with tab_brief:
        render_brief(ex, names)
    with tab_simulate:
        render_simulate(ex, names)
    with tab_negotiate:
        render_negotiate(ex, names)
    with tab_debrief:
        render_debrief(ex, names)
    with tab_methodology:
        render_methodology()


if __name__ == "__main__":
    main()
