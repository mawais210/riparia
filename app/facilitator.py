"""Streamlit facilitator application: one shared screen with four tabs
(Brief, Simulate, Negotiate, Debrief). Run with:

    streamlit run app/facilitator.py

Simulate is the joint fact-finding surface -- both parties see the same
physical simulation, with continuous sliders (climate stress, each party's
own crop-mix) alongside the discrete negotiated issues, En-ROADS style: move
a slider, the hydrograph and outcome indicators update immediately. Party
value functions stay hidden until Debrief unlocks at settlement.
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

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "src" / "riparia" / "config" / "indus_style_v1.yaml")

# Custom Functions


def _crop_profile(cfg) -> CropProfile:
    return CropProfile(
        label=cfg.label,
        water_use_mcm_per_1000ha=cfg.water_use_mcm_per_1000ha,
        income_musd_per_1000ha=cfg.income_musd_per_1000ha,
        labor_persondays_per_1000ha=cfg.labor_persondays_per_1000ha,
    )


@st.cache_resource
def _load_exercise() -> Exercise:
    cfg = load_config(CONFIG_PATH)
    return Exercise.start(cfg, n_years=3)


def _init_state() -> Exercise:
    if "exercise" not in st.session_state:
        st.session_state.exercise = _load_exercise()
    return st.session_state.exercise


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


def render_brief(ex: Exercise) -> None:
    st.header(ex.config.name)
    st.write(ex.config.description)
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


def render_simulate(ex: Exercise) -> None:
    st.subheader("Joint fact-finding: simulate a candidate package")
    package = _package_editor(ex.issues, key_prefix="sim")

    st.markdown("#### Climate stress")
    mean_flow_mult = st.slider("Mean annual flow multiplier", 0.60, 1.10, 1.00, 0.01, key="sim_climate_mean")
    timing_shift = st.slider(
        "Seasonal timing shift, months (negative = earlier snowmelt)", -2.0, 2.0, 0.0, 0.1, key="sim_climate_shift"
    )
    trend = ClimateTrend(name="custom", mean_flow_multiplier=mean_flow_mult, timing_shift_months=timing_shift)

    st.markdown("#### Domestic crop-mix response *(each party's own choice -- not negotiated)*")
    ag_cfg = ex.config.basin.agriculture
    col_a, col_b = st.columns(2)
    a_crop_mix = col_a.slider(
        "Country A: share of land in water-intensive crop",
        0.0, 1.0, ag_cfg.parties["A"].default_crop_mix_fraction, 0.05, key="sim_a_crop_mix",
    )
    b_crop_mix = col_b.slider(
        "Country B: share of land in water-intensive crop",
        0.0, 1.0, ag_cfg.parties["B"].default_crop_mix_fraction, 0.05, key="sim_b_crop_mix",
    )

    monthly = generate_climatology(ex.config.hydrology.seasonal_shape, ex.config.hydrology.mean_annual_flow_mcm)
    trended = apply_climate_trend(monthly, trend)
    n_years = 3
    trace = pd.Series(list(trended.values) * n_years, index=pd.period_range("2000-01", periods=12 * n_years, freq="M"))

    outcomes = run_basin(ex.basin_config, package, trace)

    high_crop, low_crop = _crop_profile(ag_cfg.high_water_crop), _crop_profile(ag_cfg.low_water_crop)
    a_ag = agricultural_outcomes(
        float(outcomes.a_consumptive_diversion.sum() / n_years), a_crop_mix,
        ag_cfg.parties["A"].command_area_1000ha, high_crop, low_crop,
    )
    b_ag = agricultural_outcomes(
        float(outcomes.delivered_to_b.sum() / n_years), b_crop_mix,
        ag_cfg.parties["B"].command_area_1000ha, high_crop, low_crop,
    )

    fig, ax = plt.subplots(figsize=(9, 3.5))
    outcomes.upstream_release.plot(ax=ax, label="A release (gross, at the dam)")
    outcomes.delivered_to_b.plot(ax=ax, label="delivered to B")
    outcomes.a_consumptive_diversion.plot(ax=ax, label="A's own consumptive draw", linestyle="--")
    ax.set_ylabel("MCM/month")
    ax.legend(fontsize=8)
    st.pyplot(fig)

    st.markdown("#### Outcome indicators")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("B irrigation reliability", f"{outcomes.irrigation_reliability_fraction:.0%}")
    c2.metric("B shortfall", f"{outcomes.shortfall_fraction:.0%}")
    c3.metric("A firm hydropower", f"{outcomes.firm_hydropower_gwh_per_year:.0f} GWh/yr")
    c4.metric("A storage utilization", f"{outcomes.storage_utilization_fraction:.0%}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("A agricultural income", f"${a_ag.income_musd:.0f}M/yr")
    c6.metric("A agricultural labor", f"{a_ag.labor_persondays / 1000:.0f}k person-days/yr")
    c7.metric("B agricultural income", f"${b_ag.income_musd:.0f}M/yr")
    c8.metric("B agricultural labor", f"{b_ag.labor_persondays / 1000:.0f}k person-days/yr")


def render_negotiate(ex: Exercise) -> None:
    st.subheader("Negotiate")
    st.markdown(f"**Round {ex.current_round.number}** &nbsp;&nbsp; phase: `{ex.current_round.phase.value}`")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Country A's offer")
        package_a = _package_editor(ex.issues, key_prefix="offer_a")
        if st.button("Submit A's offer", key="submit_a"):
            scores = ex.submit_offer("A", package_a)
            st.session_state["last_scores_a"] = scores
    with col2:
        st.markdown("##### Country B's offer")
        package_b = _package_editor(ex.issues, key_prefix="offer_b")
        if st.button("Submit B's offer", key="submit_b"):
            scores = ex.submit_offer("B", package_b)
            st.session_state["last_scores_b"] = scores

    if ex.current_round.offers:
        st.markdown("##### This round's offers")
        for party, pkg in ex.current_round.offers.items():
            st.write(f"**{party}**: {pkg.selections}")

    st.divider()
    st.markdown("##### Single negotiating text")
    if ex.single_negotiating_text:
        if ex.current_text is not None:
            st.write(ex.current_text.selections)
        else:
            st.write("_No text drafted yet._")

        criticism_col1, criticism_col2 = st.columns(2)
        crit_a = criticism_col1.text_area("Country A's criticism of the current text", key="crit_a")
        crit_b = criticism_col2.text_area("Country B's criticism of the current text", key="crit_b")
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
        batnas = {p: batna(ex.value_functions, p) for p in ex.value_functions}
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


def render_debrief(ex: Exercise) -> None:
    if ex.status != ExerciseStatus.SETTLED:
        st.info("Debrief is locked until the parties settle (see the Negotiate tab).")
        return

    settlement = ex.settlement
    st.subheader("Debrief")
    st.markdown(f"**Agreed package:** {settlement.package.selections}")
    st.markdown(f"**Scores:** A = {settlement.scores['A']:.1f}, B = {settlement.scores['B']:.1f} (0-100 scale)")

    st.markdown("#### Both parties' value functions, revealed")
    for party, vf in ex.value_functions.items():
        with st.expander(f"Country {party}'s weights"):
            st.write(vf.weights)

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
    ax.set_xlabel("Country A score")
    ax.set_ylabel("Country B score")
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
                f"**{party}** -- fraction of ensemble years below BATNA: "
                f"fixed = {fixed_eval.frac_below_batna[party]:.0%}, "
                f"contingent (allocation_mechanism switches to fixed_high in dry years) = "
                f"{contingent_eval.frac_below_batna[party]:.0%}"
            )

    st.markdown("#### Event log")
    st.dataframe(ex.log.to_dataframe())


def main() -> None:
    st.set_page_config(page_title="riparia", layout="wide")
    ex = _init_state()
    st.title("riparia -- transboundary water negotiation exercise")
    tab_brief, tab_simulate, tab_negotiate, tab_debrief = st.tabs(["Brief", "Simulate", "Negotiate", "Debrief"])
    with tab_brief:
        render_brief(ex)
    with tab_simulate:
        render_simulate(ex)
    with tab_negotiate:
        render_negotiate(ex)
    with tab_debrief:
        render_debrief(ex)


if __name__ == "__main__":
    main()
