"""State-contingent allocation rules: agreements whose terms switch by
hydrological state (dry/normal/wet), evaluated across a stochastic ensemble
and compared against a fixed-package agreement under the same ensemble.
"""

# Import necessary libraries
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from riparia.issues import Package
from riparia.payoffs import ValueFunction, score
from riparia.system import BasinConfig, run_basin

# Custom Functions

_PERCENTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


@dataclass
class ContingentRule:
    """A state-contingent agreement plus everything needed to evaluate it."""

    name: str
    packages_by_state: dict[str, Package]  # "dry" / "normal" / "wet" -> Package
    basin_config: BasinConfig
    value_functions: dict[str, ValueFunction]
    batnas: dict[str, float]


@dataclass
class ContingentEvaluation:
    raw: pd.DataFrame  # columns: year, tag, <party scores...>
    percentiles: dict[str, dict[float, float]]
    frac_below_batna: dict[str, float]


def _year_trace(ensemble: pd.DataFrame, year: int) -> tuple[pd.Series, str]:
    group = ensemble.xs(year, level="year").sort_index()
    tag = group["tag"].iloc[0]
    index = pd.period_range("2000-01", periods=12, freq="M")
    trace = pd.Series(group["flow_mcm"].values, index=index)
    return trace, tag


def evaluate_contingent(rule: ContingentRule, ensemble: pd.DataFrame) -> ContingentEvaluation:
    """Run `rule` across every year of `ensemble`, selecting the package for
    each year's hydrological tag, and return the resulting payoff
    distribution per party: raw scores, percentiles, and the frequency each
    party falls below its BATNA."""
    years = sorted(ensemble.index.get_level_values("year").unique())
    rows = []
    for year in years:
        trace, tag = _year_trace(ensemble, year)
        package = rule.packages_by_state[tag]
        outcomes = run_basin(rule.basin_config, package, trace)
        row = {"year": year, "tag": tag}
        for party, vf in rule.value_functions.items():
            row[party] = score(vf, package, outcomes)
        rows.append(row)
    raw = pd.DataFrame(rows)

    percentiles: dict[str, dict[float, float]] = {}
    frac_below_batna: dict[str, float] = {}
    for party in rule.value_functions:
        vals = raw[party]
        percentiles[party] = {q: float(vals.quantile(q)) for q in _PERCENTILES}
        frac_below_batna[party] = float((vals < rule.batnas[party]).mean())

    return ContingentEvaluation(raw=raw, percentiles=percentiles, frac_below_batna=frac_below_batna)


def compare_fixed_vs_contingent(
    fixed_package: Package, contingent_rule: ContingentRule, ensemble: pd.DataFrame
) -> dict[str, ContingentEvaluation]:
    """Headline stress-test comparison: the same fixed package applied every
    year vs. the state-contingent rule, both run over the same ensemble."""
    fixed_rule = ContingentRule(
        name="fixed",
        packages_by_state={"dry": fixed_package, "normal": fixed_package, "wet": fixed_package},
        basin_config=contingent_rule.basin_config,
        value_functions=contingent_rule.value_functions,
        batnas=contingent_rule.batnas,
    )
    return {
        "fixed": evaluate_contingent(fixed_rule, ensemble),
        "contingent": evaluate_contingent(contingent_rule, ensemble),
    }
