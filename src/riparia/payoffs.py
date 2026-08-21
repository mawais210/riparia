"""Party value functions (additive, PON-style, assigned from config), BATNA,
and ZOPA computation.

Scores are on a 0-100 scale per party. Each party's weights sum to 1 across
its issue-level components (scored directly from config's issue_scores
tables) and its outcome-derived components (scored from BasinOutputs via a
configured normalisation range, never from the package directly).
"""

# Import necessary libraries
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from riparia.config_schema import OutcomeTermConfig, PartyConfig, ScenarioConfig
from riparia.issues import Package
from riparia.system import BasinConfig, BasinOutputs, run_basin

# Custom Functions


@dataclass
class ValueFunction:
    party: str
    weights: dict[str, float]
    issue_scores: dict[str, dict[str, float]]
    outcome_terms: dict[str, OutcomeTermConfig]
    batna_value: float

    @classmethod
    def from_config(cls, party_key: str, party_cfg: PartyConfig) -> "ValueFunction":
        return cls(
            party=party_key,
            weights=dict(party_cfg.weights),
            issue_scores={k: dict(v) for k, v in party_cfg.issue_scores.items()},
            outcome_terms=dict(party_cfg.outcome_terms),
            batna_value=party_cfg.batna,
        )


def load_value_functions(cfg: ScenarioConfig) -> dict[str, ValueFunction]:
    return {party: ValueFunction.from_config(party, party_cfg) for party, party_cfg in cfg.parties.items()}


def batna(value_functions: dict[str, ValueFunction], party: str) -> float:
    return value_functions[party].batna_value


def score(vf: ValueFunction, package: Package, outcomes: BasinOutputs) -> float:
    """Score one package for one party, 0-100."""
    total = 0.0
    for issue_name, score_map in vf.issue_scores.items():
        level = package[issue_name]
        total += vf.weights[issue_name] * score_map[level]

    for term_name, term_cfg in vf.outcome_terms.items():
        raw = getattr(outcomes, term_cfg.source)
        span = term_cfg.norm_max - term_cfg.norm_min
        normalized = (raw - term_cfg.norm_min) / span
        normalized = min(max(normalized, 0.0), 1.0)
        if term_cfg.invert:
            normalized = 1.0 - normalized
        total += vf.weights[term_name] * normalized * 100.0

    return min(max(total, 0.0), 100.0)


@dataclass
class ScoredPackage:
    package: Package
    scores: dict[str, float]
    outcomes: BasinOutputs


def score_all_packages(
    packages: list[Package],
    basin_config: BasinConfig,
    flow_trace: pd.Series,
    value_functions: dict[str, ValueFunction],
) -> list[ScoredPackage]:
    """Run the basin model and score every party for every package, against
    a single representative flow trace (typically a normal/climatological year)."""
    results = []
    for pkg in packages:
        outcomes = run_basin(basin_config, pkg, flow_trace)
        scores = {party: score(vf, pkg, outcomes) for party, vf in value_functions.items()}
        results.append(ScoredPackage(package=pkg, scores=scores, outcomes=outcomes))
    return results


def zopa(scored_packages: list[ScoredPackage], batnas: dict[str, float]) -> list[ScoredPackage]:
    """Packages that clear every party's BATNA. Raises ValueError naming any
    party that never clears its BATNA across the enumerated space — an empty
    ZOPA is a scenario design bug, not a valid exercise outcome."""
    in_zopa = [sp for sp in scored_packages if all(sp.scores[p] >= batnas[p] for p in batnas)]
    if not in_zopa:
        blocked = [p for p in batnas if not any(sp.scores[p] >= batnas[p] for sp in scored_packages)]
        if blocked:
            raise ValueError(
                f"No ZOPA: {blocked} never reach their BATNA in any enumerated package "
                "-- rebalance issue_scores/weights or lower batna for that party"
            )
        raise ValueError(
            "No ZOPA: some packages clear each party's BATNA individually but none clears "
            "both simultaneously -- rebalance issue_scores/weights"
        )
    return in_zopa
