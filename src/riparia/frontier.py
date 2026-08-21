"""Pareto frontier, efficiency loss, post-settlement search, and reference
bargaining solutions (Nash, Kalai-Smorodinsky) over scored packages.
"""

# Import necessary libraries
from __future__ import annotations

from dataclasses import dataclass

from riparia.issues import Package
from riparia.payoffs import ScoredPackage

# Custom Functions

Parties = tuple[str, str]


def pareto_frontier(scored_packages: list[ScoredPackage], parties: Parties = ("A", "B")) -> list[ScoredPackage]:
    """Non-dominated set over the two-party score vector, sorted ascending
    by the first party's score."""
    p1, p2 = parties
    frontier = []
    for sp in scored_packages:
        dominated = False
        for other in scored_packages:
            if other is sp:
                continue
            weakly_better = other.scores[p1] >= sp.scores[p1] and other.scores[p2] >= sp.scores[p2]
            strictly_better = other.scores[p1] > sp.scores[p1] or other.scores[p2] > sp.scores[p2]
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(sp)
    frontier.sort(key=lambda sp: sp.scores[p1])
    return frontier


@dataclass
class EfficiencyLoss:
    joint_score_shortfall: float
    foregone_gain: dict[str, float]
    nearest_frontier_point: Package


def efficiency_loss(
    agreement: ScoredPackage, frontier: list[ScoredPackage], parties: Parties = ("A", "B")
) -> EfficiencyLoss:
    """How far the agreed point sits from the frontier: joint-score
    shortfall against the best joint-score frontier point, and each party's
    foregone gain against the Euclidean-nearest frontier point."""
    p1, p2 = parties
    agreement_joint = agreement.scores[p1] + agreement.scores[p2]
    best_joint = max(sp.scores[p1] + sp.scores[p2] for sp in frontier)

    def dist(sp: ScoredPackage) -> float:
        return ((sp.scores[p1] - agreement.scores[p1]) ** 2 + (sp.scores[p2] - agreement.scores[p2]) ** 2) ** 0.5

    nearest = min(frontier, key=dist)
    foregone = {
        p1: max(0.0, nearest.scores[p1] - agreement.scores[p1]),
        p2: max(0.0, nearest.scores[p2] - agreement.scores[p2]),
    }
    return EfficiencyLoss(
        joint_score_shortfall=best_joint - agreement_joint,
        foregone_gain=foregone,
        nearest_frontier_point=nearest.package,
    )


def post_settlement_search(
    agreement: ScoredPackage, scored_packages: list[ScoredPackage], parties: Parties = ("A", "B")
) -> list[ScoredPackage]:
    """Every package that scores strictly higher for BOTH parties than the
    agreement, sorted by joint gain descending."""
    p1, p2 = parties
    better = [sp for sp in scored_packages if sp.scores[p1] > agreement.scores[p1] and sp.scores[p2] > agreement.scores[p2]]
    better.sort(
        key=lambda sp: (sp.scores[p1] - agreement.scores[p1]) + (sp.scores[p2] - agreement.scores[p2]),
        reverse=True,
    )
    return better


def nash_solution(
    frontier: list[ScoredPackage], batnas: dict[str, float], parties: Parties = ("A", "B")
) -> ScoredPackage:
    """Frontier point maximising the Nash product (s1-batna1)*(s2-batna2)."""
    p1, p2 = parties

    def nash_product(sp: ScoredPackage) -> float:
        d1, d2 = sp.scores[p1] - batnas[p1], sp.scores[p2] - batnas[p2]
        return d1 * d2 if d1 > 0 and d2 > 0 else -1.0

    return max(frontier, key=nash_product)


def kalai_smorodinsky(
    frontier: list[ScoredPackage], batnas: dict[str, float], parties: Parties = ("A", "B")
) -> ScoredPackage:
    """Frontier point closest to the Kalai-Smorodinsky line: equal fractional
    progress from the disagreement point (BATNAs) toward each party's ideal
    (max attainable score on the frontier)."""
    p1, p2 = parties
    ideal1 = max(sp.scores[p1] for sp in frontier)
    ideal2 = max(sp.scores[p2] for sp in frontier)
    candidates = [sp for sp in frontier if sp.scores[p1] >= batnas[p1] and sp.scores[p2] >= batnas[p2]] or frontier

    def ks_gap(sp: ScoredPackage) -> float:
        r1 = (sp.scores[p1] - batnas[p1]) / (ideal1 - batnas[p1]) if ideal1 > batnas[p1] else 0.0
        r2 = (sp.scores[p2] - batnas[p2]) / (ideal2 - batnas[p2]) if ideal2 > batnas[p2] else 0.0
        return abs(r1 - r2)

    return min(candidates, key=ks_gap)
