"""Turn-based round state machine for a two-party negotiation exercise:
round/phase state, offers, the single negotiating text, and settlement,
all mediated through an append-only EventLog (logging_.py).
"""

# Import necessary libraries
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from riparia.config_schema import ScenarioConfig
from riparia.contingent import ContingentEvaluation, ContingentRule, compare_fixed_vs_contingent
from riparia.frontier import EfficiencyLoss, efficiency_loss, kalai_smorodinsky, nash_solution, pareto_frontier, post_settlement_search
from riparia.hydrology import generate_climatology
from riparia.issues import Issue, Package, enumerate_packages, load_issues, validate_package
from riparia.logging_ import EventLog, classify_move
from riparia.payoffs import ScoredPackage, ValueFunction, batna, load_value_functions, score, score_all_packages
from riparia.system import BasinConfig, run_basin

# Custom Functions


class Phase(str, Enum):
    JOINT_FACT_FINDING = "joint_fact_finding"
    PACKAGE_BIDDING = "package_bidding"
    SINGLE_TEXT_REVISION = "single_text_revision"
    SETTLEMENT = "settlement"


class ExerciseStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    SETTLED = "settled"
    IMPASSE = "impasse"


@dataclass
class Round:
    number: int
    phase: Phase
    offers: dict[str, Package] = field(default_factory=dict)
    criticisms: dict[str, str] = field(default_factory=dict)
    facilitator_notes: str = ""


@dataclass
class SettlementRecord:
    package: Package
    scores: dict[str, float]
    frontier: list[ScoredPackage]
    efficiency_loss: EfficiencyLoss
    pss: list[ScoredPackage]
    nash_point: ScoredPackage
    kalai_smorodinsky_point: ScoredPackage
    contingent_comparison: dict[str, ContingentEvaluation] | None = None


def default_flow_trace(cfg: ScenarioConfig, n_years: int = 3) -> pd.Series:
    """A representative multi-year trace built by repeating the scenario's
    climatology -- the standard basis for package scoring/ZOPA/frontier
    throughout this codebase (dry/wet stress testing uses the ensemble
    machinery in contingent.py/climate.py instead)."""
    monthly = generate_climatology(cfg.hydrology.seasonal_shape, cfg.hydrology.mean_annual_flow_mcm)
    index = pd.period_range("2000-01", periods=12 * n_years, freq="M")
    return pd.Series(list(monthly.values) * n_years, index=index)


@dataclass
class Exercise:
    """Holds config, round history, the current single negotiating text
    (if `single_negotiating_text` is enabled), and the settlement record
    once settled. `require_whole_packages` and `single_negotiating_text`
    are experimental treatments: toggle them to compare exercise designs."""

    config: ScenarioConfig
    issues: list[Issue]
    basin_config: BasinConfig
    value_functions: dict[str, ValueFunction]
    flow_trace: pd.Series
    require_whole_packages: bool = True
    single_negotiating_text: bool = True
    rounds: list[Round] = field(default_factory=list)
    current_text: Package | None = None
    status: ExerciseStatus = ExerciseStatus.IN_PROGRESS
    settlement: SettlementRecord | None = None
    log: EventLog = field(default_factory=EventLog)
    _all_scored_cache: list[ScoredPackage] | None = field(default=None, repr=False)
    _last_offer_scores: dict[str, float] | None = field(default=None, repr=False)

    @classmethod
    def start(
        cls,
        cfg: ScenarioConfig,
        n_years: int = 3,
        require_whole_packages: bool = True,
        single_negotiating_text: bool = True,
    ) -> "Exercise":
        issues = load_issues(cfg.issues)
        basin_config = BasinConfig.from_scenario(cfg, issues)
        value_functions = load_value_functions(cfg)
        flow_trace = default_flow_trace(cfg, n_years)

        exercise = cls(
            config=cfg,
            issues=issues,
            basin_config=basin_config,
            value_functions=value_functions,
            flow_trace=flow_trace,
            require_whole_packages=require_whole_packages,
            single_negotiating_text=single_negotiating_text,
        )
        exercise.new_round(Phase.JOINT_FACT_FINDING)
        exercise.log.append(1, Phase.JOINT_FACT_FINDING.value, "exercise_started", payload={"scenario": cfg.name})
        return exercise

    @property
    def current_round(self) -> Round:
        return self.rounds[-1]

    def all_scored_packages(self) -> list[ScoredPackage]:
        """Every feasible package, scored for both parties against
        `flow_trace`. Computed once and cached -- this is the same
        underlying enumeration frontier.py/payoffs.py operate on."""
        if self._all_scored_cache is None:
            packages = enumerate_packages(self.issues, self.config.issues.forbidden_combinations)
            self._all_scored_cache = score_all_packages(packages, self.basin_config, self.flow_trace, self.value_functions)
        return self._all_scored_cache

    def new_round(self, phase: Phase, facilitator_notes: str = "") -> Round:
        number = len(self.rounds) + 1
        round_ = Round(number=number, phase=phase, facilitator_notes=facilitator_notes)
        self.rounds.append(round_)
        self.log.append(number, phase.value, "round_started", payload={"facilitator_notes": facilitator_notes})
        return round_

    def _score_package(self, package: Package) -> dict[str, float]:
        outcomes = run_basin(self.basin_config, package, self.flow_trace)
        return {party: score(vf, package, outcomes) for party, vf in self.value_functions.items()}

    def _check_active(self) -> None:
        if self.status != ExerciseStatus.IN_PROGRESS:
            raise ValueError(f"exercise already ended (status={self.status.value}); no further moves are recorded")

    def batnas(self) -> dict[str, float]:
        return {p: batna(self.value_functions, p) for p in self.value_functions}

    def submit_offer(self, party: str, package: Package) -> dict[str, float] | None:
        """Record an offer and, if it covers every issue, score and log its
        move type and whether it clears both parties' BATNAs (in_zopa). A
        partial offer (only possible when `require_whole_packages=False`)
        can't be run through the physical model -- it's recorded but not
        scored."""
        self._check_active()
        if self.require_whole_packages:
            validate_package(package, self.issues)

        is_complete = set(package.selections.keys()) == {issue.name for issue in self.issues}
        scores = self._score_package(package) if is_complete else None
        move_type = classify_move(self._last_offer_scores, scores) if scores is not None else None
        in_zopa = all(scores[p] >= b for p, b in self.batnas().items()) if scores is not None else None
        if scores is not None:
            self._last_offer_scores = scores

        self.current_round.offers[party] = package
        self.log.append(
            self.current_round.number,
            self.current_round.phase.value,
            "offer",
            party=party,
            payload={"selections": package.selections, "scores": scores, "in_zopa": in_zopa},
            move_type=move_type,
        )
        return scores

    def offer_history(self) -> pd.DataFrame:
        """Every scored offer logged so far, one row per offer: round
        number, party, move type, each party's score, and whether the offer
        cleared both BATNAs. Lets a facilitator see round-over-round
        whether the parties are converging toward the ZOPA or drifting
        apart, instead of only finding out at settlement or impasse."""
        rows = []
        for event in self.log.events:
            if event.event_type != "offer" or event.payload.get("scores") is None:
                continue
            scores = event.payload["scores"]
            rows.append(
                {
                    "round_number": event.round_number,
                    "party": event.party,
                    "move_type": event.move_type,
                    **{f"score_{p}": s for p, s in scores.items()},
                    "in_zopa": event.payload.get("in_zopa"),
                }
            )
        return pd.DataFrame(rows)

    def submit_criticism(self, party: str, text: str) -> None:
        self._check_active()
        if not self.single_negotiating_text:
            raise ValueError("submit_criticism requires single_negotiating_text=True")
        self.current_round.criticisms[party] = text
        self.log.append(
            self.current_round.number, self.current_round.phase.value, "criticism", party=party, payload={"text": text}
        )

    def revise_text(self, new_text: Package, facilitator_notes: str = "") -> None:
        self._check_active()
        if not self.single_negotiating_text:
            raise ValueError("revise_text requires single_negotiating_text=True")
        if self.require_whole_packages:
            validate_package(new_text, self.issues)
        self.current_text = new_text
        self.log.append(
            self.current_round.number,
            self.current_round.phase.value,
            "revision",
            payload={"selections": new_text.selections, "facilitator_notes": facilitator_notes},
        )

    def declare_impasse(self, reason: str) -> None:
        """End the exercise without agreement. Distinct from `settle()`:
        no package is frozen, but the event log (and `offer_history()`)
        still records everything that was on the table, so a debrief can
        show what could have worked even though the parties walked away."""
        self._check_active()
        self.status = ExerciseStatus.IMPASSE
        self.new_round(Phase.SETTLEMENT, facilitator_notes=reason)
        self.log.append(self.current_round.number, Phase.SETTLEMENT.value, "impasse", payload={"reason": reason})

    def settle(
        self,
        package: Package,
        ensemble: pd.DataFrame | None = None,
        contingent_rule: ContingentRule | None = None,
    ) -> SettlementRecord:
        """Freeze `package` as the agreement, then automatically run frontier
        analysis, PSS search, and (if `ensemble`/`contingent_rule` given) the
        fixed-vs-contingent stress comparison, storing all three."""
        self._check_active()
        if self.require_whole_packages:
            validate_package(package, self.issues)

        scores = self._score_package(package)
        all_scored = self.all_scored_packages()
        agreement = ScoredPackage(package=package, scores=scores, outcomes=run_basin(self.basin_config, package, self.flow_trace))

        frontier = pareto_frontier(all_scored)
        loss = efficiency_loss(agreement, frontier)
        pss = post_settlement_search(agreement, all_scored)
        batnas = self.batnas()
        nash_pt = nash_solution(frontier, batnas)
        ks_pt = kalai_smorodinsky(frontier, batnas)

        contingent_comparison = None
        if ensemble is not None and contingent_rule is not None:
            contingent_comparison = compare_fixed_vs_contingent(package, contingent_rule, ensemble)

        self.settlement = SettlementRecord(
            package=package,
            scores=scores,
            frontier=frontier,
            efficiency_loss=loss,
            pss=pss,
            nash_point=nash_pt,
            kalai_smorodinsky_point=ks_pt,
            contingent_comparison=contingent_comparison,
        )
        self.status = ExerciseStatus.SETTLED
        self.new_round(Phase.SETTLEMENT)
        self.log.append(
            self.current_round.number,
            Phase.SETTLEMENT.value,
            "settlement",
            payload={"selections": package.selections, "scores": scores, "joint_score_shortfall": loss.joint_score_shortfall},
        )
        return self.settlement
