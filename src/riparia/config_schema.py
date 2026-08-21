"""Pydantic schema for riparia scenario configuration files.

All volumes are in million cubic metres (MCM). All flow rates expressed as
monthly totals are in MCM/month unless a field name says otherwise. Monetary
values are in millions of US dollars (MUSD) per year. Months are 1-12
(calendar month numbers), independent of `water_year_start`.
"""

# Import necessary libraries
from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

# Custom Functions


class TimeConfig(BaseModel):
    water_year_start: int = Field(..., ge=1, le=12)
    seasons: dict[str, list[int]]


class DailyDisaggConfig(BaseModel):
    cv_low: float = Field(..., gt=0)
    cv_high: float = Field(..., gt=0)
    phi: float = Field(..., ge=0, lt=1)


class EnsembleConfig(BaseModel):
    n_years: int = Field(..., gt=0)
    drought_prob: float = Field(..., ge=0, le=1)
    wet_prob: float = Field(..., ge=0, le=1)
    seed: int
    dry_scale: float = Field(..., gt=0, lt=1)
    wet_scale: float = Field(..., gt=1)

    @model_validator(mode="after")
    def _probs_sum_to_at_most_one(self) -> "EnsembleConfig":
        if self.drought_prob + self.wet_prob > 1.0 + 1e-9:
            raise ValueError("drought_prob + wet_prob must not exceed 1")
        return self


class HydrologyConfig(BaseModel):
    mean_annual_flow_mcm: float = Field(..., gt=0)
    seasonal_shape: dict[int, float]
    tributary_inflow_fraction: float = Field(..., ge=0)
    travel_time_days: float = Field(..., ge=0)
    muskingum_x: float = Field(..., ge=0, le=0.5)
    daily_disagg: DailyDisaggConfig
    ensemble: EnsembleConfig

    @model_validator(mode="after")
    def _shape_has_twelve_months(self) -> "HydrologyConfig":
        if set(self.seasonal_shape.keys()) != set(range(1, 13)):
            raise ValueError("seasonal_shape must have exactly months 1..12")
        return self


class ReservoirConfig(BaseModel):
    name: str
    fill_months: list[int]
    release_months: list[int]
    fill_fraction: float = Field(..., gt=0, le=1)
    min_release_mcm_per_month: float = Field(..., ge=0)
    spin_up_years: int = Field(..., ge=0)
    initial_storage_fraction: float = Field(default=0.5, ge=0, le=1)
    live_storage_mcm: float | None = Field(
        default=None,
        description="Fixed live storage; None if set dynamically from an issue level (upstream reservoir).",
    )
    enabled: bool = True


class ReservoirsConfig(BaseModel):
    upstream: ReservoirConfig
    downstream: ReservoirConfig


class IssueLevelConfig(BaseModel):
    label: str
    params: dict[str, float | int | list[int] | str] = Field(default_factory=dict)


class IssueConfig(BaseModel):
    name: str
    description: str = ""
    levels: list[IssueLevelConfig]

    @model_validator(mode="after")
    def _levels_nonempty(self) -> "IssueConfig":
        if not self.levels:
            raise ValueError(f"issue {self.name} has no levels")
        labels = [lv.label for lv in self.levels]
        if len(labels) != len(set(labels)):
            raise ValueError(f"issue {self.name} has duplicate level labels")
        return self


class IssuesConfig(BaseModel):
    definitions: list[IssueConfig]
    forbidden_combinations: list[dict[str, str]] = Field(default_factory=list)


class OutcomeTermConfig(BaseModel):
    source: str
    norm_min: float
    norm_max: float
    invert: bool = False

    @model_validator(mode="after")
    def _range_valid(self) -> "OutcomeTermConfig":
        if self.norm_max <= self.norm_min:
            raise ValueError("norm_max must exceed norm_min")
        return self


class PartyConfig(BaseModel):
    name: str
    batna: float = Field(..., ge=0, le=100)
    weights: dict[str, float]
    issue_scores: dict[str, dict[str, float]]
    outcome_terms: dict[str, OutcomeTermConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "PartyConfig":
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"party {self.name} weights sum to {total}, must sum to 1.0")
        expected_keys = set(self.issue_scores.keys()) | set(self.outcome_terms.keys())
        if set(self.weights.keys()) != expected_keys:
            raise ValueError(
                f"party {self.name} weight keys {set(self.weights.keys())} do not match "
                f"issue_scores/outcome_terms keys {expected_keys}"
            )
        return self


class BasinConfig(BaseModel):
    reservoirs: ReservoirsConfig


class ScenarioConfig(BaseModel):
    name: str
    description: str = ""
    time: TimeConfig
    hydrology: HydrologyConfig
    basin: BasinConfig
    issues: IssuesConfig
    parties: dict[Literal["A", "B"], PartyConfig]


def load_config(path: str) -> ScenarioConfig:
    """Load and validate a scenario YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return ScenarioConfig.model_validate(raw)
