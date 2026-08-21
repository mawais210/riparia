"""Issue definitions, packages, and feasible-agreement-space enumeration."""

# Import necessary libraries
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from riparia.config_schema import IssuesConfig

# Custom Functions


@dataclass(frozen=True)
class IssueLevel:
    label: str
    params: dict[str, Any]


@dataclass(frozen=True)
class Issue:
    name: str
    levels: list[IssueLevel]
    description: str = ""

    def level_labels(self) -> list[str]:
        return [lv.label for lv in self.levels]

    def level(self, label: str) -> IssueLevel:
        for lv in self.levels:
            if lv.label == label:
                return lv
        raise KeyError(f"issue {self.name} has no level {label!r}")


@dataclass(frozen=True)
class Package:
    """A whole agreement package: one chosen level per issue."""

    selections: dict[str, str]

    def __getitem__(self, issue_name: str) -> str:
        return self.selections[issue_name]

    def level_params(self, issues_by_name: dict[str, Issue], issue_name: str) -> dict[str, Any]:
        return issues_by_name[issue_name].level(self.selections[issue_name]).params

    def as_tuple(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self.selections.items()))


def load_issues(issues_config: IssuesConfig) -> list[Issue]:
    """Build domain Issue objects from a validated IssuesConfig."""
    issues = []
    for defn in issues_config.definitions:
        levels = [IssueLevel(label=lv.label, params=dict(lv.params)) for lv in defn.levels]
        issues.append(Issue(name=defn.name, levels=levels, description=defn.description))
    return issues


def validate_package(package: Package, issues: list[Issue]) -> None:
    """Raise ValueError if `package` does not select a valid level for every issue."""
    issues_by_name = {issue.name: issue for issue in issues}
    if set(package.selections.keys()) != set(issues_by_name.keys()):
        raise ValueError(
            f"package covers issues {set(package.selections.keys())}, "
            f"expected exactly {set(issues_by_name.keys())} (whole packages only)"
        )
    for issue_name, label in package.selections.items():
        if label not in issues_by_name[issue_name].level_labels():
            raise ValueError(f"{label!r} is not a valid level of issue {issue_name!r}")


def is_feasible(selections: dict[str, str], forbidden_combinations: list[dict[str, str]]) -> bool:
    for forbidden in forbidden_combinations:
        if all(selections.get(k) == v for k, v in forbidden.items()):
            return False
    return True


def enumerate_packages(issues: list[Issue], forbidden_combinations: list[dict[str, str]]) -> list[Package]:
    """Full Cartesian product over issue levels, minus any package matching a
    forbidden combination in full."""
    names = [issue.name for issue in issues]
    level_label_lists = [issue.level_labels() for issue in issues]

    packages = []
    for combo in product(*level_label_lists):
        selections = dict(zip(names, combo))
        if is_feasible(selections, forbidden_combinations):
            packages.append(Package(selections=selections))
    return packages
