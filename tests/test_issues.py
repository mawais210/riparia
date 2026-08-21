import pytest

from riparia.config_schema import load_config
from riparia.issues import enumerate_packages, is_feasible, load_issues, validate_package

CONFIG_PATH = "src/riparia/config/indus_style_v1.yaml"


def _issues():
    cfg = load_config(CONFIG_PATH)
    return cfg, load_issues(cfg.issues)


def test_load_issues_matches_config():
    cfg, issues = _issues()
    assert len(issues) == len(cfg.issues.definitions)
    names = {i.name for i in issues}
    assert names == {d.name for d in cfg.issues.definitions}


def test_enumerate_packages_within_target_range():
    cfg, issues = _issues()
    packages = enumerate_packages(issues, cfg.issues.forbidden_combinations)
    assert 2000 <= len(packages) <= 20000, f"got {len(packages)} packages"


def test_enumerate_packages_excludes_forbidden_combinations():
    cfg, issues = _issues()
    packages = enumerate_packages(issues, cfg.issues.forbidden_combinations)
    for pkg in packages:
        assert is_feasible(pkg.selections, cfg.issues.forbidden_combinations)
    total_unconstrained = 1
    for issue in issues:
        total_unconstrained *= len(issue.level_labels())
    assert len(packages) < total_unconstrained


def test_validate_package_rejects_partial_package():
    cfg, issues = _issues()
    packages = enumerate_packages(issues, cfg.issues.forbidden_combinations)
    full = packages[0]
    partial_selections = dict(full.selections)
    del partial_selections[next(iter(partial_selections))]
    from riparia.issues import Package

    with pytest.raises(ValueError):
        validate_package(Package(selections=partial_selections), issues)


def test_validate_package_accepts_full_package():
    cfg, issues = _issues()
    packages = enumerate_packages(issues, cfg.issues.forbidden_combinations)
    validate_package(packages[0], issues)
