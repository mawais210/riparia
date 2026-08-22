from riparia.config_schema import load_config
from riparia.issues import Package, load_issues, validate_package

CONFIG_PATH = "src/riparia/config/indus_basin_v1.yaml"


def test_indus_basin_has_case_studies():
    cfg = load_config(CONFIG_PATH)
    assert len(cfg.case_studies) >= 2
    names = {cs.name for cs in cfg.case_studies}
    assert any("Baglihar" in n for n in names)
    assert any("Kishenganga" in n for n in names)


def test_case_study_openings_are_valid_whole_packages():
    cfg = load_config(CONFIG_PATH)
    issues = load_issues(cfg.issues)
    for cs in cfg.case_studies:
        validate_package(Package(selections=cs.party_a_opening), issues)
        validate_package(Package(selections=cs.party_b_opening), issues)


def test_case_studies_have_cited_summary_and_outcome():
    cfg = load_config(CONFIG_PATH)
    for cs in cfg.case_studies:
        assert len(cs.summary.strip()) > 50
        assert len(cs.historical_outcome.strip()) > 50


def test_generic_scenario_has_no_case_studies():
    cfg = load_config("src/riparia/config/generic_basin_v1.yaml")
    assert cfg.case_studies == []
