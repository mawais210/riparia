import pytest

from riparia.economics import CropProfile, agricultural_outcomes, water_requirement_mcm

WHEAT = CropProfile(label="wheat", water_use_mcm_per_1000ha=4.5, income_musd_per_1000ha=0.4, labor_persondays_per_1000ha=40_000)
RICE = CropProfile(label="rice", water_use_mcm_per_1000ha=16.0, income_musd_per_1000ha=0.9, labor_persondays_per_1000ha=140_000)
COTTON = CropProfile(label="cotton", water_use_mcm_per_1000ha=7.0, income_musd_per_1000ha=1.1, labor_persondays_per_1000ha=110_000)
SUGARCANE = CropProfile(label="sugarcane", water_use_mcm_per_1000ha=22.0, income_musd_per_1000ha=1.6, labor_persondays_per_1000ha=160_000)

CROPS = {"wheat": WHEAT, "rice": RICE, "cotton": COTTON, "sugarcane": SUGARCANE}


def test_water_requirement_sums_across_crops():
    areas = {"wheat": 800, "rice": 100, "cotton": 250, "sugarcane": 50}
    req = water_requirement_mcm(areas, CROPS)
    expected = 800 * 4.5 + 100 * 16.0 + 250 * 7.0 + 50 * 22.0
    assert req == pytest.approx(expected)


def test_water_requirement_single_crop():
    assert water_requirement_mcm({"wheat": 1000}, CROPS) == pytest.approx(4500)
    assert water_requirement_mcm({"sugarcane": 1000}, CROPS) == pytest.approx(22000)


def test_full_satisfaction_when_water_ample():
    areas = {"wheat": 500, "rice": 500}
    out = agricultural_outcomes(water_available_mcm=100_000, areas_1000ha=areas, crops=CROPS)
    assert out.water_satisfaction_fraction == pytest.approx(1.0)
    assert out.income_musd == pytest.approx(500 * 0.4 + 500 * 0.9)
    assert out.labor_persondays == pytest.approx(500 * 40_000 + 500 * 140_000)


def test_partial_satisfaction_scales_income_and_labor():
    areas = {"wheat": 500, "rice": 500}
    required = water_requirement_mcm(areas, CROPS)
    full = agricultural_outcomes(water_available_mcm=required * 2, areas_1000ha=areas, crops=CROPS)
    half = agricultural_outcomes(water_available_mcm=required / 2, areas_1000ha=areas, crops=CROPS)
    assert half.water_satisfaction_fraction == pytest.approx(0.5)
    assert half.income_musd == pytest.approx(full.income_musd * 0.5)
    assert half.labor_persondays == pytest.approx(full.labor_persondays * 0.5)


def test_more_water_intensive_cropping_pattern_raises_requirement_and_income_potential():
    low_water_pattern = {"wheat": 1000}
    high_water_pattern = {"sugarcane": 1000}
    low = agricultural_outcomes(water_available_mcm=100_000, areas_1000ha=low_water_pattern, crops=CROPS)
    high = agricultural_outcomes(water_available_mcm=100_000, areas_1000ha=high_water_pattern, crops=CROPS)
    assert high.water_used_mcm > low.water_used_mcm
    assert high.income_musd > low.income_musd


def test_four_crop_pattern_matches_manual_sum():
    areas = {"wheat": 4500, "rice": 1500, "cotton": 3000, "sugarcane": 700}
    out = agricultural_outcomes(water_available_mcm=1_000_000, areas_1000ha=areas, crops=CROPS)
    expected_income = 4500 * 0.4 + 1500 * 0.9 + 3000 * 1.1 + 700 * 1.6
    expected_labor = 4500 * 40_000 + 1500 * 140_000 + 3000 * 110_000 + 700 * 160_000
    assert out.income_musd == pytest.approx(expected_income)
    assert out.labor_persondays == pytest.approx(expected_labor)


def test_zero_requirement_is_fully_satisfied():
    out = agricultural_outcomes(water_available_mcm=0.0, areas_1000ha={}, crops=CROPS)
    assert out.water_satisfaction_fraction == pytest.approx(1.0)
    assert out.income_musd == 0.0
