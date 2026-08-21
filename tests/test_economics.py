import pytest

from riparia.economics import CropProfile, agricultural_outcomes, water_requirement_mcm

HIGH = CropProfile(label="high", water_use_mcm_per_1000ha=1.2, income_musd_per_1000ha=2.5, labor_persondays_per_1000ha=180_000)
LOW = CropProfile(label="low", water_use_mcm_per_1000ha=0.5, income_musd_per_1000ha=1.0, labor_persondays_per_1000ha=70_000)


def test_water_requirement_is_linear_blend():
    req_all_high = water_requirement_mcm(1.0, 1000, HIGH, LOW)
    req_all_low = water_requirement_mcm(0.0, 1000, HIGH, LOW)
    req_half = water_requirement_mcm(0.5, 1000, HIGH, LOW)
    assert req_all_high == pytest.approx(1200)
    assert req_all_low == pytest.approx(500)
    assert req_half == pytest.approx((req_all_high + req_all_low) / 2)


def test_full_satisfaction_when_water_ample():
    out = agricultural_outcomes(water_available_mcm=10_000, crop_mix_fraction=0.5, command_area_1000ha=1000, high_water_crop=HIGH, low_water_crop=LOW)
    assert out.water_satisfaction_fraction == pytest.approx(1.0)
    assert out.income_musd == pytest.approx(0.5 * 1000 * 2.5 + 0.5 * 1000 * 1.0)


def test_partial_satisfaction_scales_income_and_labor():
    full = agricultural_outcomes(water_available_mcm=10_000, crop_mix_fraction=0.5, command_area_1000ha=1000, high_water_crop=HIGH, low_water_crop=LOW)
    required = water_requirement_mcm(0.5, 1000, HIGH, LOW)
    half = agricultural_outcomes(water_available_mcm=required / 2, crop_mix_fraction=0.5, command_area_1000ha=1000, high_water_crop=HIGH, low_water_crop=LOW)
    assert half.water_satisfaction_fraction == pytest.approx(0.5)
    assert half.income_musd == pytest.approx(full.income_musd * 0.5)
    assert half.labor_persondays == pytest.approx(full.labor_persondays * 0.5)


def test_higher_crop_mix_fraction_increases_water_requirement_and_income_potential():
    low_mix = agricultural_outcomes(water_available_mcm=10_000, crop_mix_fraction=0.1, command_area_1000ha=1000, high_water_crop=HIGH, low_water_crop=LOW)
    high_mix = agricultural_outcomes(water_available_mcm=10_000, crop_mix_fraction=0.9, command_area_1000ha=1000, high_water_crop=HIGH, low_water_crop=LOW)
    assert high_mix.water_used_mcm > low_mix.water_used_mcm
    assert high_mix.income_musd > low_mix.income_musd


def test_crop_mix_fraction_clipped_to_valid_range():
    out_over = agricultural_outcomes(water_available_mcm=10_000, crop_mix_fraction=1.5, command_area_1000ha=1000, high_water_crop=HIGH, low_water_crop=LOW)
    out_at_one = agricultural_outcomes(water_available_mcm=10_000, crop_mix_fraction=1.0, command_area_1000ha=1000, high_water_crop=HIGH, low_water_crop=LOW)
    assert out_over.income_musd == pytest.approx(out_at_one.income_musd)
