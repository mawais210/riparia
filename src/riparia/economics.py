"""Each party's own domestic agricultural response to whatever water a
candidate package delivers to it.

This is deliberately NOT a negotiated issue (see docs/METHODOLOGY.md,
"Why crop-mix isn't a Package field"): neither party negotiates the
other's farmers' cropping choices in a water treaty. `crop_mix_fraction`
is each party's own continuous domestic policy lever, explored via a
slider in the facilitator app, layered on top of the 7 negotiated issues.

The water-yield response is a linearised version of the FAO crop-water
production function (Doorenbos & Kassam 1979, FAO Irrigation and Drainage
Paper 33: relative yield loss proportional to relative water deficit) --
see docs/METHODOLOGY.md for the citation and the linearisation this
simplifies away.
"""

# Import necessary libraries
from __future__ import annotations

from dataclasses import dataclass

# Custom Functions


@dataclass(frozen=True)
class CropProfile:
    label: str
    water_use_mcm_per_1000ha: float
    income_musd_per_1000ha: float
    labor_persondays_per_1000ha: float


@dataclass
class AgriculturalOutcome:
    water_used_mcm: float
    income_musd: float
    labor_persondays: float
    water_satisfaction_fraction: float


def _blend(f: float, area: float, high: CropProfile, low: CropProfile, field: str) -> float:
    return f * area * getattr(high, field) + (1 - f) * area * getattr(low, field)


def water_requirement_mcm(
    crop_mix_fraction: float,
    command_area_1000ha: float,
    high_water_crop: CropProfile,
    low_water_crop: CropProfile,
) -> float:
    """Full (unconstrained-by-supply) water requirement, MCM, of the blended
    cropping pattern. This is the same figure used both as a party's
    irrigation/consumptive demand target (system.py) and as the denominator
    of `agricultural_outcomes`'s satisfaction fraction, so the two stay
    consistent with each other."""
    f = min(max(crop_mix_fraction, 0.0), 1.0)
    return _blend(f, command_area_1000ha, high_water_crop, low_water_crop, "water_use_mcm_per_1000ha")


def agricultural_outcomes(
    water_available_mcm: float,
    crop_mix_fraction: float,
    command_area_1000ha: float,
    high_water_crop: CropProfile,
    low_water_crop: CropProfile,
) -> AgriculturalOutcome:
    """Blend two crop archetypes by `crop_mix_fraction` (share of command
    area in `high_water_crop`), then scale income/labor by how much of the
    resulting water requirement `water_available_mcm` actually covers.

    Args:
        water_available_mcm: water actually delivered/diverted for this
            party's agricultural use this period, MCM.
        crop_mix_fraction: 0-1, share of `command_area_1000ha` planted with
            `high_water_crop` (the rest with `low_water_crop`).
        command_area_1000ha: total irrigated command area, thousand hectares.
        high_water_crop, low_water_crop: the two crop archetypes to blend.

    Returns:
        AgriculturalOutcome with water used, income (MUSD), labor demand
        (person-days), and the water-satisfaction fraction actually
        achieved (1.0 = full requirement met).
    """
    f = min(max(crop_mix_fraction, 0.0), 1.0)
    water_required = water_requirement_mcm(f, command_area_1000ha, high_water_crop, low_water_crop)
    potential_income = _blend(f, command_area_1000ha, high_water_crop, low_water_crop, "income_musd_per_1000ha")
    potential_labor = _blend(f, command_area_1000ha, high_water_crop, low_water_crop, "labor_persondays_per_1000ha")

    satisfaction = 1.0 if water_required <= 0 else min(1.0, water_available_mcm / water_required)

    return AgriculturalOutcome(
        water_used_mcm=min(water_available_mcm, water_required),
        income_musd=satisfaction * potential_income,
        labor_persondays=satisfaction * potential_labor,
        water_satisfaction_fraction=satisfaction,
    )
