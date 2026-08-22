"""Each party's own domestic agricultural response to whatever water a
candidate package delivers to it.

This is deliberately NOT a negotiated issue (see docs/METHODOLOGY.md,
"Why crop-mix isn't a Package field"): neither party negotiates the
other's farmers' cropping choices in a water treaty. Each party's cropping
pattern -- how many hectares it plants of each named crop -- is its own
continuous domestic policy lever, explored via sliders in the facilitator
app, layered on top of the negotiated issues.

Water use, income, and labor are computed per crop and summed, so a
party's total agricultural demand and outcomes reflect an actual cropping
pattern (e.g. wheat + rice + cotton + sugarcane), not a single blended
archetype. The water-yield response is a linearised version of the FAO
crop-water production function (Doorenbos & Kassam 1979, FAO Irrigation
and Drainage Paper 33: relative yield loss proportional to relative water
deficit) -- see docs/METHODOLOGY.md for the citation and the linearisation
this simplifies away.
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


def water_requirement_mcm(areas_1000ha: dict[str, float], crops: dict[str, CropProfile]) -> float:
    """Full (unconstrained-by-supply) water requirement, MCM, of a cropping
    pattern: sum over every crop of area x that crop's water use per
    1000ha. This is the same figure used both as a party's irrigation or
    consumptive demand target (system.py) and as the denominator of
    `agricultural_outcomes`'s satisfaction fraction, so the two stay
    consistent with each other.

    Args:
        areas_1000ha: crop name -> planted area, thousand hectares.
        crops: crop name -> CropProfile. Every key in `areas_1000ha` must
            also be a key in `crops`.
    """
    return sum(areas_1000ha[name] * crops[name].water_use_mcm_per_1000ha for name in areas_1000ha)


def agricultural_outcomes(
    water_available_mcm: float,
    areas_1000ha: dict[str, float],
    crops: dict[str, CropProfile],
) -> AgriculturalOutcome:
    """Score a cropping pattern by how much of its water requirement
    `water_available_mcm` actually covers, scaling income and labor by that
    same satisfaction fraction across every crop.

    Args:
        water_available_mcm: water actually delivered/diverted for this
            party's agricultural use this period, MCM.
        areas_1000ha: crop name -> planted area, thousand hectares.
        crops: crop name -> CropProfile.

    Returns:
        AgriculturalOutcome with water used, income (MUSD), labor demand
        (person-days), and the water-satisfaction fraction actually
        achieved (1.0 = full requirement met, across every crop).
    """
    water_required = water_requirement_mcm(areas_1000ha, crops)
    potential_income = sum(areas_1000ha[name] * crops[name].income_musd_per_1000ha for name in areas_1000ha)
    potential_labor = sum(areas_1000ha[name] * crops[name].labor_persondays_per_1000ha for name in areas_1000ha)

    satisfaction = 1.0 if water_required <= 0 else min(1.0, water_available_mcm / water_required)

    return AgriculturalOutcome(
        water_used_mcm=min(water_available_mcm, water_required),
        income_musd=satisfaction * potential_income,
        labor_persondays=satisfaction * potential_labor,
        water_satisfaction_fraction=satisfaction,
    )
