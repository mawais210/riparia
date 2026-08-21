"""Structural climate-change scenarios layered on top of the base
dry/normal/wet stochastic ensemble (hydrology.generate_ensemble):
mean-flow trends, seasonal-timing shifts (e.g. earlier snowmelt), and
discrete multi-year drought sequences. Feeds the same
ContingentRule/evaluate_contingent machinery in contingent.py -- a
multi-year drought sequence is exactly the condition under which
allocation_mechanism choices stop being interchangeable at full-climatology
flow (see LIMITATIONS.md).
"""

# Import necessary libraries
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from riparia.hydrology import generate_ensemble

# Custom Functions


@dataclass(frozen=True)
class ClimateTrend:
    """A structural shift applied to the base climatology before the
    stochastic ensemble is drawn.

    Attributes:
        name: label.
        mean_flow_multiplier: scales the annual total (1.0 = no change,
            0.9 = -10% mean annual flow).
        timing_shift_months: shifts the seasonal curve earlier (negative,
            e.g. earlier snowmelt) or later (positive) by this many months,
            via circular interpolation. 0 = no shift.
        variance_multiplier: scales the ensemble's dry/wet scaling spread
            (see `generate_climate_ensemble`); 1.0 = no change.
    """

    name: str
    mean_flow_multiplier: float
    timing_shift_months: float
    variance_multiplier: float = 1.0


BASELINE = ClimateTrend(name="baseline", mean_flow_multiplier=1.0, timing_shift_months=0.0)
MODERATE_WARMING_2050 = ClimateTrend(
    name="moderate_warming_2050", mean_flow_multiplier=0.90, timing_shift_months=-0.75, variance_multiplier=1.15
)
SEVERE_WARMING_2080 = ClimateTrend(
    name="severe_warming_2080", mean_flow_multiplier=0.78, timing_shift_months=-1.5, variance_multiplier=1.35
)


def _circular_shift_monthly(monthly: pd.Series, shift_months: float) -> pd.Series:
    values = monthly.values
    n = len(values)
    extended_positions = np.arange(-n, 2 * n)
    extended_values = np.tile(values, 3)
    sample_positions = np.arange(n) - shift_months
    shifted_values = np.interp(sample_positions, extended_positions, extended_values)
    return pd.Series(shifted_values, index=monthly.index)


def apply_climate_trend(monthly: pd.Series, trend: ClimateTrend) -> pd.Series:
    """Apply a structural trend to a monthly climatology (MCM/month,
    indexed 1-12): rescale the annual total, then circularly shift the
    seasonal curve by `trend.timing_shift_months`."""
    scaled = monthly * trend.mean_flow_multiplier
    if trend.timing_shift_months != 0:
        return _circular_shift_monthly(scaled, trend.timing_shift_months)
    return scaled.copy()


def generate_climate_ensemble(
    monthly: pd.Series,
    trend: ClimateTrend,
    n_years: int,
    drought_prob: float,
    wet_prob: float,
    seed: int,
    dry_scale: float = 0.7,
    wet_scale: float = 1.3,
    drought_sequence_years: int = 0,
    drought_sequence_prob: float = 0.0,
    drought_sequence_severity: float = 0.85,
) -> pd.DataFrame:
    """Trend-adjust the climatology, draw the usual dry/normal/wet ensemble
    on top of it, then optionally overwrite consecutive-year drought
    sequences (tagged "drought_sequence") to represent sustained multi-year
    stress that a single-year ensemble draw cannot produce.

    Args:
        monthly: base climatology, MCM/month, indexed 1-12.
        trend: structural ClimateTrend to apply before ensemble generation.
        n_years, drought_prob, wet_prob, seed, dry_scale, wet_scale: as
            `hydrology.generate_ensemble`, with the dry/wet scale spread
            widened by `trend.variance_multiplier`.
        drought_sequence_years: length, in years, of an injected drought
            sequence. 0 disables sequence injection.
        drought_sequence_prob: probability, checked at each year not
            already inside a sequence, that a sequence of length
            `drought_sequence_years` starts there.
        drought_sequence_severity: multiplier applied to flow for every
            year inside an injected sequence (compounds with the trend).

    Returns:
        pd.DataFrame like `hydrology.generate_ensemble`'s, tag additionally
        including "drought_sequence" if any were injected.
    """
    trended = apply_climate_trend(monthly, trend)
    spread_dry = 1.0 - (1.0 - dry_scale) * trend.variance_multiplier
    spread_wet = 1.0 + (wet_scale - 1.0) * trend.variance_multiplier
    ensemble = generate_ensemble(trended, n_years, drought_prob, wet_prob, seed, spread_dry, spread_wet)

    if drought_sequence_years > 1 and drought_sequence_prob > 0:
        ensemble = _inject_drought_sequences(
            ensemble, drought_sequence_years, drought_sequence_prob, drought_sequence_severity, seed
        )
    return ensemble


def _inject_drought_sequences(
    ensemble: pd.DataFrame, sequence_years: int, sequence_prob: float, severity: float, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1_000_000)
    years = sorted(ensemble.index.get_level_values("year").unique())
    ensemble = ensemble.copy()

    i = 0
    while i < len(years):
        if i + sequence_years <= len(years) and rng.random() < sequence_prob:
            for j in range(sequence_years):
                year = years[i + j]
                mask = ensemble.index.get_level_values("year") == year
                ensemble.loc[mask, "flow_mcm"] = ensemble.loc[mask, "flow_mcm"] * severity
                ensemble.loc[mask, "tag"] = "drought_sequence"
            i += sequence_years
        else:
            i += 1
    return ensemble
