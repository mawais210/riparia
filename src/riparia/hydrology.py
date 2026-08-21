"""Deterministic-given-seed monthly/daily flow generation for a single gauge.

Units: all flow/volume quantities are in million cubic metres (MCM). Monthly
values are MCM for that calendar month (a volume, not a rate). Daily values
are MCM/day. All functions that take a `seed` are deterministic for that seed.
"""

# Import necessary libraries
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

# Custom Functions

_MONTHS = list(range(1, 13))
_DAYS_IN_MONTH_NONLEAP = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


def generate_climatology(shape: dict[int, float], mean_flow: float) -> pd.Series:
    """Normalise a 12-month seasonal shape and scale to an annual total.

    Args:
        shape: mapping month (1-12) -> relative weight, need not sum to 1.
        mean_flow: mean annual total flow volume, MCM/year.

    Returns:
        pd.Series indexed 1..12 of monthly flow volumes, MCM/month, summing
        to `mean_flow`.
    """
    if set(shape.keys()) != set(_MONTHS):
        raise ValueError("shape must define exactly months 1..12")
    raw = pd.Series({m: shape[m] for m in _MONTHS}).sort_index()
    if (raw < 0).any():
        raise ValueError("shape weights must be non-negative")
    normalized = raw / raw.sum()
    return normalized * mean_flow


def _day_of_year_midpoints(year: int) -> np.ndarray:
    """Day-of-year (1-indexed) of each month's mid-point, for PCHIP anchor points."""
    days = np.array([_DAYS_IN_MONTH_NONLEAP[m] + (1 if (m == 2 and _is_leap(year)) else 0) for m in _MONTHS])
    cumulative_start = np.concatenate([[0], np.cumsum(days)[:-1]])
    return cumulative_start + days / 2.0


def _is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def disaggregate_to_daily(
    monthly: pd.Series,
    year: int,
    cv_low: float,
    cv_high: float,
    phi: float,
    seed: int,
) -> pd.Series:
    """Disaggregate a 12-month volume series into a daily rate series (MCM/day).

    Method: PCHIP interpolation of monthly mean daily rate through month
    mid-points to get a smooth seasonal curve, multiplied by a lognormal
    AR(1) daily multiplier whose standard deviation scales with the
    climatological flow level (`cv_low` in the driest month, `cv_high` in
    the wettest). Each month's daily values are then rescaled by a constant
    so the monthly mean of the daily series exactly equals the input
    monthly mean rate (volume / days-in-month).

    Args:
        monthly: pd.Series indexed 1..12, monthly volumes in MCM.
        year: calendar year, used only to size February and day-of-year axis.
        cv_low, cv_high: coefficient of variation of the daily multiplier in
            the lowest- and highest-flow months respectively.
        phi: AR(1) persistence of the log-multiplier, in [0, 1).
        seed: RNG seed, makes the result deterministic.

    Returns:
        pd.Series indexed by a daily DatetimeIndex for `year`, values MCM/day.
    """
    if set(monthly.index) != set(_MONTHS):
        raise ValueError("monthly must be indexed 1..12")
    rng = np.random.default_rng(seed)

    days_in_month = {m: _DAYS_IN_MONTH_NONLEAP[m] + (1 if (m == 2 and _is_leap(year)) else 0) for m in _MONTHS}
    monthly_rate = pd.Series({m: monthly[m] / days_in_month[m] for m in _MONTHS}).sort_index()

    mids = _day_of_year_midpoints(year)
    interpolator = PchipInterpolator(mids, monthly_rate.values, extrapolate=True)

    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    doy = np.arange(1, len(dates) + 1)
    smooth_rate = np.clip(interpolator(doy), a_min=0.0, a_max=None)

    share = monthly_rate / monthly_rate.sum()
    lo, hi = share.min(), share.max()
    span = max(hi - lo, 1e-12)
    day_month = dates.month.values
    cv_by_day = np.array([cv_low + (cv_high - cv_low) * (share[m] - lo) / span for m in day_month])

    n = len(dates)
    innovations = rng.normal(size=n)
    log_mult = np.zeros(n)
    log_mult[0] = cv_by_day[0] * innovations[0]
    for t in range(1, n):
        log_mult[t] = phi * log_mult[t - 1] + np.sqrt(max(1 - phi**2, 0.0)) * cv_by_day[t] * innovations[t]

    daily = smooth_rate * np.exp(log_mult)
    result = pd.Series(daily, index=dates)

    # Rescale each month so its mean exactly matches the target monthly rate.
    for m in _MONTHS:
        mask = result.index.month == m
        current_mean = result[mask].mean()
        if current_mean > 0:
            result[mask] = result[mask] * (monthly_rate[m] / current_mean)
        else:
            result[mask] = monthly_rate[m]

    return result


def generate_ensemble(
    monthly: pd.Series,
    n_years: int,
    drought_prob: float,
    wet_prob: float,
    seed: int,
    dry_scale: float = 0.7,
    wet_scale: float = 1.3,
) -> pd.DataFrame:
    """Generate an ensemble of annual hydrological traces tagged dry/normal/wet.

    Dry years scale the whole hydrograph down by `dry_scale` and additionally
    attenuate the driest quartile of months (deepening the recession); wet
    years scale up by `wet_scale` and additionally boost the wettest quartile
    (peakier hydrograph). Normal years are the unscaled climatology.

    Args:
        monthly: pd.Series indexed 1..12, monthly volumes in MCM (climatology).
        n_years: number of annual traces to generate.
        drought_prob, wet_prob: probability a given year is tagged dry / wet;
            remainder is normal. Must satisfy drought_prob + wet_prob <= 1.
        seed: RNG seed.
        dry_scale, wet_scale: uniform multipliers for dry/wet years.

    Returns:
        pd.DataFrame, index = (year, month) MultiIndex, columns ["flow_mcm",
        "tag"], tag in {"dry", "normal", "wet"}.
    """
    if drought_prob + wet_prob > 1.0 + 1e-9:
        raise ValueError("drought_prob + wet_prob must not exceed 1")
    rng = np.random.default_rng(seed)

    q1, q3 = monthly.quantile(0.25), monthly.quantile(0.75)
    driest_mask = monthly <= q1
    wettest_mask = monthly >= q3

    rows = []
    probs = [drought_prob, wet_prob, 1.0 - drought_prob - wet_prob]
    tags = rng.choice(["dry", "wet", "normal"], size=n_years, p=probs)
    for year_idx, tag in enumerate(tags):
        trace = monthly.copy()
        if tag == "dry":
            trace = trace * dry_scale
            trace[driest_mask] = trace[driest_mask] * dry_scale
        elif tag == "wet":
            trace = trace * wet_scale
            trace[wettest_mask] = trace[wettest_mask] * wet_scale
        for m in _MONTHS:
            rows.append({"year": year_idx, "month": m, "flow_mcm": trace[m], "tag": tag})

    df = pd.DataFrame(rows).set_index(["year", "month"])
    return df


def aggregate(series: pd.Series, interval_months: int) -> pd.Series:
    """Sum a monthly (or finer, pre-resampled-to-monthly) volume series over
    fixed-width windows of `interval_months` consecutive entries.

    Args:
        series: pd.Series of volumes (MCM), any monotonic index.
        interval_months: number of consecutive entries per aggregation window.

    Returns:
        pd.Series indexed 0..n_windows-1 (window number), summed volumes.
        A trailing partial window (fewer than `interval_months` entries) is
        included as-is.
    """
    if interval_months <= 0:
        raise ValueError("interval_months must be positive")
    values = series.values
    n = len(values)
    n_windows = int(np.ceil(n / interval_months))
    out = {}
    for w in range(n_windows):
        chunk = values[w * interval_months : (w + 1) * interval_months]
        out[w] = chunk.sum()
    return pd.Series(out)
