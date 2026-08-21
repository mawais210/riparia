# Limitations

This document names every place a physical, economic, or negotiation-analytic
simplification was made, and why. `riparia` is a *teaching and negotiation-
methodology instrument*, not a calibrated hydrological or economic model of
any real basin. If you use it to make an argument about a real basin, the
boundary below is the line between "what the model computed" and "what the
scenario author assumed."

See `docs/METHODOLOGY.md` for the citations behind the relationships that
*are* grounded in a published method (crop-water response, Muskingum
routing, the negotiation-analytic framework). This file is about what's
*not* grounded, or is grounded but simplified.

## Hydrology

- **Reservoir operating rule** (`reservoir.py`) is a stylised "receding-
  horizon drawdown to zero storage by the end of every release season" rule.
  Real reservoir operators do not target zero storage; they hold carryover
  storage for multi-year drought resilience. This means a single average
  year's simulation cannot show the mechanism-dependent protection that
  `allocation_mechanism` provides — that only shows up under genuine
  hydrological stress across *years* (see `contingent.py`). Verified
  empirically during the v0.2 build: `fixed_low` vs `fixed_high` release
  floors are numerically indistinguishable at full climatology flow (spill
  dominates, or the floor is simply never binding) and only diverge once the
  trace is scaled down to dry-year levels.
- **Muskingum routing** uses a single reach, single travel time, and no
  attenuation calibration against any real channel geometry. `travel_time_days`
  defaults near zero for the shipped scenario, so routing effectively
  passes through unchanged; the interface exists for scenarios that need it.
- **Daily disaggregation** (`hydrology.disaggregate_to_daily`) uses a
  lognormal AR(1) multiplier layered on a PCHIP-interpolated seasonal curve.
  This is a statistical convenience, not a stochastic weather generator; it
  has no representation of actual storm events, snowmelt timing shifts, or
  spatial correlation across sub-catchments.
- **Tributary inflow below the dam** is modeled as a fixed fraction of that
  month's mainstem inflow at the dam (`tributary_inflow_fraction`), not an
  independently gauged tributary series.

## Hydropower

- `firm_hydropower_gwh_per_year` uses an illustrative constant
  (`k_energy = 0.01` GWh per MCM at full head) and a linear "head factor"
  interpolated from live storage capacity (0.3 at zero storage to 1.0 at
  15,000 MCM). This is **not** derived from any turbine curve, actual dam
  height, or plant factor — it exists so the model has *a* hydropower
  signal that responds directionally to storage capacity and release
  reliability, not to produce an energy estimate anyone should cite.

## Economics / demand side

- **Two crop archetypes only** (`economics.py`): a stylised "water-intensive
  cash crop" and "low-water staple," blended linearly by
  `default_crop_mix_fraction`. Real cropping patterns have dozens of crops
  with nonlinear substitution; this is a deliberately coarse two-point
  approximation.
- **Linearised yield-water response**: income and labor scale *linearly*
  with the water-satisfaction fraction (`min(1, water_available /
  water_required)`). The FAO method this is inspired by (Doorenbos & Kassam
  1979, see `docs/METHODOLOGY.md`) uses a crop-specific yield-response
  factor `Ky` and is not linear in general; this model sets `Ky = 1`
  implicitly for every crop and every growth stage, which is a real
  simplification, not just a rounding choice.
- **Crop mix is fixed at `default_crop_mix_fraction`** for all package
  scoring, ZOPA, and frontier computation. The facilitator app's crop-mix
  slider is a sensitivity/what-if display layered on top, not a live input
  to the negotiated frontier — see `docs/METHODOLOGY.md`, "Why crop-mix
  isn't a Package field."
- **Command areas, crop water-use/income/labor coefficients** in
  `indus_style_v1.yaml` are illustrative round numbers chosen to produce a
  plausible order of magnitude (millions of hectares, single-digit billions
  of USD), not sourced from any agricultural census or FAO AQUASTAT extract.
- A's consumptive demand and B's irrigation demand are both derived from the
  *same* two-crop economics, scaled by each party's own command area and
  crop mix — there is no representation of industrial, municipal, or
  environmental flow demand for either party.

## Allocation mechanism

- The `zonal_formula` level of `allocation_mechanism` is a stylised
  two-zone (baseline-passthrough + shared-surplus) rule loosely inspired by
  the 1960 Indus Waters Treaty's division of the Indus system into Eastern
  and Western rivers. It is **not** a reproduction of the actual IWT
  allocation formula, volumes, or legal text — see `docs/METHODOLOGY.md`
  for what it does compute.
- `adaptive_rule_curve`'s `target_storage_fraction` curve is a single
  generic annual shape applied regardless of which `filling_window` level a
  package also selects, so it does not perfectly track an arbitrarily
  chosen fill season — an approximation, not a bug.

## Negotiation-analytic layer

- Value functions are **assigned from configuration** (PON-style), not
  elicited from real negotiators. Every `issue_scores` and `outcome_terms`
  weight in `indus_style_v1.yaml` is the scenario author's judgment about
  what each party would plausibly value, calibrated only so the resulting
  ZOPA and Pareto frontier are non-degenerate (see the frontier smoke test
  in `tests/test_frontier.py`) — not validated against any real
  negotiators' actual preferences.
- `pareto_frontier` is computed by brute-force O(n²) pairwise dominance
  checking over the fully enumerated package space (a few thousand
  packages). This does not scale to a continuous-issue or very-large
  discrete space; see `docs/METHODOLOGY.md`'s note on why crop-mix is kept
  out of `Package` for exactly this reason.

## Scope not yet built

Climate-trend/event scenarios beyond the dry/normal/wet stochastic
ensemble, `engine.py`/round-state machinery, the facilitator interface(s),
and calibration to any real basin (Indus, Nile, Central Asian rivers) are
tracked as follow-on work — see the project's v0.2 plan, not covered by
this document.
