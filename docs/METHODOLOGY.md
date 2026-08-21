# Methodology

This document explains what `riparia` computes, why it's structured the
way it is, and where each piece of the method comes from. It's written for
someone thinking about the negotiation-methodology side (not just the code)
— if you only want the module map, skip to "Backend structure" below.

## 1. What kind of tool this is

`riparia` is a **two-party negotiation exercise with a physical model
underneath it**. Two fictional (or eventually real-basin-calibrated)
countries — an upstream riparian ("Country A") and a downstream riparian
("Country B") — negotiate a package of terms governing a shared river.
Whatever they agree to gets run through an actual hydrological/economic
simulation, and the negotiation-analytic layer tells both the facilitator
and, at debrief, the parties themselves: how good was this deal, relative
to every other deal that was available?

That last question — "relative to every other deal that was available" —
is the organizing idea, and it comes from a specific tradition in
negotiation research: **negotiation analysis**, as developed by Howard
Raiffa and collaborators (Raiffa 1982; Raiffa, Richardson & Metcalfe 2002).
Negotiation analysis treats a negotiation as a search problem over a space
of possible agreements, evaluates each point in that space by both
parties' own (possibly very different) values, and asks whether the
parties actually found an efficient point or left value on the table. That
framing is why this codebase enumerates the *entire* feasible agreement
space (`issues.py`) rather than only simulating whatever the parties happen
to propose.

## 2. The negotiation-analytic vocabulary, and where it lives in the code

| Concept | What it means | Where it's computed | Source |
|---|---|---|---|
| **Issue / Package** | A negotiable dimension (e.g. "filling window") and a complete set of choices across all of them | `issues.py`: `Issue`, `Package` | Standard multi-issue negotiation framing; see Raiffa 1982, ch. 7-9 |
| **BATNA** (Best Alternative To a Negotiated Agreement) | The value a party gets if talks fail — the reservation point below which no rational party accepts a deal | `payoffs.py`: `batna()`, `config.parties[*].batna` | Term coined in Fisher & Ury 1981 |
| **ZOPA** (Zone of Possible Agreement) | The set of packages that clear *both* parties' BATNAs simultaneously — the only packages a facilitator should bother testing with the parties | `payoffs.py`: `zopa()` | Standard negotiation-analytic term, used throughout Raiffa 1982 |
| **Pareto / efficient frontier** | The set of packages where no other package makes *both* parties better off — the "you can't improve one without hurting the other" boundary | `frontier.py`: `pareto_frontier()` | Raiffa 1982, ch. 9 ("the efficient frontier") |
| **Post-settlement settlement (PSS)** | After an agreement is reached, search for a *different* package that both parties would prefer — the single most important debrief moment, because it shows value that was left on the table even by a "successful" negotiation | `frontier.py`: `post_settlement_search()` | Raiffa's own term; see Raiffa, Richardson & Metcalfe 2002, ch. 11 |
| **Nash bargaining solution** | The frontier point maximizing the product of both parties' gains above their BATNAs — a symmetry-respecting reference point, not a prediction | `frontier.py`: `nash_solution()` | Nash 1950 |
| **Kalai-Smorodinsky solution** | The frontier point giving both parties the same *fraction* of their maximum possible gain — a different fairness axiom than Nash's | `frontier.py`: `kalai_smorodinsky()` | Kalai & Smorodinsky 1975 |
| **Single negotiating text** | Instead of parties trading counter-offers, a facilitator maintains one draft; parties criticize it, the facilitator revises, repeat — reduces positional bargaining | Planned for `engine.py` (not yet built) | Procedure associated with Roger Fisher, used at the 1978 Camp David negotiations; discussed in Fisher & Ury 1981 and Raiffa 1982 |
| **Joint fact-finding** | Parties build a shared, agreed picture of the facts (here: what a candidate package physically does to the river) *before* arguing about what's fair | The Streamlit "Simulate" tab (planned) exposes the same hydrological simulation to both parties | Standard consensus-building practice; see Susskind & Cruikshank 1987 |
| **Value function assigned from config, not elicited** ("PON-style") | Rather than eliciting each party's real preferences (which real negotiators won't disclose), the facilitator assigns illustrative weights so the exercise can run | `payoffs.py`: `ValueFunction`, `indus_style_v1.yaml` `parties.*.weights` | Standard practice for negotiation-simulation design in the tradition of Harvard's Program on Negotiation (PON) |

**How these fit together, end to end:** `issues.py` enumerates every
feasible package → `system.py` runs the physical model for each one →
`payoffs.py` scores every package for both parties → `payoffs.py.zopa()`
filters to the ones that clear both BATNAs → `frontier.py.pareto_frontier()`
finds the efficient boundary of that set → when the parties actually settle
on something, `frontier.py.efficiency_loss()` and `post_settlement_search()`
tell you how far from the frontier they landed and what they missed. The
Nash and Kalai-Smorodinsky points are reference markers shown at debrief,
never targets the facilitator pushes the parties toward.

## 3. Why this needs a physical model at all

A negotiation-analytic layer on its own can score any *hypothetical* set of
issue combinations, but two riparian countries aren't actually negotiating
abstract points — "medium storage capacity" and "moderate release
guarantee" only mean something once you know what they *do* to the river.
That's `hydrology.py` → `reservoir.py` → `system.py`: for a given package,
simulate the actual monthly water balance and hand the negotiation layer
real numbers (how much water reaches Country B in the dry season, how much
firm hydropower Country A gets, and so on). This is what makes
`post_settlement_search()` meaningful instead of a toy exercise: the
packages it finds are *physically real* alternatives, not points made up to
fill in a frontier.

## 4. Backend structure

```
hydrology.py    generate monthly/daily/ensemble river flow
     |
reservoir.py    mass-balance a single reservoir under a release rule
     |
system.py       couple both parties' reservoirs + routing + demand into
                 one basin run for a candidate package  ->  BasinOutputs
     |
economics.py    each party's own crop-mix response to the water it got
     |
payoffs.py      score BasinOutputs for both parties, 0-100  ->  ZOPA
     |
frontier.py     Pareto frontier, PSS, efficiency loss, Nash / K-S points
     |
contingent.py   re-run all of the above across a dry/normal/wet ensemble
                 -- does this agreement hold up under hydrological stress?
```

`issues.py` sits alongside `system.py`: it owns the definition of what a
"package" is and enumerates the full feasible space (2,000-20,000 packages
for the default scenario) that `payoffs.py`/`frontier.py` operate over.

### The methods behind each physical/economic relationship

- **Reservoir operation** (`reservoir.py`): a fill-season/release-season
  rule with a receding-horizon drawdown (empty storage by the end of every
  release season) — a stylised version of standard single-purpose reservoir
  operating rules, not a calibrated real operating manual. See
  LIMITATIONS.md for exactly what this simplifies away.
- **Channel routing** (`system.py: route_muskingum`): the Muskingum method
  (McCarthy 1938), the standard hydraulic routing technique for translating
  an upstream release hydrograph into a downstream arrival hydrograph given
  a reach's travel time (`K`) and a shape parameter (`x`).
- **Daily flow disaggregation** (`hydrology.py:
  disaggregate_to_daily`): PCHIP interpolation of the seasonal curve plus a
  lognormal AR(1) daily multiplier — a standard approach in stochastic
  hydrology for generating a plausible daily trace from monthly statistics
  (see e.g. Salas 1993 for the general family of AR-based disaggregation
  methods); not fit to any observed daily gauge record.
- **Crop water-yield response** (`economics.py`): inspired by the FAO
  crop-water production function (Doorenbos & Kassam 1979, FAO Irrigation
  and Drainage Paper No. 33) — relative yield loss proportional to relative
  water deficit — but **linearised** (this codebase sets the crop yield
  response factor `Ky = 1` implicitly; the FAO method's `Ky` varies by crop
  and growth stage). See LIMITATIONS.md.

## 5. The allocation-mechanism issue: encoding the real tension

A specific scenario dynamic motivated a mid-build redesign: if Country A's
own water demand grows and it stores/diverts more to meet it, that has real
consequences for Country B — and *how* the two countries have agreed to
share the river determines whether B is protected from that or not. That's
what `allocation_mechanism` (an `issues.py` issue, computed in
`system.py: _make_floor_fn`) represents: not "how much water A must
release" as a single negotiated number, but "by what *rule* is A's release
computed":

| Mechanism | Rule | Real-world echo |
|---|---|---|
| `fixed_volume` | Flat MCM/month regardless of actual flow | Simplest treaty form; strong in name, but only as protective as its number relative to actual flow — see LIMITATIONS.md's empirical finding that a fixed floor can be irrelevant in a wet year and severely binding in a dry one |
| `percentage_of_flow` | A releases a fixed % of that month's *actual* inflow | Scales with hydrology automatically, but so does B's shortfall risk in a drought — protection and risk move together |
| `zonal_formula` | Guaranteed passthrough below a threshold, B gets a share of anything above it | Loosely echoes the two-zone logic of the 1960 Indus Waters Treaty's division of the Indus system — **not a reproduction of the actual IWT allocation**, see LIMITATIONS.md |
| `adaptive_rule_curve` | A's obligation depends on A's *own* current storage relative to a target curve | A real reservoir-operations pattern (rule-curve operation), giving A discretion but not unlimited discretion |

This is also why `agricultural_outcomes()` in `economics.py` and A's own
consumptive demand (`basin.demand.a_consumptive_shape` in the scenario
config) exist: A's growing demand is subtracted from its reservoir release
*before* the water reaches the border, so a weak `allocation_mechanism`
genuinely can let A's domestic growth erode what B receives — the
mechanism the parties negotiate is what stands (or doesn't) between A's
growth and B's entitlement.

## 6. Why crop-mix isn't a `Package` field

Continuous variables (like "what fraction of Country B's land is planted
with a water-intensive crop") don't fit `issues.py`'s fully-enumerated
discrete package space without exploding it combinatorially (adding one
more roughly-continuous dimension per party would multiply the ~2,880-
package space by 25-100x, well past the 2,000-20,000 target that keeps
`pareto_frontier()`'s O(n²) brute-force dominance check fast). More
importantly, it's also not what real treaties negotiate: countries
negotiate volumes, timing, and allocation *mechanisms* — not each other's
farmers' cropping choices. So crop-mix is each party's own continuous
domestic policy response to whatever a candidate package delivers,
computed by `economics.py` from a fixed default in the scenario config for
all frontier/ZOPA analysis, and explorable via a slider in the facilitator
app as a "what if we planted differently" sensitivity display layered on
top of — not inside — the negotiated frontier.

## References

- Doorenbos, J., & Kassam, A. H. (1979). *Yield Response to Water*. FAO
  Irrigation and Drainage Paper No. 33. Rome: Food and Agriculture
  Organization of the United Nations.
- Fisher, R., & Ury, W. (1981). *Getting to Yes: Negotiating Agreement
  Without Giving In*. Boston: Houghton Mifflin.
- Kalai, E., & Smorodinsky, M. (1975). Other Solutions to Nash's Bargaining
  Problem. *Econometrica*, 43(3), 513-518.
- McCarthy, G. T. (1938). *The Unit Hydrograph and Flood Routing*.
  Providence, RI: U.S. Army Corps of Engineers, North Atlantic Division.
- Nash, J. (1950). The Bargaining Problem. *Econometrica*, 18(2), 155-162.
- Raiffa, H. (1982). *The Art and Science of Negotiation*. Cambridge, MA:
  Harvard University Press (Belknap Press).
- Raiffa, H., Richardson, J., & Metcalfe, D. (2002). *Negotiation Analysis:
  The Science and Art of Collaborative Decision Making*. Cambridge, MA:
  Harvard University Press (Belknap Press).
- Salas, J. D. (1993). Analysis and Modeling of Hydrologic Time Series. In
  D. R. Maidment (Ed.), *Handbook of Hydrology* (ch. 19). New York:
  McGraw-Hill.
- Susskind, L., & Cruikshank, J. (1987). *Breaking the Impasse: Consensual
  Approaches to Resolving Public Disputes*. New York: Basic Books.
- The Indus Waters Treaty (1960), between the Government of India and the
  Government of Pakistan, brokered by the International Bank for
  Reconstruction and Development (World Bank). Referenced only for the
  general two-zone structure that loosely inspired the `zonal_formula`
  allocation mechanism — see LIMITATIONS.md for what is and isn't
  reproduced.
