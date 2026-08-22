# Methodology

This document explains what `riparia` computes and where each piece of the
method comes from. It assumes no prior background in hydrology or
negotiation theory; the glossary in section 1 defines terms from both
fields plainly before the rest of the document uses them.

## 1. Glossary

### Water and hydrology terms

- **MCM**: million cubic metres. The volume unit used throughout this
  codebase for river flow, reservoir storage, and demand.
- **Water year**: a 12-month accounting period that starts at whatever
  month makes sense for a basin's hydrology (often the start of the dry
  season), rather than always starting in January.
- **Streamflow / discharge**: the volume of water passing a point in the
  river per unit time.
- **Climatology**: the average seasonal pattern of streamflow across many
  years, e.g. "this river typically carries 18% of its annual flow in
  August."
- **Ensemble**: a set of simulated flow traces (here: dry, normal, and wet
  years) used to test how an agreement performs across a range of
  hydrological conditions, not just one average year.
- **Reservoir live storage**: the portion of a reservoir's volume that can
  actually be filled and drawn down for water management, as opposed to
  dead storage that sits below the lowest outlet.
- **Fill season / release season**: the months when a reservoir is
  operated to accumulate water versus the months it is drawn down to meet
  downstream needs.
- **Spill**: water released because the reservoir is full, not because an
  operating rule called for it.
- **Routing**: translating a flow hydrograph measured at one point in a
  river into the (delayed, smoothed) hydrograph it produces further
  downstream. The Muskingum method is a standard technique for this.
- **Firm yield / firm power**: the minimum, most-dependable output (water
  or hydropower) a system can deliver even in a bad year, as distinct from
  its average output.
- **Command area**: the irrigated land area served by a given water
  supply.
- **Consumptive use**: water withdrawn and not returned to the river (used
  for irrigation, evaporation, etc.), as opposed to water that is diverted
  and then returned downstream.
- **Riparian**: a country or party with territory bordering or crossed by
  a shared river. "Upstream riparian" and "downstream riparian" describe
  relative position along the river.

### Negotiation terms

- **Issue**: one negotiable dimension of the agreement (for example, how
  much storage capacity the upstream reservoir gets).
- **Package**: a complete set of choices, one per issue. Negotiators trade
  whole packages, not single issues in isolation.
- **BATNA**: Best Alternative To a Negotiated Agreement. What a party gets
  if talks fail. No rational party accepts a deal worse than its BATNA.
- **ZOPA**: Zone of Possible Agreement. The set of packages that beat both
  parties' BATNAs at once. Also called the **contract zone**.
- **Pareto frontier** (or **efficient frontier**): the set of packages
  where no other package makes both parties better off at the same time.
  Moving off the frontier means at least one party could have done better
  without the other doing worse.
- **Post-settlement settlement (PSS)**: after a deal is signed, a search
  for a different package both parties would have preferred. It shows
  value that was left unclaimed even by an agreement both sides accepted.
- **Nash bargaining solution**: a reference point on the frontier,
  computed as the package that maximizes the product of both parties'
  gains above their BATNAs.
- **Kalai-Smorodinsky solution**: a different reference point, the package
  giving both parties the same fraction of their best possible gain.
- **Single negotiating text**: a negotiating procedure where a facilitator
  keeps one draft agreement; parties critique it instead of trading
  counter-offers, and the facilitator revises it based on that feedback.
- **Joint fact-finding**: parties agreeing on the facts of a situation
  before arguing about what's fair, so the argument is about values, not
  about whose numbers are right.
- **Distributive bargaining**: dividing a fixed amount of value. One
  party's gain is the other's loss.
- **Integrative bargaining**: finding trades that increase the total value
  available, so both parties can gain at once.
- **Logrolling**: trading across issues that the two parties weight
  differently, so each side gives up something it cares about less in
  exchange for something it cares about more.
- **Value function**: a scoring formula that converts a package (or its
  physical consequences) into a single number representing how good that
  outcome is for one party.

## 2. What kind of tool this is

`riparia` is a two-party negotiation exercise built on a physical model.
An upstream riparian ("Country A") and a downstream riparian ("Country B")
negotiate a package of terms governing a shared river. Whatever they agree
to is run through a hydrological and economic simulation, and a
negotiation-analytic layer scores the result for both parties: how good is
this deal, compared to every other deal that was actually available?

That comparison is the organizing idea, and it comes from a specific
research tradition: negotiation analysis, developed by Howard Raiffa and
collaborators (Raiffa 1982; Raiffa, Richardson & Metcalfe 2002).
Negotiation analysis treats a negotiation as a search over a space of
possible agreements, scores each point in that space by both parties' own
values, and checks whether the parties found an efficient point or left
value on the table. This is why `issues.py` enumerates the entire feasible
agreement space: it lets the tool score real alternatives instead of only
whatever the parties happen to propose.

## 3. Negotiation analysis: the underlying theory

### 3.1 Creating value vs. claiming it

A negotiation is two games at once. One is cooperative: can the parties
find trades that make both better off? The other is competitive: given a
set of efficient deals, which one do they land on, and who gets more of
the joint gain? Lax & Sebenius (1986) call the tension between these two
games the negotiator's dilemma. Behavior that helps a party claim more
value, such as bluffing about its priorities or holding firm, tends to
undermine the information-sharing that would let the parties find more
value to claim in the first place.

`frontier.py: post_settlement_search()` exists as a separate step after
settlement for this reason. It lets a facilitator show, once the
competitive game is over and the deal is signed, how much cooperative
value went unclaimed because the parties never fully disclosed what they
valued. It's a teaching device for the dilemma, not a fix for it.

### 3.2 Multi-attribute value functions

`payoffs.py`'s `ValueFunction` is a weighted sum of per-issue and
per-outcome scores, each on a 0-100 scale. This is a simplified
application of multi-attribute utility theory (Keeney & Raiffa 1976): a
framework for representing preferences over an outcome with several
attributes as one scalar value, so different packages become comparable
on a single scale. The framework forces a specific modeling choice into
the open: whose weights, on which attributes. Changing `parties.A.weights`
in a scenario file changes Country A's entire value function, not a
cosmetic setting.

### 3.3 Integrative bargaining and logrolling

Walton & McKersie (1965) drew the classic distinction between
distributive and integrative bargaining. Logrolling is the mechanism for
integrative gains across issues: if Country A cares much more about
storage capacity than about financing, and Country B is the reverse, a
package trading more storage for A against a larger payment to B is a
trade both sides value, not a concession. This is why `financing_transfer`
exists as an issue at all (without a transferable dimension the frontier
collapses into a purely distributive exercise), and why `logging_.py:
classify_move()` labels a move "integrative" when it grows the joint score
rather than reallocating a fixed total.

### 3.4 Even swaps

Hammond, Keeney & Raiffa (1998) describe even swaps: a manual technique
for resolving multi-attribute tradeoffs by asking how much of one
attribute a party would give up for one more unit of another, and using
the answers to collapse attributes together. This codebase encodes those
tradeoff rates as fixed weights rather than eliciting them interactively.
A facilitator tool that walks a party through even swaps live would
extend the Negotiate tab; it isn't built yet.

## 4. Concepts and where they live in the code

| Concept | Where it's computed | Source |
|---|---|---|
| Issue / Package | `issues.py`: `Issue`, `Package` | Raiffa 1982, ch. 7-9 |
| BATNA | `payoffs.py`: `batna()`, `config.parties[*].batna` | Fisher & Ury 1981 |
| ZOPA | `payoffs.py`: `zopa()` | Raiffa 1982 |
| Pareto frontier | `frontier.py`: `pareto_frontier()` | Raiffa 1982, ch. 9 |
| Post-settlement settlement | `frontier.py`: `post_settlement_search()` | Raiffa, Richardson & Metcalfe 2002, ch. 11 |
| Nash bargaining solution | `frontier.py`: `nash_solution()` | Nash 1950 |
| Kalai-Smorodinsky solution | `frontier.py`: `kalai_smorodinsky()` | Kalai & Smorodinsky 1975 |
| Single negotiating text | `engine.py`: `Exercise.revise_text()`, `Exercise.submit_criticism()` | Procedure used at the 1978 Camp David negotiations; Fisher & Ury 1981, Raiffa 1982 |
| Joint fact-finding | `app/facilitator.py`, Simulate tab | Susskind & Cruikshank 1987 |
| Value function assigned from config ("PON-style") | `payoffs.py`: `ValueFunction`, scenario YAML `parties.*.weights` | Design practice from Harvard's Program on Negotiation |

How these fit together: `issues.py` enumerates every feasible package.
`system.py` runs the physical model for each one. `payoffs.py` scores
every package for both parties and `zopa()` filters to the packages that
clear both BATNAs. `frontier.py: pareto_frontier()` finds the efficient
boundary of that set. When the parties settle on something,
`efficiency_loss()` and `post_settlement_search()` show how far from the
frontier they landed and what they missed. The Nash and Kalai-Smorodinsky
points are reference markers shown at debrief, never targets pushed on
the parties during the negotiation.

## 5. Why the physical model matters

A negotiation-analytic layer alone can score any hypothetical issue
combination. "Medium storage capacity" and "moderate release guarantee"
mean something only once you know what they do to the actual river. `hydrology.py`, `reservoir.py`, and `system.py` simulate the
monthly water balance for a given package and hand the negotiation layer
real numbers: how much water reaches Country B in the dry season, how
much firm hydropower Country A gets. This is what makes
`post_settlement_search()` a real finding rather than a toy exercise: the
packages it surfaces are physically real alternatives.

## 6. Backend structure

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
                 to test whether an agreement holds up under stress
```

`issues.py` sits alongside `system.py`: it defines what a package is and
enumerates the full feasible space (2,000-20,000 packages for the default
scenario) that `payoffs.py` and `frontier.py` operate over.

### Methods behind each physical or economic relationship

- **Reservoir operation** (`reservoir.py`): a fill-season/release-season
  rule with a receding-horizon drawdown, emptying storage by the end of
  every release season. This is a stylised version of standard
  single-purpose reservoir operating rules, not a calibrated real
  operating manual. See LIMITATIONS.md for what this simplifies away.
- **Channel routing** (`system.py: route_muskingum`): the Muskingum method
  (McCarthy 1938), a standard hydraulic technique for translating an
  upstream release hydrograph into a downstream arrival hydrograph given a
  reach's travel time (`K`) and shape parameter (`x`).
- **Daily flow disaggregation** (`hydrology.py: disaggregate_to_daily`):
  PCHIP interpolation of the seasonal curve plus a lognormal AR(1) daily
  multiplier, a standard approach in stochastic hydrology for generating a
  plausible daily trace from monthly statistics (see Salas 1993 for the
  general family of AR-based disaggregation methods). Not fit to any
  observed daily gauge record.
- **Crop water-yield response** (`economics.py`): based on the FAO
  crop-water production function (Doorenbos & Kassam 1979, FAO Irrigation
  and Drainage Paper No. 33), where relative yield loss is proportional to
  relative water deficit. Simplified here to a linear response (the FAO
  method's yield-response factor `Ky` varies by crop and growth stage;
  this codebase fixes it at 1 implicitly). See LIMITATIONS.md.

## 7. The allocation-mechanism issue

A specific dynamic drove a mid-build redesign of this issue: if Country
A's own water demand grows and it stores or diverts more to meet it, that
has real consequences for Country B. Whether B is protected from those
consequences depends on how the two countries have agreed to share the
river. `allocation_mechanism` (an `issues.py` issue, computed in
`system.py: _make_floor_fn`) represents that agreement as a rule for
computing A's release, not a single negotiated number:

| Mechanism | Rule | Real-world echo |
|---|---|---|
| `fixed_volume` | Flat MCM/month regardless of actual flow | The simplest treaty form. Only as protective as its number relative to actual flow: LIMITATIONS.md documents an empirical finding that a fixed floor can be irrelevant in a wet year and severely binding in a dry one |
| `percentage_of_flow` | A releases a fixed share of that month's actual inflow | Scales with hydrology automatically, but B's shortfall risk in a drought scales with it too |
| `zonal_formula` | Guaranteed passthrough below a threshold, B gets a share of anything above it | Loosely echoes the two-zone logic of the 1960 Indus Waters Treaty's division of the Indus system. This is not a reproduction of the actual IWT allocation; see LIMITATIONS.md |
| `adaptive_rule_curve` | A's obligation depends on A's own current storage relative to a target curve | A real reservoir-operations pattern, giving A discretion but not unlimited discretion |

`agricultural_outcomes()` in `economics.py` and A's own consumptive demand
(`basin.demand.a_consumptive_shape` in the scenario config) exist for the
same reason: A's growing demand is subtracted from its reservoir release
before the water reaches the border. A weak `allocation_mechanism` lets
A's domestic growth erode what B receives. The mechanism the parties
negotiate determines whether it does.

## 8. Why crop-mix isn't a Package field

A continuous variable, such as what fraction of Country B's land is
planted with a water-intensive crop, doesn't fit `issues.py`'s
fully-enumerated discrete package space without exploding it
combinatorially. Adding one such dimension per party would multiply the
roughly 2,880-package space by 25-100x, past the 2,000-20,000 target that
keeps `pareto_frontier()`'s brute-force dominance check fast. Real
treaties also don't negotiate this: countries negotiate volumes, timing,
and allocation mechanisms, not each other's farmers' cropping choices.

So crop-mix is each party's own continuous domestic response to whatever a
candidate package delivers. `economics.py` computes it from a fixed
default in the scenario config for all frontier and ZOPA analysis, and the
facilitator app exposes it as a slider: a "what if we planted differently"
sensitivity display layered on top of the negotiated frontier, not part of
it.

## 9. Multi-basin support

A scenario file, not the code, defines a basin. Every module from
`issues.py` down reads a `ScenarioConfig` built from whichever YAML file is
loaded; nothing in the code itself is specific to any one basin. The
default scenario, `generic_basin_v1.yaml`, is deliberately unnamed for this
reason.

`indus_basin_v1.yaml` is the first example of a basin-specific scenario:
the same underlying model, renamed and paired with a `factsheet` field
(real Indus Waters Treaty history, structure, and challenges) that the
app's Brief tab renders. Its hydrology and economics numbers are still the
same illustrative model as the generic scenario, not sourced from real
Indus data.

### Historical case studies

`indus_basin_v1.yaml` also defines `case_studies`: real, cited disputes
(the 2005-2007 Baglihar Dam difference, resolved by a Neutral Expert; the
2010-2013 Kishenganga arbitration, resolved by a Court of Arbitration) that
the Negotiate tab can load as opening positions for both parties. Each
entry's `summary` and `historical_outcome` name their sources (World Bank,
the Permanent Court of Arbitration's case record, the American Society of
International Law) rather than asserting facts unsourced.

The pedagogical point: both real disputes were resolved by third-party
*adjudication* under the treaty, not by the two countries reaching a
bilateral negotiated agreement. Loading a case study asks the same
question this tool asks of every exercise -- can the parties find the
ZOPA and settle -- against positions that, historically, adjudication
had to resolve instead. The mapping onto this tool's 6 issues is a
simplification: the real disputes turned on specific engineering
parameters (freeboard, spillway gate elevation, pondage volume, minimum
environmental flow) that aren't individually modeled here. See
LIMITATIONS.md.

The facilitator app's sidebar lists every validated YAML file in
`src/riparia/config/`, so adding a Nile-style or Central-Asian scenario
means authoring a new scenario file (issues, value functions, hydrology,
agriculture, and optionally a factsheet) and placing it in that directory.
It appears in the dropdown automatically. What it does not do
automatically is calibrate that file's numbers to a real basin's actual
hydrology and economics; that needs real data (GRDC gauge records, FAO
AQUASTAT) and is separate follow-on work, not a code change. See
LIMITATIONS.md.

## References

- Doorenbos, J., & Kassam, A. H. (1979). *Yield Response to Water*. FAO
  Irrigation and Drainage Paper No. 33. Rome: Food and Agriculture
  Organization of the United Nations.
- Fisher, R., & Ury, W. (1981). *Getting to Yes: Negotiating Agreement
  Without Giving In*. Boston: Houghton Mifflin.
- Hammond, J. S., Keeney, R. L., & Raiffa, H. (1998). Even Swaps: A
  Rational Method for Making Trade-offs. *Harvard Business Review*, 76(2),
  137-149.
- Kalai, E., & Smorodinsky, M. (1975). Other Solutions to Nash's Bargaining
  Problem. *Econometrica*, 43(3), 513-518.
- Keeney, R. L., & Raiffa, H. (1976). *Decisions with Multiple Objectives:
  Preferences and Value Tradeoffs*. New York: Wiley.
- Lax, D. A., & Sebenius, J. K. (1986). *The Manager as Negotiator:
  Bargaining for Cooperation and Competitive Gain*. New York: Free Press.
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
- Walton, R. E., & McKersie, R. B. (1965). *A Behavioral Theory of Labor
  Negotiations: An Analysis of a Social Interaction System*. New York:
  McGraw-Hill.
- The Indus Waters Treaty (1960), between the Government of India and the
  Government of Pakistan, brokered by the International Bank for
  Reconstruction and Development (World Bank). Referenced only for the
  general two-zone structure that loosely inspired the `zonal_formula`
  allocation mechanism; see LIMITATIONS.md for what is and isn't
  reproduced.
