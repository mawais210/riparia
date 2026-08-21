# riparia

A two-party transboundary water negotiation exercise: an upstream riparian
("Country A") and a downstream riparian ("Country B") negotiate a package of
terms governing a shared river across several rounds. A monthly water-balance
model computes the physical consequences of any agreed package; a
negotiation-analytic layer (BATNA, ZOPA, Pareto frontier, post-settlement
settlement, Nash/Kalai-Smorodinsky reference solutions) scores those
consequences against each party's value function.

This is a research/teaching instrument for negotiation methodology, not a
commercial game and not a calibrated model of any real basin — see
[LIMITATIONS.md](LIMITATIONS.md). The negotiation-analytic framework and its
citations are explained in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Install

Requires Python 3.11+.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Run the facilitator app

```bash
.venv/bin/streamlit run app/facilitator.py
```

Opens at `http://localhost:8501`. Four tabs: **Brief** (scenario overview),
**Simulate** (joint fact-finding — both parties see the same physical
simulation, with live sliders for climate stress and each party's own
crop-mix), **Negotiate** (offers, single negotiating text, settlement), and
**Debrief** (locked until settlement, then reveals both value functions, the
Pareto frontier with the agreement plotted on it, post-settlement settlement
packages the parties missed, and a dry-year stress test).

## Run the scripted example

```bash
.venv/bin/python examples/run_full_exercise.py
```

Runs a scripted four-round exercise end to end and prints the agreed
package, both parties' scores, position relative to the frontier,
efficiency loss, the top post-settlement-settlement packages missed, and the
fixed-vs-contingent stress comparison.

## Run the tests

```bash
.venv/bin/pytest
```

## Package layout

```
src/riparia/
    hydrology.py      flow generation, PCHIP+AR(1) disaggregation, ensembles
    reservoir.py       reservoir mass balance under a fill/release-season rule
    climate.py          structural climate trends + multi-year drought sequences
    economics.py        each party's own crop-mix response (income/labor)
    system.py            couples hydrology + both reservoirs + routing + demand
    issues.py             issue/package definitions, feasible-space enumeration
    payoffs.py            value functions, BATNA, ZOPA
    frontier.py            Pareto frontier, PSS, Nash/Kalai-Smorodinsky
    contingent.py            state-contingent allocation rules vs. fixed packages
    engine.py                 round state machine, single negotiating text
    logging_.py                append-only event log
    config_schema.py            pydantic schema for scenario YAML files
    config/indus_style_v1.yaml   default scenario
app/facilitator.py    Streamlit facilitator application
examples/run_full_exercise.py   scripted end-to-end example
tests/                  pytest suite
```

## Default scenario

`indus_style_v1.yaml` is a stylised, Indus-inspired two-party basin: 6
negotiated issues (upstream storage capacity, filling window, allocation
mechanism, data exchange, flood early warning, financing transfer) giving
~2,900 feasible packages, a non-empty ZOPA, and a Pareto frontier with
dozens of distinct non-dominated points — see `docs/METHODOLOGY.md` for what
each issue represents and why the space is kept to this size.
