"""Append-only event log for a negotiation exercise: every offer, revision,
criticism, simulation run, and settlement, timestamped and round-numbered.
This is the exercise's primary experimental dataset (per-round offers and
their `move_type` classification), not an afterthought -- treat it as such
when adding new event-producing actions in `engine.py`.
"""

# Import necessary libraries
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

# Custom Functions


@dataclass(frozen=True)
class Event:
    seq: int
    timestamp: str
    round_number: int
    phase: str
    event_type: str
    party: str | None
    payload: dict[str, Any]
    move_type: str | None = None


def classify_move(prev_scores: dict[str, float] | None, new_scores: dict[str, float], tolerance: float = 0.5) -> str:
    """Classify an offer relative to the previous one, by joint (summed)
    score: "initial" (no previous offer to compare against), "integrative"
    (grows the joint value -- the pie got bigger), "value_destroying" (joint
    value fell), or "distributive" (joint value ~unchanged, so any shift is
    a reallocation between parties rather than a gain for either)."""
    if prev_scores is None:
        return "initial"
    prev_joint = sum(prev_scores.values())
    new_joint = sum(new_scores.values())
    if new_joint > prev_joint + tolerance:
        return "integrative"
    if new_joint < prev_joint - tolerance:
        return "value_destroying"
    return "distributive"


@dataclass
class EventLog:
    _events: list[Event] = field(default_factory=list)

    def append(
        self,
        round_number: int,
        phase: str,
        event_type: str,
        party: str | None = None,
        payload: dict[str, Any] | None = None,
        move_type: str | None = None,
    ) -> Event:
        event = Event(
            seq=len(self._events) + 1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            round_number=round_number,
            phase=phase,
            event_type=event_type,
            party=party,
            payload=payload or {},
            move_type=move_type,
        )
        self._events.append(event)
        return event

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def to_json(self) -> str:
        return json.dumps([asdict(e) for e in self._events], indent=2, default=str)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._events:
            return pd.DataFrame(
                columns=["seq", "timestamp", "round_number", "phase", "event_type", "party", "move_type", "payload"]
            )
        return pd.DataFrame([asdict(e) for e in self._events])

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())
