"""Week 1/2 append-only in-memory event store stub."""
from __future__ import annotations

from neogen.events.molecule_events import MoleculeEventEnvelope

EventEnvelope = MoleculeEventEnvelope | object

_EVENTS: list[EventEnvelope] = []


def append_event(event: EventEnvelope) -> EventEnvelope:
    _EVENTS.append(event)
    return event


def list_events() -> list[EventEnvelope]:
    return list(_EVENTS)


def clear_events() -> None:
    _EVENTS.clear()
