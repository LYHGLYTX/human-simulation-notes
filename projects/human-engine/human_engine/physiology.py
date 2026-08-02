"""Physiology (spec §2.8): energy, sleep debt, fatigue effects.

Embodied cognition (neuroscience/theories.md §6): a tired, sleep-deprived
person regulates emotion and impulses worse. Sleep is the largest recovery
source (spec §3.3.2) — sleep repays debt, restores energy and resources.

The engine calls `update()` every tick (sleeping flag switches between
wake consumption and sleep settlement). `sleep_session()` settles a whole
night for the UI/CLI.
"""
from __future__ import annotations

from .persona import Persona
from .state import State
from ._clamp import clamp


def fatigue(state: State, persona: Persona) -> float:
    """0-1 fatigue from accumulated sleep debt (1.0 = a full night owed)."""
    return clamp(state.sleep_debt / persona.sleep_need, 0, 1)


def update(state: State, persona: Persona, dt: float, sleeping: bool = False):
    """Per-tick physiology. Returns the current fatigue 0-1."""
    # circadian phase 0-24h; night hours cost more energy (embodied rhythm)
    phase = (state.t / 3600.0) % 24.0
    state.phase = phase
    night = phase >= 22.0 or phase < 6.0
    if sleeping:
        # sleep: the largest recovery source (spec §3.3.2)
        # repay debt slightly faster than real time (efficiency ~90%)
        hours = dt / 3600.0
        state.sleep_debt = clamp(state.sleep_debt - hours * 0.9, 0, 24)
        # ~2.3 h to refill energy from empty; resources/self-control recover
        # strongly; stress settles (sleep as stress recovery)
        state.energy = clamp(state.energy + dt * 0.013, 0, persona.energy_max)
        state.resources = clamp(state.resources + dt * 0.018, 0, 100)
        state.self_control = clamp(state.self_control + dt * 0.028, 0, 100)
        state.stress = clamp(state.stress - dt * 0.007, 0, 100)
    else:
        # wake: burn energy (faster at night), accumulate debt proportional
        # to sleep need
        rate = 0.0022 * (1.5 if night else 0.85)
        state.energy = clamp(state.energy - dt * rate, 0, persona.energy_max)
        state.sleep_debt = clamp(
            state.sleep_debt + dt * (persona.sleep_need / 86400.0), 0, 24)
    return fatigue(state, persona)


def sleep_session(state: State, persona: Persona, hours: float) -> dict:
    """Settle a whole sleep session (UI/CLI convenience)."""
    dt = hours * 3600.0
    for _ in range(int(hours * 12)):   # 5-minute steps (nonlinear dynamics)
        update(state, persona, 300.0, sleeping=True)
    # clamp leftovers from stepping quantization
    state.sleep_debt = max(0.0, state.sleep_debt)
    return {
        "hours": hours,
        "energy": round(state.energy, 1),
        "resources": round(state.resources, 1),
        "self_control": round(state.self_control, 1),
        "sleep_debt": round(state.sleep_debt, 2),
        "stress": round(state.stress, 1),
    }
