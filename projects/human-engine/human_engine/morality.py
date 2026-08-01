"""Morality & deviance (spec §2.7, §3.3.3): norms, impulse, moral disengagement,
guilt/shame. Based on GST (Agnew), moral disengagement (Bandura), self-discrepancy
(Higgins), ego depletion (Baumeister)."""

from __future__ import annotations

import random

from .persona import Persona
from .state import State
from ._clamp import clamp


def compute_impulse(appr: dict, persona: Persona, state: State) -> float:
    """impulse = BAS x frustration x (1 - self_control/100)  (GST path)"""
    frustration = max(0.0, -appr["valence"]) * appr["goal_relevance"]
    frustration += max(0.0, 0.5 - appr["coping_potential"]) * 0.5
    return clamp(persona.bas * (0.3 + frustration) * (1.1 - state.self_control / 100), 0, 1)


def norm_check(behavior: str, persona: Persona, state: State) -> dict:
    """Score how much a candidate behavior violates internalized norms."""
    # which moral foundations this behavior violates
    violations = {
        "impulsive_attack": {"care": 0.8, "fairness": 0.5, "authority": 0.5},
        "impulsive_steal": {"fairness": 0.9, "authority": 0.6},
        "impulsive_selfharm": {"care": 0.7},
        "lash_out": {"care": 0.6},
    }.get(behavior, {})

    strength = 0.0
    for foundation, weight in violations.items():
        strength += persona.mft.get(foundation, 0.5) * weight
    strength = min(1.0, strength)

    # moral disengagement lowers effective norm strength (Bandura 8 mechanisms)
    disengage = max(state.moral_disengagement.values())
    effective = strength * (1.0 - disengage * 0.8)
    return {"raw": strength, "effective": effective, "disengaged": disengage}


def disengagement_pressure(persona: Persona, state: State, impulse: float,
                           rng: random.Random | None = None) -> float:
    """Under high stress + impulse, disengagement mechanisms activate."""
    rng = rng or random
    pressure = (state.stress / 100) * 0.5 + impulse * 0.5
    if pressure > 0.55 and rng.random() < pressure:
        mech = rng.choice(list(state.moral_disengagement.keys()))
        state.moral_disengagement[mech] = clamp(state.moral_disengagement[mech] + 0.05, 0, 1)
        return mech
    return ""


def after_deviance(persona: Persona, state: State, act: dict):
    """Post-behavior loop: success unpunished -> disengagement rises (learned);
    else guilt/shame (Tangney: shame targets self, more destructive)."""
    if act.get("punished") or not act.get("success"):
        # guilt (behavior-focused) or shame (self-focused) by locus
        if persona.locus > 0.5:
            state.guilt = clamp(state.guilt + 0.4, 0, 1)
        else:
            state.shame = clamp(state.shame + 0.45, 0, 1)
        # shame deepens depression tendency & defectiveness schema
        if state.shame > 0.5:
            persona.schemas["defectiveness"] = clamp(
                persona.schemas.get("defectiveness", 0.2) + 0.05, 0, 1)
            state.depression_tendency = clamp(state.depression_tendency + 0.05, 0, 1)
    else:
        # success without punishment: disengagement is learned (Bandura)
        for mech in state.moral_disengagement:
            state.moral_disengagement[mech] = clamp(
                state.moral_disengagement[mech] + 0.03, 0, 1)
