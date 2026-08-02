"""Stress & crash dynamics (spec §2.3, §3.3.2): GAS phases, COR resources,
allostatic wear, sensitization. """

from __future__ import annotations

import random

from .persona import Persona
from .state import State
from ._clamp import clamp


def apply_shock(appr: dict, persona: Persona, state: State, rng: random.Random | None = None):
    """Step 4 of the pipeline: stress accumulates, resources deplete."""
    rng = rng or random
    impact = (abs(appr["valence"]) * 0.7 + appr["goal_relevance"] * 0.5
              + (1 - appr["coping_potential"]) * 0.3)
    impact *= (1.0 + persona.neuroticism * 0.5)
    impact *= (0.9 + rng.uniform(0, 0.2))

    # mundane channel: neutral-type, low-intensity interaction (chitchat,
    # small talk) is resource-MAINTAINING, not depleting (COR: social
    # interaction is a resource; mundane contact maintains bonds). Type-based
    # gate is robust to LLM valence drift (LLMs rate greetings mildly
    # positive ~0.5, which after mixing lands ~0.2 — inside the gate).
    if (appr["event_type"] == "neutral" and abs(appr["valence"]) < 0.3
            and appr["goal_relevance"] < 0.45):
        mild = impact * 0.15
        state.stress = clamp(state.stress + mild * 4, 0, 100)
        state.resources = clamp(state.resources + mild * 10, 0, 100)
        state.self_control = clamp(state.self_control - mild * 6, 0, 100)
        check_crash(persona, state)
        return

    state.stress = clamp(state.stress + impact * 12, 0, 100)
    state.resources = clamp(state.resources - impact * 8, 0, 100)
    state.self_control = clamp(state.self_control - impact * 10, 0, 100)

    # positive events replenish resources (COR: resource gain spiral)
    if appr["valence"] > 0.3:
        state.stress = clamp(state.stress - impact * 10, 0, 100)
        state.resources = clamp(state.resources + impact * 8, 0, 100)
        state.self_control = clamp(state.self_control + impact * 6, 0, 100)

    # vigilance rises with repeated threat
    if appr["valence"] < -0.3:
        state.vigilance = clamp(state.vigilance + 0.02 * impact, 0, 1)

    # schema reinforcement: hits make the schema stronger (self-maintaining)
    for schema, strength in persona.schema_hits(appr.get("event_type", ""),
                                                appr.get("text", "")):
        persona.schemas[schema] = clamp(persona.schemas.get(schema, 0.2) + 0.01, 0, 1)

    check_crash(persona, state)


def check_crash(persona: Persona, state: State) -> bool:
    """Crash triggers (spec §3.3.2): chronic (stress>threshold & low resources)
    or acute (single shock > 0.8*threshold)."""
    if state.crashed:
        return True

    acute = state.stress > state.threshold * 0.8 and state.resources < 25
    chronic = state.stress > state.threshold and state.resources < 40

    if acute or chronic:
        state.crashed = True
        state.crash_count += 1
        # crash type by personality x situation (Fanselow defense cascade)
        if state.pad[0] < -0.4 and state.pad[2] < -0.3 and persona.bis > persona.bas:
            state.crash_type = "freeze"       # 木僵/冻结
        elif persona.bas > persona.bis and state.pad[1] > 0.3:
            state.crash_type = "fight"        # 愤怒爆发
        elif persona.attachment in ("fearful", "preoccupied") and state.pad[2] < -0.3:
            state.crash_type = "dissociate"   # 解离
        else:
            state.crash_type = "freeze"

        # permanent consequences (irreversible, spec §3.3.2)
        state.threshold = clamp(state.threshold - 8.0 * (1 - persona.resilience * 0.5), 20, 100)
        state.wear += 5.0
        state.resources = clamp(state.resources - 15, 0, 100)
        return True
    return False


def recover_from_crash(persona: Persona, state: State):
    """After a crash, recovery is slow and incomplete."""
    state.crashed = False
    state.stress = clamp(state.stress * 0.5, 0, 100)
    # recovery ceiling lowered by wear (allostatic load)
    state.resources = clamp(state.resources * 0.6, 0, 100 - state.wear)
    # depression tendency rises after crash (Beck triad activation)
    state.depression_tendency = clamp(state.depression_tendency + 0.1, 0, 1)
