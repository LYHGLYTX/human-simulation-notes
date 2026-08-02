"""Coping strategy selection (EMA: Marsella & Gratch, spec §3.4).

EMA treats appraisal as evaluation of the belief-intention-plan structure,
then selects a *coping* tendency that drives behavior. Three families:
problem-focused (直面), emotion-focused (宣泄/求助), avoidance (回避).

Modulators: perceived control & coping potential (Lazarus & Folkman),
resource level (COR), current emotion (Frijda action tendencies).
"""
from __future__ import annotations

from .persona import Persona
from .state import State

PROBLEM = "problem_focused"
EMOTION = "emotion_focused"
AVOID = "avoidance"


def select_coping(appr: dict, persona: Persona, state: State) -> str:
    control = appr.get("control", 0.5)
    coping = appr.get("coping_potential", 0.5)
    resources = state.resources
    # learned helplessness: repeated crashes/depression abandon active coping
    helpless = min(1.0, state.crash_count * 0.12
                   + state.depression_tendency * 0.3)

    # resource exhaustion or helplessness forces avoidance (COR loss spiral,
    # Seligman: passive resignation)
    if resources < 30 or coping < 0.3 or helpless > 0.5:
        return AVOID
    # high control + high coping -> face the problem
    if control > 0.6 and coping > 0.5:
        return PROBLEM
    # sad/shameful states favor emotion-focused coping (support/vent)
    if state.emotion_label in ("sadness", "distress", "shame"):
        return EMOTION
    return PROBLEM if control > 0.4 else AVOID


def modulate(scores: dict, style: str) -> dict:
    """Adjust behavior tendency scores by coping style."""
    s = dict(scores)
    boosts = {
        PROBLEM: {"confront": 1.3, "neutral_act": 1.1, "avoid": 0.75},
        EMOTION: {"vent": 1.1, "seek_support": 1.1, "avoid": 0.85},
        AVOID:   {"avoid": 1.3, "freeze": 1.3, "confront": 0.6,
                  "vent": 0.8, "seek_support": 0.7},
    }
    for k, mult in boosts.get(style, {}).items():
        if k in s:
            s[k] *= mult
    return s
