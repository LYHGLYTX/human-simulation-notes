"""Emotion dynamics: ALMA 3-layer (personality -> mood -> emotion) in PAD space.

Spec §2.1/§2.2, §3.2 step 3. Appraisal -> PAD displacement -> label mapping.
"""
from __future__ import annotations

from .persona import Persona
from .state import State
from ._clamp import clamp


# --- appraisal -> PAD displacement -------------------------------------
def apply_appraisal(appr: dict, persona: Persona, state: State):
    """Convert Scherer-style appraisal values into a PAD shock (OCC strength)."""
    valence = appr["valence"]            # -1..1 (intrinsic pleasantness)
    novelty = appr["novelty"]            # 0..1
    relevance = appr["goal_relevance"]   # 0..1
    coping = appr["coping_potential"]    # 0..1
    control = appr["control"]            # 0..1
    norm_viol = appr["norm_violation"]   # 0..1 (event violates own norms)

    # intensity: OCC-style strength = relevance * |valence| + novelty contribution
    strength = relevance * abs(valence) * 0.7 + novelty * 0.3
    strength = min(1.0, strength)

    # PAD displacement (WASABI-style impulse)
    dp = valence * 0.55 * strength
    da = (novelty * 0.45 + (1.0 - coping) * 0.25) * strength
    dd = (control - 0.5) * 0.5 * strength - norm_viol * 0.25 * strength

    p, a, d = state.pad
    state.pad = (
        clamp(p + dp, -1, 1),
        clamp(a + da, -1, 1),
        clamp(d + dd, -1, 1),
    )
    state.emotion_strength = strength

    # if valence strongly negative and relevance high -> mood shifts down slowly
    if valence < -0.5 and relevance > 0.6:
        mp, ma, md = state.mood_pad
        state.mood_pad = (clamp(mp - 0.02 * strength, -1, 1),
                          clamp(ma + 0.01 * strength, -1, 1),
                          clamp(md - 0.01 * strength, -1, 1))

    state.emotion_label = label_from_pad(state.pad)
    return strength


def label_from_pad(pad: tuple[float, float, float]) -> str:
    """Map PAD coordinates to a discrete emotion label (Russell-style)."""
    p, a, d = pad
    if p > 0.2:
        return "joy" if a > 0.15 else "calm"
    if p < -0.15:
        if a > 0.35:
            return "anger" if d > -0.1 else "fear"
        if a < -0.1:
            return "sadness" if d > -0.2 else "shame"
        if d < -0.3:
            return "shame"
        return "distress"
    if a > 0.3:
        return "anxious"
    return "neutral"
