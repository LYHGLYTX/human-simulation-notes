"""Psychometrics (spec §6.2): simplified clinical scales derived from the
engine's continuous state. Research tooling only — NOT for real diagnosis.

- PCL-5 (PTSD checklist, 0-80): re-experiencing / avoidance / hyperarousal /
  negative cognition, estimated from state + trauma channel.
- PHQ-9 (depression, 0-27): anhedonia, low energy, hopelessness, rumination.
- PANAS (positive/negative affect, 10-50 each): from PAD.
- STAI-S (state anxiety, 20-80): vigilance + arousal + stress.
"""

from __future__ import annotations

from .engine import Engine


def pcl5(engine: Engine) -> dict:
    s = engine.state
    mem = engine.memory
    frags = [f for f in mem.trauma if not f.consolidated]
    reexperiencing = min(5.0, sum(f.flashbacks for f in frags) * 1.5
                         + (1.5 if any(f.flashbacks for f in frags) else 0))
    avoidance = min(5.0, s.vigilance * 4 + len(frags) * 0.8)
    hyperarousal = min(5.0, s.vigilance * 2.5 + s.stress / 25)
    negative_cog = min(5.0, s.depression_tendency * 3 + s.shame * 2
                       + (0.5 if s.crash_count else 0))
    score = (reexperiencing + avoidance + hyperarousal + negative_cog) * 4
    return {"score": round(min(80.0, score), 1),
            "reexperiencing": round(reexperiencing, 1),
            "avoidance": round(avoidance, 1),
            "hyperarousal": round(hyperarousal, 1),
            "negative_cognition": round(negative_cog, 1)}


def phq9(engine: Engine) -> dict:
    s = engine.state
    anhedonia = min(3.0, (1 - s.pad[0]) * 1.5 + s.depression_tendency)
    low_energy = min(3.0, (1 - s.energy / 100) * 2 + s.wear / 10)
    hopeless = min(3.0, s.crash_count * 0.8 + s.depression_tendency * 1.2)
    rumination = min(3.0, s.depression_tendency * 2.5)
    sleep = min(3.0, s.sleep_debt / 3)
    appetite = min(3.0, abs(s.pad[0]) * 1.2 if s.pad[0] < 0 else 0)
    concentration = min(3.0, (1 - s.self_control / 100) * 1.5 + s.vigilance)
    psychomotor = min(3.0, s.gas_phase in ("exhausted", "crashed") and 2 or 0)
    self_harm = min(3.0, s.depression_tendency * 1.5)
    items = [anhedonia, low_energy, hopeless, rumination, sleep, appetite,
             concentration, psychomotor, self_harm]
    return {"score": round(sum(items) * 3, 1), "items": [round(x, 1) for x in items]}


def panas(engine: Engine) -> dict:
    s = engine.state
    pos = max(5.0, min(50.0, (s.pad[0] + 1) * 12 + s.pad[1] * 8 + 15))
    neg = max(5.0, min(50.0, (1 - s.pad[0]) * 10 + s.pad[1] * 6
                       + s.depression_tendency * 15 + 8))
    return {"positive": round(pos, 1), "negative": round(neg, 1)}


def stai_s(engine: Engine) -> dict:
    s = engine.state
    score = 20 + s.vigilance * 25 + s.stress * 0.3 + s.pad[1] * 12
    return {"score": round(min(80.0, max(20.0, score)), 1)}


def full_report(engine: Engine) -> dict:
    return {
        "pcl5": pcl5(engine),
        "phq9": phq9(engine),
        "panas": panas(engine),
        "stai_s": stai_s(engine),
        "state": {
            "stress": round(engine.state.stress, 1),
            "resources": round(engine.state.resources, 1),
            "depression_tendency": round(engine.state.depression_tendency, 2),
            "vigilance": round(engine.state.vigilance, 2),
            "crash_count": engine.state.crash_count,
            "threshold": round(engine.state.threshold, 1),
            "trauma_unprocessed": sum(
                1 for f in engine.memory.trauma if not f.consolidated),
        },
    }
