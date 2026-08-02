"""Interview-based persona generation (1,000 People paradigm, arXiv:2411.10109).

The 1,000-People study showed that LLM agents built from ~2h structured
interviews reproduce their humans' questionnaire responses ~85% of the time.
This module implements that paradigm's offline half: a fixed interview script
whose answers map to OCEAN / schemas / attachment, plus a generated life
history. The same mapping layer can consume LLM-produced answers later.

Usage:
    from human_engine.persona_generation import interview, persona_from_answers
    p = interview()                      # deterministic offline interview
    p2 = persona_from_answers({"q1": 0.9, ...}, seed=1)   # from any answers
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .persona import Persona

# ---------------------------------------------------------------------------
# Interview script: (id, question, target dimension, direction)
# direction: +1 = higher answer -> higher trait, -1 = inverse
INTERVIEW: list[tuple[str, str, str, int]] = [
    # extraversion
    ("q1", "遇到陌生人时，你会主动开始聊天吗？", "extraversion", +1),
    ("q2", "周末你更喜欢一个人待着，对吗？", "extraversion", -1),
    # neuroticism
    ("q3", "小事不顺时你会反复想很久吗？", "neuroticism", +1),
    ("q4", "你觉得自己心态很稳、不容易慌，对吗？", "neuroticism", -1),
    # openness
    ("q5", "你对新事物、新想法总是充满好奇吗？", "openness", +1),
    ("q6", "你更喜欢按老习惯做事，对吗？", "openness", -1),
    # agreeableness
    ("q7", "看到别人难过时你会忍不住想帮忙，对吗？", "agreeableness", +1),
    ("q8", "你常常觉得别人太蠢、不想理会，对吗？", "agreeableness", -1),
    # conscientiousness
    ("q9", "你做事习惯提前计划、按计划执行吗？", "conscientiousness", +1),
    ("q10", "你经常拖延到最后一刻，对吗？", "conscientiousness", -1),
    # attachment & schemas
    ("q11", "你常担心重要的人会离开你吗？", "abandonment", +1),
    ("q12", "你觉得大多数人是值得信任的，对吗？", "mistrust", -1),
    ("q13", "小时候，重要的人离开你时你几乎崩溃，对吗？", "attachment_trauma", +1),
    ("q14", "你觉得自己在很多事上无能为力，对吗？", "failure", +1),
    # life history
    ("q15", "你经历过一次重大的失去（亲人/恋人/工作）吗？", "loss", +1),
    ("q16", "你曾经被人当众羞辱或孤立过吗？", "humiliation", +1),
    ("q17", "有人在你最需要时坚定地支持过你吗？", "support", +1),
]

# life-history templates: (text, base valence, base arousal, driven by answer)
HISTORY_TEMPLATES: list[tuple[str, float, float, str]] = [
    ("重要的人离开了自己，之后很久都缓不过来", -0.85, 0.75, "loss"),
    ("在人群中被当众羞辱，留下了很深的阴影", -0.8, 0.7, "humiliation"),
    ("在最难的时候有人坚定地陪在身边", 0.65, 0.5, "support"),
]


def persona_from_answers(answers: dict[str, float], seed: int = 0) -> Persona:
    """Map interview answers (each 0-1: 0=strongly disagree, 1=strongly agree)
    to a Persona with a generated life history."""
    rng = random.Random(seed)

    # average answers per dimension (with direction)
    sums: dict[str, list[float]] = {}
    for qid, _q, dim, direction in INTERVIEW:
        if qid not in answers:
            continue
        v = max(0.0, min(1.0, answers[qid]))
        sums.setdefault(dim, []).append(v if direction > 0 else 1.0 - v)
    avg = {dim: sum(vs) / len(vs) for dim, vs in sums.items()}

    def g(name: str, default: float = 0.5) -> float:
        return round(max(0.05, min(0.95, avg.get(name, default))), 2)

    # attachment style from abandonment concern + trust + trauma
    abandon = g("abandonment")
    mistrust = g("mistrust")
    trauma = g("attachment_trauma")
    if abandon > 0.55 and trauma > 0.5:
        attachment = "fearful"
    elif abandon > 0.55:
        attachment = "preoccupied"
    elif mistrust > 0.6 and abandon < 0.4:
        attachment = "dismissing"
    else:
        attachment = "safe"

    # life history from the experience answers (times spread over the past)
    history = []
    t = -365.0 * rng.randint(2, 8)
    for text, base_val, base_ar, dim in HISTORY_TEMPLATES:
        ans = g(dim, 0.4)
        if ans < 0.25:          # didn't happen / barely
            continue
        intensity = 0.3 + 0.7 * ans
        history.append({
            "t": t, "text": text,
            "valence": base_val * intensity,
            "arousal": min(1.0, base_ar * (0.6 + ans)),
            "importance": min(1.0, 0.4 + ans * 0.55),
        })
        t += rng.randint(200, 900)

    return Persona(
        openness=g("openness"),
        conscientiousness=g("conscientiousness"),
        extraversion=g("extraversion"),
        agreeableness=g("agreeableness"),
        neuroticism=g("neuroticism"),
        attachment=attachment,
        schemas={
            "abandonment": round(0.2 + abandon * 0.6, 2),
            "mistrust": round(0.2 + mistrust * 0.6, 2),
            "defectiveness": round(0.2 + g("humiliation", 0.3) * 0.4, 2),
            "failure": round(0.2 + g("failure") * 0.5, 2),
            "subjugation": 0.2, "emotional_deprivation": 0.25,
        },
        history=history,
    )


# deterministic default interviewee (a moderately scarred, introverted person)
DEFAULT_ANSWERS: dict[str, float] = {
    "q1": 0.35, "q2": 0.7, "q3": 0.75, "q4": 0.3,
    "q5": 0.55, "q6": 0.45, "q7": 0.6, "q8": 0.35,
    "q9": 0.65, "q10": 0.4, "q11": 0.8, "q12": 0.25,
    "q13": 0.7, "q14": 0.55, "q15": 0.8, "q16": 0.65, "q17": 0.6,
}


def interview(answers: dict[str, float] | None = None,
              seed: int = 0) -> Persona:
    """Run the (offline) interview: fixed questions, supplied answers
    (defaults if None) -> generated Persona."""
    return persona_from_answers(answers or dict(DEFAULT_ANSWERS), seed=seed)


def summarize(persona: Persona) -> str:
    """Short natural-language summary of the interviewed person."""
    return (f"访谈生成个体：外倾{persona.extraversion:.2f} 神经质{persona.neuroticism:.2f} "
            f"依恋={persona.attachment} 图式={ {k: round(v, 2) for k, v in persona.schemas.items()} } "
            f"经历{len(persona.history)}段")
