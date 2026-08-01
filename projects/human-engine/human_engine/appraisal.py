"""Appraisal: event structurization + Scherer checklist (spec §3.2 steps 1-2).

The real system uses an LLM for this (LLMClient.appraise). MockLLM here is a
keyword/rule-based implementation so the engine runs fully offline (M1).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .persona import Persona
from .state import State

# Event type table: type -> (valence, novelty, goal_relevance, keywords)
EVENT_TYPES: dict[str, dict] = {
    "praise":       {"valence": 0.65, "novelty": 0.3, "relevance": 0.4, "kw": ["夸", "赞", "表扬", "谢谢", "praise", "thank", "great job", "你真棒", "good", "道歉", "对不起", "抱歉", "sorry", "认错", "道歉信", "我爱你", "喜欢你", "在一起", "表白", "想你了", "抱抱"]},
    "success":      {"valence": 0.7,  "novelty": 0.3, "relevance": 0.6, "kw": ["成功", "赢了", "通过", "升职", "success", "won", "promotion", "考过", "录取"]},
    "criticism":    {"valence": -0.4, "novelty": 0.3, "relevance": 0.5, "kw": ["批评", "指责", "挑剔", "criticism", "criticize", "你不行", "废物", "没用"]},
    "rejection":    {"valence": -0.5, "novelty": 0.4, "relevance": 0.6, "kw": ["拒绝", "不行", "驳回", "没通过", "reject", "refuse", "denied", "没选上", "滚"]},
    "conflict":     {"valence": -0.5, "novelty": 0.4, "relevance": 0.6, "kw": ["吵架", "冲突", "争执", "吵", "conflict", "argue", "fight", "骂", "草泥马", "傻逼", "混蛋", "妈的", "他妈的", "滚他妈的"]},
    "humiliation":  {"valence": -0.8, "novelty": 0.5, "relevance": 0.8, "kw": ["羞辱", "当众", "嘲笑", "丢人", "humiliate", "mock", "laugh at", "公开"]},
    "betrayal":     {"valence": -0.85, "novelty": 0.6, "relevance": 0.9, "kw": ["背叛", "骗", "出卖", "隐瞒", "betray", "cheat", "lie", "利用", "据为己有", "不还", "抢功", "在背后", "曝光", "泄露", "隐私"]},
    "abandonment":  {"valence": -0.9, "novelty": 0.5, "relevance": 0.95, "kw": ["分手", "离开我", "抛弃", "不要我", "abandon", "left me", "breakup", "离婚", "走了", "没人要", "没人记得", "累赘"]},
    "loss":         {"valence": -0.75, "novelty": 0.6, "relevance": 0.8, "kw": ["去世", "死了", "失去", "破产", "失业", "loss", "died", "death", "fired", "失去工作", "重病", "离世", "猝死"]},
    "threat":       {"valence": -0.7, "novelty": 0.7, "relevance": 0.85, "kw": ["威胁", "恐吓", "打", "暴力", "threat", "violence", "杀", "刀", "囚禁", "虐待", "跟踪", "绑架"]},
    "help":         {"valence": 0.55, "novelty": 0.3, "relevance": 0.5, "kw": ["帮助", "支持", "陪", "帮", "help", "support", "照顾"]},
    "neutral":      {"valence": 0.0,  "novelty": 0.2, "relevance": 0.2, "kw": []},
}


@dataclass
class Event:
    type: str
    text: str
    intensity: float = 1.0
    subject: str = ""
    object: str = ""


@dataclass
class Appraisal:
    novelty: float = 0.2
    valence: float = 0.0
    goal_relevance: float = 0.3
    coping_potential: float = 0.6
    norm_violation: float = 0.0
    control: float = 0.5
    event_type: str = "neutral"
    text: str = ""

    def as_dict(self):
        return {
            "novelty": self.novelty, "valence": self.valence,
            "goal_relevance": self.goal_relevance,
            "coping_potential": self.coping_potential,
            "norm_violation": self.norm_violation, "control": self.control,
            "event_type": self.event_type, "text": self.text,
        }


def perceive(raw: str) -> Event:
    """Rule-based structurization of a raw event text (MockLLM.perceive)."""
    best, best_score = "neutral", 0.0
    for etype, spec in EVENT_TYPES.items():
        hits = sum(1 for kw in spec["kw"] if kw in raw)
        if hits > best_score:
            best, best_score = etype, hits
    return Event(type=best, text=raw)


def appraise(event: Event, persona: Persona, state: State,
             rng: random.Random | None = None) -> Appraisal:
    """Rule-based Scherer checklist (MockLLM.appraise) with schema biases."""
    rng = rng or random
    spec = EVENT_TYPES.get(event.type, EVENT_TYPES["neutral"])

    valence = spec["valence"] * event.intensity
    novelty = min(1.0, spec["novelty"] * event.intensity + rng.uniform(0, 0.1))

    # goal relevance: event type + schema hits + attachment bias
    relevance = spec["relevance"]
    for _schema, strength in persona.schema_hits(event.type, event.text):
        relevance += strength * 0.5
    relevance *= persona.attachment_bias(event.type)
    relevance = min(1.0, relevance)

    # coping potential: high resources -> high; high stress -> low;
    # hypervigilance lowers it (threat overestimation, Ehlers & Clark)
    coping = clamp01(0.5 + (state.resources - 50) / 200 - state.stress / 300
                     - state.vigilance * 0.15)

    # control: internal locus raises perceived control; freeze lowers
    control = clamp01(persona.locus * 0.6 + (1 - state.crashed) * 0.2
                      - state.vigilance * 0.1)

    # norm violation: events touching moral foundations
    norm_violation = 0.0
    if event.type in ("humiliation", "betrayal", "threat"):
        norm_violation = (1 - persona.mft.get("care", 0.5)) * 0.3 + 0.2

    return Appraisal(novelty=novelty, valence=valence, goal_relevance=relevance,
                     coping_potential=coping, norm_violation=norm_violation,
                     control=control, event_type=event.type, text=event.text)


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))
