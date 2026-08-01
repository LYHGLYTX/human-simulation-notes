"""LLM client protocol + MockLLM (spec §4). Engine calls these; swap in
OpenAICompatLLM later without touching engine code."""

from __future__ import annotations

from dataclasses import dataclass

from . import appraisal as _appraisal
from .persona import Persona
from .state import State


@dataclass
class Action:
    kind: str            # behavioral category
    text: str            # generated behavior / utterance
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class LLMClient:
    """Protocol: perceive / appraise / generate."""

    def perceive(self, raw: str, persona: Persona) -> _appraisal.Event: ...
    def appraise(self, event, persona: Persona, state: State) -> _appraisal.Appraisal: ...
    def generate(self, snapshot: dict, context: list, decision: dict) -> Action: ...


class MockLLM(LLMClient):
    """Rule-based offline implementation (M1)."""

    def perceive(self, raw: str, persona: Persona) -> _appraisal.Event:
        return _appraisal.perceive(raw)

    def appraise(self, event, persona: Persona, state: State) -> _appraisal.Appraisal:
        return _appraisal.appraise(event, persona, state)

    def generate(self, snapshot: dict, context: list, decision: dict) -> Action:
        kind = decision.get("behavior", "neutral_act")
        label = snapshot["emotion_label"]
        pad = snapshot["pad"]

        template = {
            "confront": "你盯着对方，声音发抖但一字一句地说：\"你再说一遍。\"",
            "avoid": "你低下头，装作没听见，快步走开。",
            "vent": "你把门猛地关上，把桌上的东西摔在地上。",
            "seek_support": "你拿起手机，犹豫很久，给唯一信任的人发了条消息。",
            "impulsive_attack": "你冲上去推了他一把，脑子里只有一个念头：让他闭嘴。",
            "impulsive_selfharm": "你把自己关进房间，手臂上传来一阵刺痛。",
            "freeze": "你站在原地，脑子一片空白，什么也说不出来。",
            "fight": "你爆发了，砸了手边的东西，吼出压抑很久的话。",
            "dissociate": "世界突然变得很远，你看着自己像在看别人。",
            "neutral_act": "你平静地处理了这件事。",
        }
        text = template.get(kind, template["neutral_act"])

        # mood coloring of inner monologue (PAD -> style)
        if pad[0] < -0.3:
            inner = "心里堵得慌" if pad[1] < 0 else "心里烧着一团火"
        elif pad[0] > 0.3:
            inner = "胸口松了一些"
        else:
            inner = "说不上什么感觉"
        return Action(kind=kind, text=f"{text}（{inner}）")
