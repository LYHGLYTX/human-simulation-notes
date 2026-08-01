"""LLM client protocol + MockLLM + OpenAICompatLLM (spec §4).

Engine calls these; OpenAICompatLLM activates when HE_API_KEY is set.
Env config:
  HE_API_KEY    API key (required for real LLM)
  HE_BASE_URL   OpenAI-compatible base url (default https://api.openai.com/v1)
  HE_MODEL      model name (default gpt-4o-mini)
  HE_PROXY      http proxy for the API call, e.g. http://127.0.0.1:2080
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
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


class OpenAICompatLLM(LLMClient):
    """Real LLM via any OpenAI-compatible endpoint (OpenAI / DeepSeek /
    SiliconFlow / local vLLM...). Activates when HE_API_KEY is set."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, proxy: str | None = None,
                 temperature: float = 0.4):
        self.api_key = api_key or os.environ.get("HE_API_KEY", "")
        self.base_url = (base_url or os.environ.get("HE_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("HE_MODEL") or "gpt-4o-mini"
        self.proxy = proxy if proxy is not None else os.environ.get("HE_PROXY", "")
        self.temperature = temperature

    # ------------------------------------------------------------------
    def _chat(self, system: str, user: str) -> str:
        """One chat completion call; returns assistant text. Raises on error."""
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        opener = urllib.request.build_opener()
        if self.proxy:
            opener.add_handler(urllib.request.ProxyHandler(
                {"http": self.proxy, "https": self.proxy}))
        with opener.open(req, timeout=90) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Pull the first {...} block out of a response, tolerate prose."""
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError(f"no JSON in LLM reply: {text[:200]}")
        return json.loads(m.group(0))

    # ------------------------------------------------------------------
    def perceive(self, raw: str, persona: Persona) -> _appraisal.Event:
        types = ", ".join(_appraisal.EVENT_TYPES.keys())
        sysp = ("你是人类行为模拟器的感知模块。把事件文本转成结构化 JSON。"
                f"事件类型必须是以下之一: {types}")
        out = self._extract_json(self._chat(sysp,
            f'事件: {raw}\n输出 JSON: {{"type": "类型", "intensity": 0到1, '
            f'"subject": "施加者(可空)", "object": "承受者(可空)"}}'))
        etype = out.get("type", "neutral")
        if etype not in _appraisal.EVENT_TYPES:
            etype = "neutral"
        return _appraisal.Event(
            type=etype,
            intensity=max(0.0, min(1.0, float(out.get("intensity", 1.0)))),
            subject=str(out.get("subject", "")),
            object=str(out.get("object", "")),
            text=raw)

    def appraise(self, event, persona: Persona, state: State) -> _appraisal.Appraisal:
        sysp = ("你是情绪评价模块（Scherer appraisal theory）。根据事件与状态，"
                "输出六维评价的 JSON（全部 0-1，valence 0=极负面 1=极正面）。")
        user = (f"人格: {persona.summary_text()}\n"
                f"当前: 应激{state.stress:.0f}/100 资源{state.resources:.0f}/100 "
                f"情绪{state.emotion_label} 崩溃={'是' if state.crashed else '否'}\n"
                f"事件: {event.text}\n"
                '输出 JSON: {"novelty":0-1,"valence":0-1,"goal_relevance":0-1,'
                '"coping_potential":0-1,"norm_violation":0-1,"control":0-1}')
        try:
            d = self._extract_json(self._chat(sysp, user))
        except Exception:
            # fall back to rule-based appraisal so the sim never hard-crashes
            return _appraisal.appraise(event, persona, state)
        return _appraisal.Appraisal(
            novelty=float(d.get("novelty", 0.3)),
            valence=float(d.get("valence", 0.0)),
            goal_relevance=float(d.get("goal_relevance", 0.3)),
            coping_potential=float(d.get("coping_potential", 0.5)),
            norm_violation=float(d.get("norm_violation", 0.0)),
            control=float(d.get("control", 0.5)),
            event_type=event.type, text=event.text)

    def generate(self, snapshot: dict, context: list, decision: dict) -> Action:
        mem = "\n".join(f"- {m.get('text', '')}" for m in context[:4]) or "（无）"
        sysp = ("你是被模拟者本人。基于你的人格、当前状态与记忆，用第一人称"
                "生成面对当前情境的反应。只输出 JSON，反应必须与状态一致："
                "崩溃时无法冷静，绝望时不会乐观。")
        user = (f"人格:\n{snapshot.get('persona_summary', '') or ''}\n"
                f"当前状态: 情绪={snapshot['emotion_label']} "
                f"PAD={[round(v,2) for v in snapshot['pad']]} "
                f"应激={snapshot['stress']:.0f} 资源={snapshot['resources']:.0f} "
                f"自控={snapshot['self_control']:.0f} "
                f"内疚={snapshot['guilt']:.2f} 羞耻={snapshot['shame']:.2f} "
                f"崩溃={'是' if snapshot.get('crashed') else '否'}\n"
                f"相关记忆:\n{mem}\n"
                f"行为决策: {decision.get('behavior', 'neutral_act')}\n"
                '输出 JSON: {"text": "反应（1-3句，中文）", "tone": "语气词"}')
        try:
            d = self._extract_json(self._chat(sysp, user))
            return Action(kind=decision.get("behavior", "neutral_act"),
                          text=str(d.get("text", "")), details={"tone": d.get("tone")})
        except Exception:
            # fallback: keep engine alive with a template line
            return Action(kind=decision.get("behavior", "neutral_act"),
                          text="（沉默）")


def make_llm() -> LLMClient:
    """Auto-select: real LLM if HE_API_KEY set, else MockLLM."""
    if os.environ.get("HE_API_KEY"):
        return OpenAICompatLLM()
    return MockLLM()
