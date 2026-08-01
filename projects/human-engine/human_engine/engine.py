"""Engine: tick loop + event pipeline + decision (spec §3)."""

from __future__ import annotations

import random

from .persona import Persona, default_persona
from .state import State
from .memory import Memory
from . import emotion, stress, morality
from .llm import LLMClient, MockLLM
from ._clamp import clamp


TRAUMA_TYPES = {"humiliation", "betrayal", "abandonment", "threat", "loss"}


class Engine:
    def __init__(self, persona: Persona | None = None,
                 llm: LLMClient | None = None,
                 seed: int | None = None):
        self.persona = persona or default_persona()
        self.llm = llm or MockLLM()
        self.rng = random.Random(seed)
        self.state = State(t=0.0)
        # init mood from personality attractor
        self.state.mood_pad = self.persona.mood_attractor()
        # threshold from resilience
        self.state.threshold = 55.0 + self.persona.resilience * 30.0
        # seed life history into episodic memory (with trauma channel for severe ones)
        self.memory = Memory()
        self._seed_history()
        self.event_queue: list[str] = []

    def _seed_history(self):
        for h in self.persona.history:
            item = self.memory.add_episodic(
                t=h["t"], text=h["text"], valence=h["valence"],
                arousal=h["arousal"], importance=h["importance"])
            if abs(h["valence"]) > 0.7 and h["arousal"] > 0.5:
                self.memory.add_trauma_fragment(
                    t=h["t"], text=h["text"], arousal=h["arousal"],
                    trigger_words=[kw for kw in ["抛弃", "离开", "羞辱", "嘲笑", "孤立", "背叛"]
                                   if kw in h["text"]] or ["抛弃"])

    # ------------------------------------------------------------------
    def tick(self, dt: float = 1.0):
        self.state.relax(self.persona, dt)
        # process queued events
        while self.event_queue:
            raw = self.event_queue.pop(0)
            self.handle_event(raw)
        return self.state

    def inject(self, raw: str):
        self.event_queue.append(raw)

    # ------------------------------------------------------------------
    def handle_event(self, raw: str) -> dict:
        """The 8-step pipeline (spec §3.2)."""
        s = self.state

        # 1. perceive
        event = self.llm.perceive(raw, self.persona)
        # 2. appraise
        appr = self.llm.appraise(event, self.persona, s)
        a = appr.as_dict()
        a["text"] = raw

        # flashback check (trauma channel, bypasses normal retrieval)
        frag = self.memory.check_flashback(raw, s)
        flashback = False
        if frag is not None:
            flashback = True
            s.vigilance = clamp(s.vigilance + 0.15, 0, 1)
            # PAD dragged toward trauma state
            p, a0, d = s.pad
            s.pad = (clamp(p - 0.4, -1, 1), clamp(a0 + 0.35, -1, 1), clamp(d - 0.3, -1, 1))

        # 3. emotion shock
        strength = emotion.apply_appraisal(a, self.persona, s)

        # 4. stress accumulation + crash check
        stress.apply_shock(a, self.persona, s, self.rng)

        # 5. memory write
        importance = strength * 0.7 + a["goal_relevance"] * 0.3
        self.memory.add_episodic(t=s.t, text=raw, valence=a["valence"],
                                 arousal=abs(a["valence"]) * 0.6 + a["novelty"] * 0.4,
                                 importance=importance)
        if (event.type in TRAUMA_TYPES and abs(a["valence"]) > 0.6
                and strength > 0.45):
            # trigger words = the event type's keywords (generalized cues)
            self.memory.add_trauma_fragment(
                t=s.t, text=raw, arousal=abs(a["valence"]),
                trigger_words=EVENT_KEYWORDS(event.type)[:3])

        # 6. decision
        decision = self.decide(a, flashback)

        # 7. generate action
        snapshot = s.snapshot()
        context = [m.as_dict() for m in
                   self.memory.retrieve(raw, s, k=3, rng=self.rng)]
        if flashback and frag:
            context.insert(0, {"t": frag.t, "text": f"[闪回] {frag.fragment}",
                               "valence": -0.9, "arousal": frag.arousal,
                               "importance": 1.0})
        action = self.llm.generate(snapshot, context, decision)

        # post-behavior morality loop (if deviant behavior)
        if decision.get("deviant"):
            act = {"success": not s.crashed,
                   "punished": self.rng.random() < 0.3}
            morality.after_deviance(self.persona, s, act)
            action.details["aftermath"] = act

        # 8. record pipeline
        s.last_pipeline = {
            "raw": raw, "event_type": event.type, "appraisal": a,
            "strength": strength, "flashback": flashback,
            "decision": decision, "action": action.text,
            "memory_after": self.memory.summarize(),
        }
        return s.last_pipeline

    # ------------------------------------------------------------------
    def decide(self, a: dict, flashback: bool) -> dict:
        """Behavior selection: emotion tendency x impulse x norms (spec §3.4)."""
        s = self.state
        label = s.emotion_label

        # crash behavior spectrum
        if s.crashed:
            if s.crash_type == "fight":
                return {"behavior": "fight", "deviant": True}
            if s.crash_type == "dissociate":
                return {"behavior": "dissociate", "deviant": False}
            return {"behavior": "freeze", "deviant": False}
        if flashback:
            return {"behavior": "freeze", "deviant": False}

        # Frijda action tendencies per emotion
        tendency = {
            "anger": {"confront": 2.0, "vent": 1.2, "impulsive_attack": 0.8},
            "fear": {"avoid": 2.0, "freeze": 0.6},
            "anxious": {"avoid": 1.2, "seek_support": 0.8},
            "sadness": {"vent": 1.6, "seek_support": 1.4, "avoid": 0.5},
            "shame": {"avoid": 2.0, "seek_support": 0.3},
            "distress": {"seek_support": 1.0, "vent": 1.0},
            "joy": {"confront": 0.8, "neutral_act": 1.0},
            "calm": {"neutral_act": 1.6, "confront": 0.5},
            "neutral": {"neutral_act": 1.0},
        }.get(label, {"neutral_act": 1.0})

        # impulse & norms
        impulse = morality.compute_impulse(a, self.persona, s)
        s.impulse = impulse
        deviant_candidate = impulse > 0.6
        norm = morality.norm_check("impulsive_attack", self.persona, s)
        if deviant_candidate and norm["effective"] > 0.5:
            deviant_candidate = False   # norms still hold
        mech = morality.disengagement_pressure(self.persona, s, impulse, self.rng)

        scores = dict(tendency)
        if deviant_candidate:
            scores["impulsive_attack"] = scores.get("impulsive_attack", 0) + 2.5 + impulse
        scores.setdefault("neutral_act", 0.4)

        behavior = max(scores, key=scores.get)
        return {"behavior": behavior,
                "deviant": behavior in ("impulsive_attack", "impulsive_selfharm", "fight"),
                "impulse": impulse, "norm_effective": norm["effective"],
                "disengagement": mech}


def EVENT_KEYWORDS(etype: str) -> list[str]:
    from .appraisal import EVENT_TYPES
    return EVENT_TYPES.get(etype, {}).get("kw", [])
