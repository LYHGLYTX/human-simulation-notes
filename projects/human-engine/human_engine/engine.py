"""Engine: tick loop + event pipeline + decision (spec §3)."""

from __future__ import annotations

import random

from .persona import Persona, default_persona
from .state import State
from .memory import Memory
from . import emotion, stress, morality, physiology, coping
from .relations import Relations
from .llm import LLMClient, MockLLM, make_llm
from ._clamp import clamp


TRAUMA_TYPES = {"humiliation", "betrayal", "abandonment", "threat", "loss"}


class Engine:
    def __init__(self, persona: Persona | None = None,
                 llm: LLMClient | None = None,
                 seed: int | None = None):
        self.persona = persona or default_persona()
        self.rng = random.Random(seed)
        self.llm = llm or make_llm()
        if isinstance(self.llm, MockLLM) and self.llm.rng is None:
            self.llm.rng = self.rng  # share seeded rng for reproducibility
        self.state = State(t=0.0)
        # init mood from personality attractor
        self.state.mood_pad = self.persona.mood_attractor()
        # threshold from resilience
        self.state.threshold = 55.0 + self.persona.resilience * 30.0
        # seed life history into episodic memory (with trauma channel for severe ones)
        self.memory = Memory()
        self._seed_history()
        # MemoryBank: emotion decay shares the memory forgetting time base
        self.state.decay_lambda = self.memory.decay_lambda
        # PsychSim belief layer: seed from early-maladaptive schemas
        sch = self.persona.schemas
        b = self.state.beliefs
        b["abandonment"] = 0.2 + sch.get("abandonment", 0.2) * 0.8
        b["mistrust"] = 0.2 + sch.get("mistrust", 0.2) * 0.8
        b["helplessness"] = 0.15 + sch.get("failure", 0.2) * 0.5
        b["world_danger"] = 0.2 + sch.get("mistrust", 0.2) * 0.5 + \
            sch.get("abandonment", 0.2) * 0.3
        # social relations graph (spec §2.6)
        self.relations = Relations()
        self.event_queue: list[str] = []

    def _update_beliefs(self, event_type: str, a: dict):
        """Beliefs are updated by experience (slow, bounded drift)."""
        b = self.state.beliefs
        if event_type in ("betrayal", "abandonment"):
            b["mistrust"] = clamp(b["mistrust"] + 0.08, 0, 1)
            b["abandonment"] = clamp(b["abandonment"] + 0.1, 0, 1)
            b["support_available"] = clamp(b["support_available"] - 0.05, 0, 1)
        elif event_type == "rejection":
            b["abandonment"] = clamp(b["abandonment"] + 0.06, 0, 1)
        elif event_type == "threat":
            b["world_danger"] = clamp(b["world_danger"] + 0.1, 0, 1)
        elif event_type in ("humiliation", "criticism"):
            b["helplessness"] = clamp(b["helplessness"] + 0.05, 0, 1)
        elif event_type in ("help", "praise"):
            b["support_available"] = clamp(b["support_available"] + 0.06, 0, 1)
            b["mistrust"] = clamp(b["mistrust"] - 0.03, 0, 1)
            b["helplessness"] = clamp(b["helplessness"] - 0.04, 0, 1)
        elif event_type == "success":
            b["helplessness"] = clamp(b["helplessness"] - 0.05, 0, 1)

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
    def tick(self, dt: float = 1.0, sleeping: bool = False):
        """Advance time: relaxation + physiology (wake or sleep) + relations,
        then process queued events."""
        self.state.relax(self.persona, dt)
        physiology.update(self.state, self.persona, dt, sleeping=sleeping)
        self.relations.relax(dt)
        # process queued events
        while self.event_queue:
            raw = self.event_queue.pop(0)
            self.handle_event(raw)
        return self.state

    def inject(self, raw: str):
        self.event_queue.append(raw)

    def sleep(self, hours: float = 8.0) -> dict:
        """A sleep session: physiology settlement + memory consolidation
        (reconsolidation window, Nader) + reflection (generative_agents)."""
        from .physiology import sleep_session
        r = sleep_session(self.state, self.persona, hours)
        archived = self.memory.consolidate_all(self.state.t)
        insight = self.memory.reflect(self.state.t)
        r["archived"] = archived
        r["insight"] = insight.text if insight else None
        return r

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

        # belief-colored appraisal (PsychSim: beliefs modulate threat/control)
        b = s.beliefs
        a["coping_potential"] = clamp(
            a["coping_potential"] * (1.0 - b["world_danger"] * 0.2
                                      - b["helplessness"] * 0.25), 0, 1)
        if event.type in ("abandonment", "rejection", "betrayal"):
            a["goal_relevance"] = clamp(
                a["goal_relevance"] * (1.0 + b["abandonment"] * 0.35), 0, 1)
        if event.type == "threat":
            a["coping_potential"] = clamp(
                a["coping_potential"] * (1.0 - b["world_danger"] * 0.3), 0, 1)

        # 2.5 social relation update (spec §2.6): who is this about?
        rel_change = self.relations.apply_event(
            raw, event.type, a["valence"], self.persona, s, s.t)

        # beliefs update from experience (associative learning, Beck/Ehlers)
        self._update_beliefs(event.type, a)

        # flashback check (trauma channel, bypasses normal retrieval)
        frag = self.memory.check_flashback(raw, s)
        flashback = False
        if frag is not None:
            # probabilistic intrusion: arousal x depression, low when resources high
            prob = min(1.0, 0.3 + frag.arousal * 0.3
                       + s.depression_tendency * 0.25
                       - s.resources / 400)
            if self.rng.random() < prob:
                flashback = True
                frag.flashbacks += 1
                s.vigilance = clamp(s.vigilance + 0.15, 0, 1)
                # PAD dragged toward trauma state
                p, a0, d = s.pad
                s.pad = (clamp(p - 0.4, -1, 1), clamp(a0 + 0.35, -1, 1),
                         clamp(d - 0.3, -1, 1))
            else:
                # near-miss: avoidance pre-activation (anxiety near trauma cues)
                s.vigilance = clamp(s.vigilance + 0.06, 0, 1)
                s.stress = clamp(s.stress + 3.0, 0, 100)

        # 3. emotion shock
        strength = emotion.apply_appraisal(a, self.persona, s)

        # PVLV-lite reward learning: valence outcome vs expectation (RPE)
        outcome = max(0.0, a["valence"])
        s.rpe = outcome - s.reward_expectation
        s.reward_expectation = clamp(s.reward_expectation + 0.1 * s.rpe, 0, 1)
        # dopamine-like: positive RPE lifts the mood baseline, negative sags it
        if abs(s.rpe) > 0.15:
            mp, ma, md = s.mood_pad
            s.mood_pad = (clamp(mp + s.rpe * 0.03, -1, 1), ma, md)

        # HTM-lite prediction error: surprising event sequences raise
        # arousal/attention (novelty signal)
        if s.last_event_type and event.type != s.last_event_type:
            s.prediction_error = clamp(s.prediction_error + 0.3, 0, 1)
        else:
            s.prediction_error = clamp(s.prediction_error - 0.1, 0, 1)
        s.last_event_type = event.type
        if s.prediction_error > 0.5:
            a["novelty"] = clamp(a["novelty"] + 0.15, 0, 1)

        # 4. stress accumulation + crash check
        stress.apply_shock(a, self.persona, s, self.rng)

        # 5. memory write
        importance = strength * 0.7 + a["goal_relevance"] * 0.3
        item = self.memory.add_episodic(t=s.t, text=raw, valence=a["valence"],
                                        arousal=abs(a["valence"]) * 0.6 + a["novelty"] * 0.4,
                                        importance=importance)
        # Mem0-style self-editing: same-theme entries stop competing
        self.memory.prune_duplicates(item)
        if (event.type in TRAUMA_TYPES and abs(a["valence"]) > 0.6
                and strength > 0.45):
            # trigger words = the event type's keywords (generalized cues)
            self.memory.add_trauma_fragment(
                t=s.t, text=raw, arousal=abs(a["valence"]),
                trigger_words=EVENT_KEYWORDS(event.type)[:3])

        # 6. decision
        decision = self.decide(a, flashback)
        if rel_change:
            decision["relation"] = rel_change
            # NOTE: relation consequences of the behavior itself are applied
            # in act() — in game mode the player's pick overrides the engine's
            # automatic decision, so applying here would double-count.

        # 7. generate action
        snapshot = s.snapshot()
        snapshot["persona_summary"] = self.persona.summary_text()
        snapshot["relations_summary"] = self.relations.summary_text()
        context = [m.as_dict() for m in
                   self.memory.retrieve(raw, s, k=3, rng=self.rng)]
        insight = self.memory.latest_insight()
        if insight:
            context.append({"t": insight.t, "text": f"[自我认知] {insight.text}",
                            "valence": insight.valence, "arousal": 0.3,
                            "importance": 0.6})
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
            "relation_change": rel_change,
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
            "relief": {"neutral_act": 1.8, "confront": 0.3},
            "hopeful": {"neutral_act": 1.4, "confront": 0.7, "seek_support": 0.6},
            "contempt": {"confront": 1.8, "impulsive_attack": 1.2, "vent": 0.8},
            "neutral": {"neutral_act": 1.0},
        }.get(label, {"neutral_act": 1.0})

        # power asymmetry suppresses direct confrontation in negative events
        # (Fanselow defense cascade: no escape -> avoid/freeze)
        power = self.relations.power_pressure(a.get("valence", 0.0))
        if power > 0.25:
            for k in ("confront", "impulsive_attack"):
                if k in tendency:
                    tendency[k] *= 1.0 - power * 0.6
            tendency["avoid"] = tendency.get("avoid", 0) + power * 1.2
            tendency["seek_support"] = tendency.get("seek_support", 0) + power * 0.4

        # impulse & norms (GST -> deviance)
        impulse = morality.compute_impulse(a, self.persona, s)
        s.impulse = impulse
        # opportunity factors lower the barrier (routine activity theory:
        # anonymity / absent guardianship)
        anonymous = any(k in a["text"] for k in
                        ["网上", "深夜", "没人", "匿名", "一个人", "暗处", "无人"])
        deviant_candidate = impulse > (0.45 if anonymous else 0.6)
        norm = morality.norm_check("impulsive_attack", self.persona, s)
        if deviant_candidate and norm["effective"] > 0.5:
            deviant_candidate = False   # norms still hold
        mech = morality.disengagement_pressure(self.persona, s, impulse, self.rng)

        scores = dict(tendency)
        # EMA: appraisal -> coping strategy -> behavior tendencies
        style = coping.select_coping(a, self.persona, s)
        scores = coping.modulate(scores, style)
        # learned helplessness: repeated crashes suppress active coping and
        # raise freezing (Seligman); depression adds rumination-driven passivity
        helpless = min(1.0, s.crash_count * 0.12 + s.depression_tendency * 0.3)
        if helpless > 0:
            for k in ("confront", "vent", "seek_support"):
                if k in scores:
                    scores[k] *= 1.0 - helpless * 0.8
            # Seligman: learned helplessness ends in freezing, not fleeing
            scores["freeze"] = scores.get("freeze", 0) + helpless * 1.4
            scores["avoid"] = scores.get("avoid", 0) + helpless * 0.2
        if deviant_candidate:
            scores["impulsive_attack"] = scores.get("impulsive_attack", 0) + 2.5 + impulse
        scores.setdefault("neutral_act", 0.4)

        # FLAME delayed burst: banked anger erupts once it passes a threshold
        # (repeated suppression / low self-control -> explosion)
        burst = s.suppressed_anger
        if burst > 0.5:
            scores["vent"] = scores.get("vent", 0) + burst * 1.4
            scores["confront"] = scores.get("confront", 0) + burst * 1.2
            scores["impulsive_attack"] = scores.get(
                "impulsive_attack", 0) + burst * 1.1

        # Active-inference heuristic (pymdp-lite): mood as prior preference C
        # — positive mood favors engagement, negative mood withdrawal
        prior = s.mood_pad[0]
        for k in ("confront", "neutral_act", "seek_support"):
            if k in scores:
                scores[k] *= 1.0 + prior * 0.25
        for k in ("avoid", "freeze"):
            if k in scores:
                scores[k] *= 1.0 - prior * 0.25
        # epistemic value: novel situations favor exploration
        if a.get("novelty", 0.0) > 0.55:
            scores["neutral_act"] = scores.get("neutral_act", 0) + 0.3
            scores["seek_support"] = scores.get("seek_support", 0) + 0.15

        behavior = max(scores, key=scores.get)
        return {"behavior": behavior,
                "deviant": behavior in ("impulsive_attack", "impulsive_selfharm", "fight"),
                "impulse": impulse, "norm_effective": norm["effective"],
                "disengagement": mech, "anonymous": anonymous,
                "coping": style}


    # ------------------------------------------------------------------
    def _process_recent_trauma(self):
        """Safe-context retelling consolidates the most recent unprocessed
        trauma fragment (Foa emotional processing; Nader reconsolidation)."""
        for frag in reversed(self.memory.trauma):
            if not frag.consolidated:
                frag.consolidated = True
                return frag
        return None

    def act(self, behavior: str, pipeline: dict | None = None) -> dict:
        """Execute a chosen behavior (game mode: player picks; spec §5.2).

        Behavior consequences update state; deviant behaviors run the
        morality aftermath loop (spec §3.3.3).
        """
        s = self.state
        p = s.pad
        effects = {
            "confront":  {"resources": -5, "self_control": -5},
            "avoid":     {"resources": -2},
            "vent":      {"stress": -10, "resources": -8},
            "seek_support": {"resources": 10, "stress": -8, "guilt": -0.1, "shame": -0.1},
            "impulsive_attack": {"stress": -12, "vigilance": 0.05},
            "impulsive_selfharm": {"stress": -15, "resources": -10, "depression": 0.05},
            "freeze":    {},
            "fight":     {"stress": -10, "resources": -5},
            "dissociate": {"stress": -10},
            "neutral_act": {"stress": -3},
        }
        eff = dict(effects.get(behavior, {}))

        # social support amplifies seek_support payoff (Hobfoll COR: support
        # is a resource; weak ties -> little relief)
        if behavior == "seek_support":
            support = self.relations.support_score()
            eff["resources"] = eff.get("resources", 0) + int(support * 15)
            eff["stress"] = eff.get("stress", 0) - support * 8

        # catharsis: confronting while angry releases some stress
        if behavior == "confront" and p[0] < -0.2 and p[1] > 0.3:
            eff = {**eff, "stress": eff.get("stress", 0) - 8}
        # FLAME: venting/confronting/fighting discharges banked anger
        if behavior in ("vent", "confront", "impulsive_attack", "fight"):
            s.suppressed_anger = max(0.0, s.suppressed_anger - 0.6)
            # discharge also costs the self-control that held it back
            s.self_control = clamp(s.self_control - 3.0, 0, 100)
        # avoidance doesn't resolve the stressor: mood sags slightly
        if behavior == "avoid":
            mp, ma, md = s.mood_pad
            s.mood_pad = (clamp(mp - 0.02, -1, 1), ma, md)
            s.vigilance = clamp(s.vigilance + 0.05, 0, 1)

        s.stress = clamp(s.stress + eff.get("stress", 0), 0, 100)
        s.resources = clamp(s.resources + eff.get("resources", 0), 0, 100)
        s.self_control = clamp(s.self_control + eff.get("self_control", 0), 0, 100)
        s.guilt = clamp(s.guilt + eff.get("guilt", 0), 0, 1)
        s.shame = clamp(s.shame + eff.get("shame", 0), 0, 1)
        s.vigilance = clamp(s.vigilance + eff.get("vigilance", 0), 0, 1)
        s.depression_tendency = clamp(
            s.depression_tendency + eff.get("depression", 0), 0, 1)

        deviant = behavior in ("impulsive_attack", "impulsive_selfharm", "fight")
        if deviant:
            act_res = {"success": self.rng.random() < 0.7,
                       "punished": self.rng.random() < 0.3}
            morality.after_deviance(self.persona, s, act_res)
        # behavior consequences on relations (confront/fight vs power,
        # support deepens the closest bond)
        rel_after = self.relations.apply_behavior(behavior)
        if rel_after:
            eff["relation"] = rel_after
        if behavior in ("impulsive_selfharm", "impulsive_attack"):
            s.impulse = 0.3  # discharge

        # safe-context retelling processes the most recent trauma fragment
        processed = None
        if self.memory.trauma and behavior in ("seek_support", "confront"):
            if behavior == "seek_support" or s.resources > 50:
                processed = self._process_recent_trauma()

        # memory of the response
        self.memory.add_episodic(t=s.t, text=f"[应对] {behavior}", valence=p[0],
                                 arousal=abs(p[0]), importance=0.4)
        snapshot = s.snapshot()
        snapshot["persona_summary"] = self.persona.summary_text()
        snapshot["relations_summary"] = self.relations.summary_text()
        action = self.llm.generate(snapshot,
                                   [m.as_dict() for m in
                                    self.memory.retrieve("", s, k=2, rng=self.rng)],
                                   {"behavior": behavior, "deviant": deviant})
        return {"behavior": behavior, "deviant": deviant, "action": action.text,
                "effects": eff,
                "processed_trauma": processed.fragment if processed else None}

    def options(self, pipeline: dict | None = None) -> list[dict]:
        """Candidate behaviors for the player (game mode), with recommendation."""
        s = self.state
        if s.crashed:
            return [{"kind": k, "recommended": k == s.crash_type}
                    for k in dict.fromkeys([s.crash_type, "freeze"])]

        base = ["confront", "avoid", "vent", "seek_support"]
        if s.impulse > 0.55:
            base.append("impulsive_attack")
        if s.depression_tendency > 0.4:
            base.append("impulsive_selfharm")
        if s.emotion_label in ("calm", "neutral", "joy"):
            base.insert(0, "neutral_act")

        recommended = {
            "anger": "confront", "fear": "avoid", "anxious": "avoid",
            "sadness": "vent", "shame": "avoid", "distress": "vent",
            "joy": "neutral_act", "calm": "neutral_act", "neutral": "neutral_act",
        }.get(s.emotion_label, "neutral_act")

        out = []
        for k in dict.fromkeys(base):
            out.append({"kind": k, "recommended": k == recommended})
        return out


def EVENT_KEYWORDS(etype: str) -> list[str]:
    from .appraisal import EVENT_TYPES
    return EVENT_TYPES.get(etype, {}).get("kw", [])
