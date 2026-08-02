"""Social relations graph (spec §2.6).

Per person: affinity (好感) / trust (信任) / power_diff (权力差) /
attachment_activation (依恋激活) / interaction history.
Social support = sum over relations -> COR resource (Hobfoll).

Theory anchors:
- Kelley interdependence, Sternberg: relations as multi-dimensional states
- Ehlers & Clark / attachment (Bowlby): betrayal & abandonment crush trust,
  activate the attachment system (modulated by attachment style)
- Fanselow defense cascade: power asymmetry suppresses confrontation
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from .persona import Persona
from .state import State
from ._clamp import clamp

# role -> (canonical name, kind, base power_diff)
ROLE_TABLE: list[tuple[re.Pattern, str, str, float]] = [
    (re.compile(r"老板|上司|领导|经理|主管|导师"), "老板", "authority", 0.35),
    (re.compile(r"同事|搭档|队友"), "同事", "colleague", 0.1),
    (re.compile(r"女朋友|男朋友|女友|男友|对象|恋人|伴侣|老婆|老公|妻子|丈夫|配偶"), "伴侣", "partner", 0.0),
    (re.compile(r"朋友|哥们|闺蜜|好友|死党"), "朋友", "friend", 0.0),
    (re.compile(r"妈妈|爸爸|母亲|父亲|父母|家人|儿子|女儿|孩子|弟弟|妹妹|哥哥|姐姐"), "家人", "family", 0.2),
    (re.compile(r"陌生人|网友|路人|房东"), "陌生人", "stranger", 0.0),
]

# event type -> (affinity delta, trust delta, attachment delta) for negative events
EVENT_IMPACT: dict[str, tuple[float, float, float]] = {
    "betrayal":    (-0.18, -0.30, 0.30),
    "abandonment": (-0.22, -0.20, 0.40),
    "rejection":   (-0.15, -0.10, 0.25),
    "humiliation": (-0.12, -0.10, 0.15),
    "threat":      (-0.12, -0.12, 0.20),
    "conflict":    (-0.10, -0.06, 0.10),
    "criticism":   (-0.06, -0.04, 0.05),
    "loss":        (-0.05,  0.00, 0.10),
    "praise":      ( 0.14,  0.10, -0.05),
    "help":        ( 0.16,  0.14, -0.08),
    "success":     ( 0.08,  0.06, -0.03),
}


@dataclass
class Relation:
    name: str
    kind: str = "stranger"
    affinity: float = 0.5          # 0-1 好感
    trust: float = 0.5             # 0-1 信任
    power_diff: float = 0.0        # -1..1; >0 = 对方权力更大
    attachment_activation: float = 0.0   # 0-1, decays slowly
    interactions: int = 0
    last_t: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


class Relations:
    def __init__(self):
        self.people: dict[str, Relation] = {}

    # ------------------------------------------------------------------
    def get_or_create(self, name: str, kind: str, power_diff: float) -> Relation:
        rel = self.people.get(name)
        if rel is None:
            rel = Relation(name=name, kind=kind, power_diff=power_diff)
            self.people[name] = rel
        return rel

    def infer_subject(self, raw: str) -> tuple[str, str, float] | None:
        """Find who this event is about, from raw text (MockLLM path)."""
        for pat, name, kind, power in ROLE_TABLE:
            if pat.search(raw):
                return name, kind, power
        return None

    # ------------------------------------------------------------------
    def apply_event(self, raw: str, event_type: str, valence: float,
                    persona: Persona, state: State, t: float) -> dict | None:
        """Event pipeline hook: update the involved relation (if identifiable).
        Returns a change record for the pipeline log, or None."""
        subj = self.infer_subject(raw)
        if subj is None:
            return None
        name, kind, power = subj
        rel = self.get_or_create(name, kind, power)
        rel.interactions += 1
        rel.last_t = t

        d_aff, d_trust, d_attach = EVENT_IMPACT.get(event_type, (0.0, 0.0, 0.0))
        # positive events scale with relevance; negative scaled by valence
        if valence > 0.1:
            factor = 0.6 + 0.8 * min(1.0, valence)
        elif valence < -0.1:
            factor = 0.6 + 1.4 * min(1.0, -valence)
        else:
            factor = 0.3
        d_aff *= factor
        d_trust *= factor

        # attachment style modulates the attachment-system response
        style_mod = {"preoccupied": 1.5, "fearful": 1.3,
                     "safe": 1.0, "dismissing": 0.4}.get(persona.attachment, 1.0)
        d_attach *= style_mod

        rel.affinity = clamp(rel.affinity + d_aff, 0, 1)
        rel.trust = clamp(rel.trust + d_trust, 0, 1)
        rel.attachment_activation = clamp(
            rel.attachment_activation + d_attach, 0, 1)
        # authority events shift power asymmetry upward
        if rel.kind == "authority" and event_type in ("humiliation", "threat",
                                                      "criticism", "conflict"):
            rel.power_diff = clamp(rel.power_diff + 0.12, -1, 1)

        return {"name": name, "kind": kind,
                "d_affinity": round(d_aff, 3), "d_trust": round(d_trust, 3),
                "d_attachment": round(d_attach, 3),
                "affinity": round(rel.affinity, 3),
                "trust": round(rel.trust, 3),
                "power_diff": round(rel.power_diff, 3)}

    def apply_behavior(self, behavior: str) -> dict | None:
        """Behavior consequences on relations (confront/fight vs power; support
        deepens the closest bond)."""
        if not self.people:
            return None
        if behavior in ("confront", "fight"):
            # standing up to an authority reduces the perceived power gap
            sub = max(self.people.values(), key=lambda r: r.power_diff)
            if sub.power_diff > 0.05:
                sub.power_diff = clamp(sub.power_diff - 0.15, -1, 1)
                return {"name": sub.name, "power_diff": round(sub.power_diff, 3)}
        elif behavior == "seek_support":
            # support-seeking strengthens the closest bond
            best = max(self.people.values(), key=lambda r: r.affinity * r.trust)
            best.affinity = clamp(best.affinity + 0.02, 0, 1)
            best.trust = clamp(best.trust + 0.015, 0, 1)
            return {"name": best.name, "affinity": round(best.affinity, 3),
                    "trust": round(best.trust, 3)}
        return None

    # ------------------------------------------------------------------
    def support_score(self) -> float:
        """Social support as a COR resource (0-1): sum of affinity x trust."""
        score = sum(r.affinity * r.trust for r in self.people.values())
        return clamp(score * 0.35, 0, 1)

    def power_pressure(self, valence: float) -> float:
        """How suppressed the person feels by a power asymmetry right now
        (Fanselow: no escape -> freeze). Only matters in negative events."""
        if valence >= 0:
            return 0.0
        top = max((r.power_diff for r in self.people.values()), default=0.0)
        return clamp(top, 0, 1)

    def relax(self, dt: float):
        """Attachment activation decays slowly toward 0 (minutes-hours)."""
        for r in self.people.values():
            if r.attachment_activation > 0:
                r.attachment_activation = clamp(
                    r.attachment_activation - 0.0008 * dt, 0, 1)

    # ------------------------------------------------------------------
    def summary_text(self) -> str:
        """Short natural-language summary for LLM injection."""
        if not self.people:
            return "关系：暂无亲近之人（孤立）"
        parts = []
        for r in sorted(self.people.values(),
                        key=lambda r: -r.affinity * r.trust):
            parts.append(f"{r.name}(好感{r.affinity:.2f} 信任{r.trust:.2f})")
        return "关系：" + "、".join(parts)

    def snapshot(self) -> list[dict]:
        return [r.as_dict() for r in self.people.values()]
