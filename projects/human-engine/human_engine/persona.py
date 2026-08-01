"""Persona: static attributes of the simulated person (spec §1)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Persona:
    """Static attributes. Set at creation; drift slowly over time (spec §1.3)."""

    # Big Five (0-1)
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5

    # Dark triad (0-1)
    narcissism: float = 0.2
    machiavellianism: float = 0.2
    psychopathy: float = 0.1

    # Gray RST (0-1)
    bis: float = 0.5          # behavioral inhibition (anxiety)
    bas: float = 0.5          # behavioral activation (impulsivity)

    # Attachment: safe / preoccupied / dismissing / fearful
    attachment: str = "safe"

    # Locus of control (1 = internal)
    locus: float = 0.5

    # Kohlberg stage 1-6
    kohlberg: int = 3

    # Moral foundations weights
    mft: dict = field(default_factory=lambda: {
        "care": 0.6, "fairness": 0.6, "loyalty": 0.5,
        "authority": 0.4, "purity": 0.3, "liberty": 0.5,
    })

    # Early maladaptive schemas (Young): threat amplification per schema
    schemas: dict = field(default_factory=lambda: {
        "abandonment": 0.3, "mistrust": 0.3, "defectiveness": 0.2,
        "failure": 0.2, "subjugation": 0.2, "emotional_deprivation": 0.2,
    })

    # Resilience (Bonanno): raises crash threshold, speeds recovery
    resilience: float = 0.5

    # Baseline physiology
    energy_max: float = 100.0
    sleep_need: float = 8.0  # hours per day

    # Life history: list of {t, text, valence, arousal, importance}
    history: list = field(default_factory=list)

    # ------------------------------------------------------------------
    def mood_attractor(self) -> tuple[float, float, float]:
        """ALMA: personality -> mood baseline in PAD space."""
        p = (self.extraversion * 0.5 + self.agreeableness * 0.3 - self.neuroticism * 0.5)
        a = (self.extraversion * 0.2 + self.neuroticism * 0.25)
        d = (self.extraversion * 0.2 + self.openness * 0.15 - self.neuroticism * 0.15)
        return _clamp(p, -0.5, 0.5), _clamp(a, -0.3, 0.3), _clamp(d, -0.4, 0.4)

    def attachment_bias(self, event_type: str) -> float:
        """How strongly attachment-relevant events hit this persona."""
        table = {
            "safe": {"abandonment": 1.0, "rejection": 0.7, "betrayal": 0.8},
            "preoccupied": {"abandonment": 1.8, "rejection": 1.5, "betrayal": 1.5},
            "dismissing": {"abandonment": 0.4, "rejection": 0.3, "betrayal": 0.8},
            "fearful": {"abandonment": 1.6, "rejection": 1.3, "betrayal": 1.6},
        }
        return table.get(self.attachment, {}).get(event_type, 1.0)

    def schema_hits(self, event_type: str, text: str) -> list[tuple[str, float]]:
        """Which schemas this event touches, and by how much."""
        hits = []
        schema_keywords = {
            "abandonment": ["离开", "抛弃", "分手", "不要我", "alone", "abandon"],
            "mistrust": ["骗", "背叛", "利用", "lies", "betray"],
            "defectiveness": ["没用", "废物", "失败者", "丢人", "useless", "worthless"],
            "failure": ["失败", "搞砸", "不及格", "fail", "screw"],
            "subjugation": ["命令", "必须服从", "控制", "forced", "control"],
            "emotional_deprivation": ["没人关心", "冷漠", "ignore", "cold"],
        }
        for schema, kws in schema_keywords.items():
            if any(k in text for k in kws):
                hits.append((schema, self.schemas.get(schema, 0.0)))
        return hits

    def summary_text(self) -> str:
        """Natural-language persona summary for LLM injection (spec §4.3)."""
        lines = [
            f"人格：外倾{self.extraversion:.2f} 宜人{self.agreeableness:.2f} "
            f"尽责{self.conscientiousness:.2f} 神经质{self.neuroticism:.2f} "
            f"开放{self.openness:.2f}",
            f"依恋风格：{self.attachment}；BIS(焦虑)={self.bis:.2f} BAS(冲动)={self.bas:.2f}",
            f"核心图式：{', '.join(k for k, v in sorted(self.schemas.items(), key=lambda x: -x[1])[:3] if v > 0.25) or '无显著'}",
        ]
        return "\n".join(lines)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def default_persona() -> Persona:
    """A starting persona with a mildly traumatic history."""
    return Persona(
        neuroticism=0.62,
        bis=0.6,
        bas=0.45,
        attachment="fearful",
        schemas={
            "abandonment": 0.55, "mistrust": 0.5, "defectiveness": 0.35,
            "failure": 0.3, "subjugation": 0.2, "emotional_deprivation": 0.4,
        },
        history=[
            {"t": -3650, "text": "母亲在自己 8 岁时离家，之后再无联系",
             "valence": -0.9, "arousal": 0.8, "importance": 0.95},
            {"t": -800, "text": "高中时被同学当众羞辱并孤立一年",
             "valence": -0.8, "arousal": 0.7, "importance": 0.85},
        ],
    )
