"""Dynamic state of the simulated person (spec §2)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from .persona import Persona, _clamp


@dataclass
class State:
    """All dynamic state, updated by the engine each tick."""

    # --- emotion layer (seconds) ---
    pad: tuple[float, float, float] = (0.0, 0.0, 0.0)   # P/A/D in [-1,1]
    emotion_label: str = "neutral"
    emotion_strength: float = 0.0
    emotion_decay_mod: float = 1.0   # WASABI secondary emotion: cognitive
                                     # depth slows relaxation (0.5-1.0)
    suppressed_anger: float = 0.0    # FLAME delayed burst: banked anger 0-1
    decay_lambda: float = 0.0005     # shared with memory (MemoryBank: emotion
                                     # decay uses the same time base)

    # --- mood layer (minutes-hours) ---
    mood_pad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    depression_tendency: float = 0.0   # 0-1
    vigilance: float = 0.0             # 0-1 (hypervigilance after trauma)

    # --- stress & resources ---
    stress: float = 10.0               # 0-100
    resources: float = 80.0            # 0-100
    threshold: float = 70.0            # crash threshold; permanently drops
    wear: float = 0.0                  # allostatic wear (permanent)
    crash_count: int = 0               # total crashes (learned helplessness)
    gas_phase: str = "normal"          # normal/alert/resistance/exhausted/crashed
    crashed: bool = False
    crash_type: str = ""               # freeze / fight / dissociate

    # --- cognitive control ---
    self_control: float = 80.0         # 0-100
    impulse: float = 0.0               # 0-1

    # --- morality ---
    guilt: float = 0.0                 # 0-1
    shame: float = 0.0                 # 0-1
    moral_disengagement: dict = field(default_factory=lambda: {m: 0.2 for m in
        ["moral_justification", "euphemism", "diffusion", "blaming_victim",
         "dehumanization", "displacement", "distortion", "minimizing"]})

    # --- beliefs (PsychSim-style explicit belief layer) ---
    # 0-1 strengths about self/world/others; color appraisal (Ehlers & Clark
    # threat overestimation; Beck negative triad)
    beliefs: dict = field(default_factory=lambda: {
        "world_danger": 0.3,       # 世界是危险的
        "abandonment": 0.3,        # 我总会被抛弃
        "mistrust": 0.3,           # 人不值得信任
        "helplessness": 0.2,       # 我无能为力
        "support_available": 0.4,  # 有人会帮我
    })

    # --- physiology ---
    energy: float = 80.0
    sleep_debt: float = 0.0            # hours
    phase: float = 0.0                 # circadian phase 0-24h (updated per tick)

    # --- learning (PVLV-lite RPE + HTM-lite prediction error) ---
    reward_expectation: float = 0.5    # expected valence of the world 0-1
    rpe: float = 0.0                   # last reward prediction error
    prediction_error: float = 0.0      # event-sequence surprise 0-1
    last_event_type: str = ""

    # --- time ---
    t: float = 0.0                     # simulated seconds

    # --- episode log (last event pipeline record) ---
    last_pipeline: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def relax(self, persona: Persona, dt: float):
        """Relaxation dynamics (WASABI + ALMA), spec §3.1 step 1."""
        p, a, d = self.pad
        mp, ma, md = self.mood_pad
        # WASABI: secondary (cognitive) emotions relax slower; MemoryBank:
        # emotion decay shares the same time base as the forgetting curve
        mod = self.emotion_decay_mod
        k = min(1.0, self.decay_lambda * 16 * dt)   # 0.008/s at default
        # so a 60s tick only decays ~48% of the shock instead of erasing it
        self.pad = (
            _clamp(p + (mp - p) * k * mod, -1, 1),
            _clamp(a + (ma - a) * k * 1.25 * mod, -1, 1),
            _clamp(d + (md - d) * k * 0.75 * mod, -1, 1),
        )
        # banked anger digests slowly; cognitive depth fades back to 1.0;
        # prediction error (htm-lite surprise) decays over time
        self.suppressed_anger = _clamp(self.suppressed_anger - 0.0006 * dt, 0, 1)
        self.emotion_decay_mod = _clamp(self.emotion_decay_mod + 0.0002 * dt, 0.5, 1.0)
        self.prediction_error = _clamp(self.prediction_error - 0.001 * dt, 0, 1)
        # mood -> personality attractor (very slow, hours)
        ap, aa, ad = persona.mood_attractor()
        k = min(1.0, 0.0005 * dt)
        self.mood_pad = (
            _clamp(mp + (ap - mp) * k, -1, 1),
            _clamp(ma + (aa - ma) * k, -1, 1),
            _clamp(md + (ad - md) * k, -1, 1),
        )
        # stress decays slowly; resources recover; self-control recovers.
        # fatigue (sleep debt) slows recovery and stress settling (embodied
        # cognition: a sleep-deprived person regulates worse)
        from .physiology import fatigue
        fat = fatigue(self, persona)
        self.stress = _clamp(
            self.stress - 0.004 * dt * (1 + persona.resilience * 0.5)
            * (1 - fat * 0.35), 0, 100)
        self.resources = _clamp(
            self.resources + 0.008 * dt * (1 + persona.resilience), 0, 100)
        self.self_control = _clamp(
            self.self_control + 0.02 * dt * (1 - fat * 0.5), 0, 100)
        # guilt/shame fade
        self.guilt = _clamp(self.guilt - 0.001 * dt, 0, 1)
        self.shame = _clamp(self.shame - 0.0008 * dt, 0, 1)
        # gas phase
        self._update_gas_phase()
        self.t += dt

    def _update_gas_phase(self):
        if self.crashed:
            self.gas_phase = "crashed"
        elif self.stress > 80:
            self.gas_phase = "exhausted"
        elif self.stress > 55:
            self.gas_phase = "resistance"
        elif self.stress > 30:
            self.gas_phase = "alert"
        else:
            self.gas_phase = "normal"

    def snapshot(self) -> dict:
        return asdict(self)

    def short_summary(self) -> str:
        p, a, d = self.pad
        return (f"[t={self.t:.0f}s] 情绪={self.emotion_label}(P{p:+.2f} A{a:+.2f} D{d:+.2f}) "
                f"应激={self.stress:.0f}/阈值{self.threshold:.0f} 资源={self.resources:.0f} "
                f"自控={self.self_control:.0f} 阶段={self.gas_phase}"
                + (" [崩溃!]" if self.crashed else ""))
