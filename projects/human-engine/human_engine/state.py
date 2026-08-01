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

    # --- physiology ---
    energy: float = 80.0
    sleep_debt: float = 0.0            # hours

    # --- time ---
    t: float = 0.0                     # simulated seconds

    # --- episode log (last event pipeline record) ---
    last_pipeline: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def relax(self, persona: Persona, dt: float):
        """Relaxation dynamics (WASABI + ALMA), spec §3.1 step 1."""
        p, a, d = self.pad
        mp, ma, md = self.mood_pad
        # emotion -> mood baseline (slow: emotion lasts minutes, ~6%/s)
        # so a 60s tick only decays ~48% of the shock instead of erasing it
        self.pad = (
            _clamp(p + (mp - p) * min(1.0, 0.008 * dt), -1, 1),
            _clamp(a + (ma - a) * min(1.0, 0.010 * dt), -1, 1),
            _clamp(d + (md - d) * min(1.0, 0.006 * dt), -1, 1),
        )
        # mood -> personality attractor (very slow, hours)
        ap, aa, ad = persona.mood_attractor()
        k = min(1.0, 0.0005 * dt)
        self.mood_pad = (
            _clamp(mp + (ap - mp) * k, -1, 1),
            _clamp(ma + (aa - ma) * k, -1, 1),
            _clamp(md + (ad - md) * k, -1, 1),
        )
        # stress decays slowly; resources recover; self-control recovers
        self.stress = _clamp(self.stress - 0.004 * dt * (1 + persona.resilience * 0.5), 0, 100)
        self.resources = _clamp(self.resources + 0.008 * dt * (1 + persona.resilience), 0, 100)
        self.self_control = _clamp(self.self_control + 0.02 * dt, 0, 100)
        # guilt/shame fade
        self.guilt = _clamp(self.guilt - 0.001 * dt, 0, 1)
        self.shame = _clamp(self.shame - 0.0008 * dt, 0, 1)
        # physiology
        self.energy = _clamp(self.energy - 0.002 * dt, 0, persona.energy_max)
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
