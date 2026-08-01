"""Memory systems (spec §2.5): episodic store + retrieval, trauma channel, semantics."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict

from .persona import Persona
from .state import State


@dataclass
class MemoryItem:
    t: float
    text: str
    valence: float
    arousal: float
    importance: float
    consolidation: float = 1.0   # 0..1, grows with offline reflection

    def as_dict(self):
        return asdict(self)


@dataclass
class TraumaFragment:
    """Sensory fragment in the trauma channel (Ehlers & Clark / DRT)."""
    trigger_words: list
    fragment: str            # raw sensory fragment text
    arousal: float
    t: float
    flashbacks: int = 0      # how many times it has intruded
    consolidated: bool = False   # once contextualized, stops forcing flashbacks


class Memory:
    def __init__(self, decay_lambda: float = 0.0005):
        self.episodic: list[MemoryItem] = []
        self.trauma: list[TraumaFragment] = []
        self.decay_lambda = decay_lambda   # forgetting curve rate

    # --- write ----------------------------------------------------------
    def add_episodic(self, t: float, text: str, valence: float,
                     arousal: float, importance: float):
        item = MemoryItem(t=t, text=text, valence=valence,
                          arousal=arousal, importance=importance)
        self.episodic.append(item)
        return item

    def add_trauma_fragment(self, t: float, text: str, arousal: float,
                            trigger_words: list[str]):
        frag = TraumaFragment(trigger_words=trigger_words, fragment=text,
                              arousal=arousal, t=t)
        self.trauma.append(frag)
        return frag

    # --- retrieval ------------------------------------------------------
    def retrieve(self, query: str, state: State, k: int = 4,
                 rng: random.Random | None = None) -> list[MemoryItem]:
        """Score = recency decay x importance x (1+arousal) x mood congruence.
        (generative_agents x Ebbinghaus x Bower)"""
        rng = rng or random
        mood_p = state.mood_pad[0]
        scored = []
        rumination = state.depression_tendency > 0.4  # Nolen-Hoeksema: rumination
        for it in self.episodic:
            age = max(0.0, state.t - it.t)
            recency = math.exp(-self.decay_lambda * age)
            congruence = 1.0 + max(-0.5, min(0.5, it.valence * mood_p))
            score = recency * it.importance * (1.0 + it.arousal * 0.5) * congruence
            if rumination and it.valence < -0.3:
                score *= 1.0 + state.depression_tendency * 0.6
            scored.append((score + rng.uniform(0, 0.05), it))
        scored.sort(key=lambda x: -x[0])
        return [it for _, it in scored[:k]]

    def check_flashback(self, text: str, state: State) -> TraumaFragment | None:
        """Trauma channel: trigger words force intrusive flashback (spec §3.3.1)."""
        for frag in self.trauma:
            if frag.consolidated:
                continue
            if any(w in text for w in frag.trigger_words):
                return frag
        return None

    def consolidate(self, fragment: TraumaFragment):
        """Contextualize a trauma fragment (Foa emotional processing)."""
        fragment.consolidated = True

    def summarize(self) -> str:
        return (f"episodic={len(self.episodic)} trauma_fragments={len(self.trauma)} "
                f"consolidated={sum(1 for f in self.trauma if f.consolidated)}")
