"""Memory systems (spec §2.5): episodic store + retrieval, trauma channel,
semantic self-narratives, offline consolidation.

Theory anchors:
- Ebbinghaus forgetting curve; McGaugh emotional consolidation
- generative_agents: retrieval = recency x importance x relevance
- ACT-R base-level activation + associative boost (shared-cue relevance)
- Letta/MemGPT: memory pressure -> archival (working memory stays small)
- Mem0: self-editing — supersede duplicate/same-topic entries
- Nader reconsolidation: sleep is the consolidation window
- Conway: semantic self-narratives distilled from episodic life events
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field, asdict

from .persona import Persona
from .state import State

MAX_EPISODIC = 120          # memory pressure threshold (archival)

# theme words used for reflection clustering & relevance association
TOPIC_WORDS = [
    "抛弃", "背叛", "羞辱", "嘲笑", "孤立", "拒绝", "分手", "威胁", "失去",
    "失败", "骗", "离开", "冷漠", "批评", "抛弃", "忽视", "不信任", "争吵",
]


def _topics(text: str) -> set[str]:
    return {w for w in TOPIC_WORDS if w in text}


@dataclass
class MemoryItem:
    t: float
    text: str
    valence: float
    arousal: float
    importance: float
    consolidation: float = 0.5   # 0..1, grows during sleep (McGaugh/Nader)
    archived: bool = False       # pressure-archived: excluded from retrieval
    superseded: bool = False     # replaced by a later duplicate (Mem0)

    def as_dict(self):
        return asdict(self)


@dataclass
class SemanticItem:
    """Self-narrative distilled by reflection (Conway semantic memory)."""
    text: str
    t: float
    valence: float
    confidence: float = 0.5

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
        self.semantic: list[SemanticItem] = []
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

    def add_semantic(self, text: str, t: float, valence: float,
                     confidence: float = 0.5) -> SemanticItem:
        """Add a self-narrative insight; same-topic insight is UPDATEd
        (Mem0 self-edit style)."""
        new = SemanticItem(text=text, t=t, valence=valence,
                           confidence=confidence)
        for i, s in enumerate(self.semantic):
            if _topics(s.text) & _topics(text):
                self.semantic[i] = new
                return new
        self.semantic.append(new)
        return new

    # --- self-editing (Mem0) -------------------------------------------
    def prune_duplicates(self, item: MemoryItem):
        """A new event supersedes older entries of the same theme (they keep
        existing but stop competing in retrieval)."""
        new_topics = _topics(item.text)
        if not new_topics:
            return 0
        n = 0
        for old in self.episodic:
            if old is item or old.superseded:
                continue
            if _topics(old.text) & new_topics and old.t < item.t - 3600:
                old.superseded = True
                old.importance *= 0.5
                n += 1
        return n

    # --- retrieval ------------------------------------------------------
    def retrieve(self, query: str, state: State, k: int = 4,
                 rng: random.Random | None = None) -> list[MemoryItem]:
        """Score = recency decay x importance x (1+arousal) x mood congruence
        x thematic relevance (generative_agents / ACT-R associative boost)
        x consolidation. (Ebbinghaus x Bower x generative_agents)"""
        rng = rng or random
        mood_p = state.mood_pad[0]
        q_topics = _topics(query)
        scored = []
        rumination = state.depression_tendency > 0.4  # Nolen-Hoeksema
        for it in self.episodic:
            if it.archived or it.superseded:
                continue
            age = max(0.0, state.t - it.t)
            recency = math.exp(-self.decay_lambda * age)
            congruence = 1.0 + max(-0.5, min(0.5, it.valence * mood_p))
            relevance = 1.0 + min(1.0, len(q_topics & _topics(it.text)) * 0.4)
            score = (recency * it.importance * (1.0 + it.arousal * 0.5)
                     * congruence * relevance * (0.4 + 0.6 * it.consolidation))
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

    # --- offline consolidation (sleep window, Letta + Nader) ------------
    def consolidate_all(self, t: float) -> int:
        """Sleep-time: episodic consolidation grows; under memory pressure
        the oldest low-importance entries are archived (Letta archival)."""
        # memory pressure -> archive only the excess over the cap, so the
        # active set stays at MAX_EPISODIC instead of draining to zero
        archived = 0
        active = sum(1 for it in self.episodic if not it.archived)
        over = active - MAX_EPISODIC
        if over > 0:
            candidates = sorted(
                (it for it in self.episodic if not it.archived),
                key=lambda x: (x.importance, -x.t))
            for it in candidates[:over]:
                it.archived = True
                archived += 1
        for it in self.episodic:
            if it.consolidation < 1.0:
                it.consolidation = min(1.0, it.consolidation + 0.3)
        return archived

    def reflect(self, t: float) -> SemanticItem | None:
        """generative_agents reflection: cluster recent negative experiences
        into a self-narrative insight (semantic memory)."""
        recent = [it for it in self.episodic
                  if it.t > t - 86400 * 3 and it.valence < -0.3]
        if len(recent) < 3:
            return None
        cnt = Counter()
        for it in recent:
            cnt.update(_topics(it.text))
        if not cnt:
            return None
        topic, n = cnt.most_common(1)[0]
        text = (f"我好像总在经历「{topic}」相关的事"
                f"（最近 {len(recent)} 次负面经历里，{n} 次与此有关）")
        return self.add_semantic(text, t, -0.4,
                                 confidence=min(1.0, 0.3 + n * 0.15))

    def latest_insight(self) -> SemanticItem | None:
        return self.semantic[-1] if self.semantic else None

    def summarize(self) -> str:
        active = sum(1 for it in self.episodic
                     if not it.archived and not it.superseded)
        return (f"episodic={len(self.episodic)}(活跃{active}) "
                f"semantic={len(self.semantic)} "
                f"trauma_fragments={len(self.trauma)} "
                f"consolidated={sum(1 for f in self.trauma if f.consolidated)}")
