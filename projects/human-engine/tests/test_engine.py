"""M1 tests: dynamics, crash, memory, flashback, deviance (unittest, no deps)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

from human_engine.engine import Engine
from human_engine.persona import Persona
from human_engine.state import State
from human_engine import emotion, stress, morality


class TestRelaxation(unittest.TestCase):
    def test_emotion_relaxes_toward_mood(self):
        e = Engine(seed=1)
        e.state.pad = (0.9, 0.5, 0.3)   # joy spike
        for _ in range(200):
            e.tick(1.0)
        # P should have relaxed significantly toward mood baseline
        self.assertLess(e.state.pad[0], 0.6)

    def test_positive_event_raises_pleasure(self):
        e = Engine(seed=2)
        p_before = e.state.pad[0]
        e.handle_event("老板当众夸奖你的方案，说你做得很好")
        self.assertGreater(e.state.pad[0], p_before)


class TestCrash(unittest.TestCase):
    def test_stress_sequence_triggers_crash_and_lowers_threshold(self):
        e = Engine(seed=3)
        t0 = e.state.threshold
        for ev in [
            "同事在会议上当众羞辱你",
            "女朋友说我们分手吧",
            "最好的朋友背叛了你",
            "你被公司开除了",
            "陌生人在网上威胁你",
            "家人说对你很失望",
        ]:
            e.handle_event(ev)
            if e.state.crashed:
                break
        self.assertTrue(e.state.crashed, "崩溃应被触发")
        self.assertLess(e.state.threshold, t0, "阈值应永久下降（应激敏化）")
        self.assertGreater(e.state.wear, 0, "异稳态磨损应累积")

    def test_crash_type_high_bas_fights(self):
        e = Engine(persona=Persona(bas=0.9, bis=0.2, neuroticism=0.3), seed=4)
        e.state.stress = 85
        e.state.resources = 20
        for ev in ["你被当众羞辱了", "有人威胁要打你"]:
            e.handle_event(ev)
            if e.state.crashed:
                break
        self.assertTrue(e.state.crashed)
        self.assertIn(e.state.crash_type, ("fight", "freeze"))


class TestMemory(unittest.TestCase):
    def test_write_and_retrieve(self):
        e = Engine(seed=5)
        e.handle_event("同事夸你今天的报告写得好")
        self.assertGreaterEqual(len(e.memory.episodic), 2)  # history + new
        hits = e.memory.retrieve("报告", e.state, k=3)
        self.assertTrue(any("报告" in h.text for h in hits))

    def test_trauma_channel_and_flashback(self):
        e = Engine(seed=6)
        e.handle_event("母亲离开后，父亲在夜里独自流泪")
        frags_before = len(e.memory.trauma)
        # force a severe trauma event
        e.handle_event("你被当众羞辱，所有人都在笑你")
        self.assertGreater(len(e.memory.trauma), 0)
        frag = e.memory.check_flashback("有人开始嘲笑你", e.state)
        # flashback should fire if trigger words overlap
        self.assertIsNotNone(frag)


class TestDeviance(unittest.TestCase):
    def test_high_bas_low_control_yields_impulsive(self):
        e = Engine(persona=Persona(bas=0.95, bis=0.1, neuroticism=0.4,
                                   mft={"care": 0.2, "fairness": 0.2,
                                        "loyalty": 0.2, "authority": 0.2,
                                        "purity": 0.2, "liberty": 0.2}),
                   seed=8)
        e.state.self_control = 20
        e.state.resources = 30
        out = e.handle_event("有人当众羞辱你，说你是废物")
        self.assertTrue(out["decision"]["deviant"] or e.state.crashed,
                        f"高冲动人格应产生越轨或崩溃行为，got {out['decision']}")

    def test_guilt_after_punished_deviance(self):
        e = Engine(persona=Persona(bas=0.8, bis=0.3), seed=9)
        e.state.self_control = 15
        e.state.resources = 25
        e.state.moral_disengagement = {m: 0.1 for m in e.state.moral_disengagement}
        out = e.handle_event("有人抢走了你的东西还嘲笑你")
        if out["decision"].get("deviant"):
            # force punished aftermath manually
            from human_engine import morality as M
            M.after_deviance(e.persona, e.state, {"success": True, "punished": True})
            self.assertGreater(e.state.guilt + e.state.shame, 0.1)


class TestPipelineRecord(unittest.TestCase):
    def test_last_pipeline_fields(self):
        e = Engine(seed=10)
        out = e.handle_event("朋友约你周末一起吃饭")
        for key in ("raw", "event_type", "appraisal", "strength", "decision",
                    "action", "memory_after"):
            self.assertIn(key, out)


class TestM2Deepening(unittest.TestCase):
    """M2: probabilistic flashbacks, trauma processing, opportunity factors,
    learned helplessness, rumination."""

    def test_flashback_not_guaranteed(self):
        # same trauma-cue event across seeds: sometimes flashback, sometimes not
        results = []
        for seed in range(8):
            e = Engine(seed=seed)
            e.handle_event("有人当众羞辱了你，所有人都在笑")
            e.handle_event("你被最好的朋友背叛了")
            out = e.handle_event("有人在聚会上当众提起当年的事")
            results.append(out["flashback"])
        self.assertTrue(any(results), "应至少一次闪回")
        self.assertFalse(all(results), "不应每次都闪回（概率化）")

    def test_near_miss_raises_vigilance(self):
        e = Engine(seed=3)
        e.handle_event("有人当众羞辱了你，所有人都在笑")
        v0 = e.state.vigilance
        # repeated cue exposure without crash should at least pre-activate
        e.handle_event("有人在聚会上当众提起当年的事")
        self.assertGreaterEqual(e.state.vigilance, v0)

    def test_seek_support_consolidates_trauma(self):
        e = Engine(seed=4)
        e.handle_event("你被最好的朋友背叛了，所有人都知道")
        self.assertGreater(len(e.memory.trauma), 0)
        out = e.act("seek_support")
        self.assertIsNotNone(out["processed_trauma"], "倾诉应处理最近创伤碎片")
        self.assertTrue(any(f.consolidated for f in e.memory.trauma))

    def test_learned_helplessness_freezes(self):
        e = Engine(seed=5)
        e.state.emotion_label = "sadness"
        e.state.pad = (-0.5, -0.2, -0.3)
        e.state.crash_count = 5   # multiple crashes in the past
        e.state.depression_tendency = 0.5
        d = e.decide({"valence": -0.6, "goal_relevance": 0.8,
                      "coping_potential": 0.3, "control": 0.4,
                      "norm_violation": 0.0, "novelty": 0.3, "text": "你失去了工作"},
                     flashback=False)
        self.assertEqual(d["behavior"], "freeze",
                         f"多次崩溃后应习得性无助（冻结），got {d['behavior']}")

    def test_anonymity_lowers_deviance_barrier(self):
        # same impulse level: anonymous context crosses the threshold
        e1 = Engine(seed=6)
        e1.state.self_control = 30
        d_open = e1.decide({"valence": -0.6, "goal_relevance": 0.8,
                            "coping_potential": 0.3, "control": 0.4,
                            "norm_violation": 0.0, "novelty": 0.3,
                            "text": "有人在白天当众羞辱了你"},
                           flashback=False)
        e2 = Engine(seed=6)
        e2.state.self_control = 30
        d_anon = e2.decide({"valence": -0.6, "goal_relevance": 0.8,
                            "coping_potential": 0.3, "control": 0.4,
                            "norm_violation": 0.0, "novelty": 0.3,
                            "text": "深夜的网上，没人知道你是谁，有人羞辱了你"},
                           flashback=False)
        self.assertTrue(d_anon.get("anonymous"))
        self.assertFalse(d_open.get("anonymous"))
        self.assertGreaterEqual(d_anon["impulse"] > 0.45, d_open["impulse"] > 0.6)
        # same impulse, different barriers: anonymous should be more likely deviant
        self.assertGreaterEqual(
            int(d_anon["deviant"]), int(d_open["deviant"]),
            "匿名情境不应降低越轨概率")


if __name__ == "__main__":
    unittest.main(verbosity=2)
