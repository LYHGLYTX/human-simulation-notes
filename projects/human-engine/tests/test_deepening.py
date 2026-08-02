"""Tests for the deepening pass: reflection/archival/self-edit memory,
WASABI secondary emotions, FLAME delayed burst, belief layer, RPE learning,
prediction error, circadian rhythm, interview-based persona generation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# tests must never hit the real LLM: force MockLLM
os.environ.pop("HE_API_KEY", None)

import unittest

from human_engine.engine import Engine
from human_engine.persona import Persona
from human_engine.state import State
from human_engine import emotion, physiology, coping
from human_engine.memory import Memory, MAX_EPISODIC
from human_engine.persona_generation import interview, persona_from_answers


class TestMemoryDeepening(unittest.TestCase):
    def test_reflect_builds_self_narrative(self):
        e = Engine(seed=1)
        for ev in ["同事当众羞辱你", "朋友背叛了你", "家人说对你很失望"]:
            e.handle_event(ev)
        e.state.t = 86400.0 * 2
        insight = e.memory.reflect(e.state.t)
        self.assertIsNotNone(insight, "重复负面经历应提炼自我认知")
        self.assertIn("我", insight.text)
        self.assertGreaterEqual(len(e.memory.semantic), 1)

    def test_consolidate_all_archives_and_strengthens(self):
        m = Memory()
        for i in range(MAX_EPISODIC + 20):
            m.add_episodic(t=float(i), text=f"事件{i}", valence=-0.3,
                           arousal=0.3, importance=0.2 + (i % 3) * 0.1)
        archived = m.consolidate_all(t=100000.0)
        self.assertGreater(archived, 0, "记忆压力应触发归档")
        active1 = sum(1 for it in m.episodic if not it.archived)
        self.assertLessEqual(active1, MAX_EPISODIC)
        # repeated sleeps must NOT drain the active set below the cap
        m.consolidate_all(t=100000.0)
        m.consolidate_all(t=100000.0)
        active2 = sum(1 for it in m.episodic if not it.archived)
        self.assertEqual(active1, active2,
                         "归档应只处理超出上限的部分，活跃集保持稳定")
        self.assertTrue(all(it.consolidation == 1.0 for it in m.episodic),
                        "多次睡眠后所有记忆应巩固")

    def test_prune_duplicates_supersedes_old_same_theme(self):
        e = Engine(seed=2)
        old = e.memory.add_episodic(t=0.0, text="朋友背叛了我", valence=-0.8,
                                    arousal=0.7, importance=0.8)
        new = e.memory.add_episodic(t=10000.0, text="另一个朋友也背叛了我",
                                    valence=-0.8, arousal=0.7, importance=0.8)
        n = e.memory.prune_duplicates(new)
        self.assertGreater(n, 0)
        self.assertTrue(old.superseded)
        self.assertLess(old.importance, 0.5, "被取代的记忆重要性应下调")


class TestEmotionDeepening(unittest.TestCase):
    def test_wasabi_secondary_emotion_slows_decay(self):
        e = Engine(seed=3)
        a = {"valence": -0.8, "novelty": 0.5, "goal_relevance": 0.9,
             "coping_potential": 0.4, "norm_violation": 0.6, "control": 0.3}
        emotion.apply_appraisal(a, e.persona, e.state)
        self.assertLess(e.state.emotion_decay_mod, 0.9,
                        "高认知深度事件应减慢情绪弛豫（次情绪）")

    def test_flame_delayed_burst_banks_and_discharges(self):
        e = Engine(seed=4)
        e.state.self_control = 70.0   # able to suppress -> bank anger
        for ev in ["同事当众羞辱你", "老板抢了你的功劳"]:
            e.handle_event(ev)
        self.assertGreater(e.state.suppressed_anger, 0.2,
                           "可抑制的愤怒应被积压（延迟爆发）")
        e.act("vent")
        self.assertLess(e.state.suppressed_anger, 0.5, "发泄应释放积压的愤怒")

    def test_occ_extended_labels(self):
        self.assertEqual(emotion.label_from_pad((0.3, -0.3, 0.0)), "relief")
        self.assertEqual(emotion.label_from_pad((0.1, 0.4, 0.3)), "hopeful")
        self.assertEqual(emotion.label_from_pad((-0.2, 0.5, 0.2)), "contempt")
        # old regions unchanged
        self.assertEqual(emotion.label_from_pad((0.5, 0.5, 0.0)), "joy")
        self.assertEqual(emotion.label_from_pad((-0.5, 0.5, 0.0)), "anger")


class TestBeliefAndLearning(unittest.TestCase):
    def test_betrayal_strengthens_mistrust_belief(self):
        e = Engine(seed=5)
        before = e.state.beliefs["mistrust"]
        e.handle_event("你被最好的朋友背叛了")
        self.assertGreater(e.state.beliefs["mistrust"], before)
        self.assertGreater(e.state.beliefs["abandonment"], 0.3)

    def test_help_raises_support_belief(self):
        e = Engine(seed=6)
        before = e.state.beliefs["support_available"]
        e.handle_event("朋友在你困难时帮助了你，陪你到深夜")
        self.assertGreater(e.state.beliefs["support_available"], before)

    def test_rpe_adapts_to_environment(self):
        good = Engine(seed=7)
        for ev in ["老板表扬了你", "朋友送了你礼物", "你通过了考试"]:
            good.handle_event(ev)
        bad = Engine(seed=7)
        for ev in ["你被批评了", "你丢了钱包", "朋友疏远了你"]:
            bad.handle_event(ev)
        self.assertGreater(good.state.reward_expectation,
                           bad.state.reward_expectation,
                           "持续正反馈应提高奖赏期望（RPE 适应）")

    def test_prediction_error_surprise(self):
        e = Engine(seed=8)
        e.handle_event("同事在会议上嘲笑你")     # humiliation
        e.handle_event("老板突然宣布给你升职")   # surprise: success after bad
        self.assertGreater(e.state.prediction_error, 0.2,
                           "事件类型突变应产生预测误差（htm-lite）")


class TestCircadian(unittest.TestCase):
    def test_night_costs_more_energy(self):
        night = State(t=2.0 * 3600)      # 2:00
        day = State(t=14.0 * 3600)       # 14:00
        p = Persona()
        physiology.update(night, p, 3600.0)
        physiology.update(day, p, 3600.0)
        self.assertLess(night.energy, day.energy, "夜间消耗应更快")
        self.assertAlmostEqual(night.phase, 2.0, delta=0.5)
        self.assertAlmostEqual(day.phase, 14.0, delta=0.5)


class TestInterviewGeneration(unittest.TestCase):
    def test_default_interview_produces_persona(self):
        p = interview()
        self.assertGreaterEqual(len(p.history), 1, "访谈应生成人生经历")
        self.assertIn(p.attachment, ("safe", "preoccupied", "dismissing",
                                     "fearful"))
        e = Engine(persona=p, seed=9)   # must be runnable
        e.handle_event("你被当众羞辱了")
        self.assertGreater(e.state.stress, 0)

    def test_answer_direction_maps_traits(self):
        extro = persona_from_answers({**{f"q{i}": 0.5 for i in range(1, 18)},
                                      "q1": 0.95, "q2": 0.05}, seed=1)
        intro = persona_from_answers({**{f"q{i}": 0.5 for i in range(1, 18)},
                                      "q1": 0.05, "q2": 0.95}, seed=1)
        self.assertGreater(extro.extraversion, intro.extraversion)
        scarred = persona_from_answers({**{f"q{i}": 0.5 for i in range(1, 18)},
                                        "q11": 0.95, "q13": 0.95}, seed=2)
        self.assertGreater(scarred.schemas["abandonment"], 0.6)
        self.assertIn(scarred.attachment, ("fearful", "preoccupied"))


class TestCopingSelection(unittest.TestCase):
    def test_helplessness_forces_avoidance(self):
        s = State(t=0.0)
        s.crash_count = 5
        s.depression_tendency = 0.5
        style = coping.select_coping(
            {"control": 0.8, "coping_potential": 0.7}, Persona(), s)
        self.assertEqual(style, coping.AVOID,
                         "习得性无助者应放弃主动应对（回避）")

    def test_high_control_problem_focused(self):
        s = State(t=0.0)
        style = coping.select_coping(
            {"control": 0.8, "coping_potential": 0.7}, Persona(), s)
        self.assertEqual(style, coping.PROBLEM)


if __name__ == "__main__":
    unittest.main()
