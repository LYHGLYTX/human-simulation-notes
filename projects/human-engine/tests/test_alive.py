"""Tests for the "alive" layer: voice expression, intensity cues,
semantic similarity, autopilot daily life."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.pop("HE_API_KEY", None)

import random
import unittest

from human_engine.engine import Engine
from human_engine.persona import Persona, default_persona
from human_engine import voice, appraisal
from human_engine.memory import similarity
from human_engine.llm import MockLLM


class TestVoice(unittest.TestCase):
    def _snap(self, **kw):
        s = {"emotion_label": "anger", "pad": (-0.4, 0.5, -0.1),
             "crashed": False, "stress": 40, "resources": 60,
             "self_control": 50, "guilt": 0.1, "shame": 0.1}
        s.update(kw)
        return s

    def test_generates_nonempty_and_styled(self):
        p = default_persona()
        t = voice.generate("confront", self._snap(), p, [], random.Random(1))
        self.assertTrue(t.strip())
        self.assertIn("（", t)          # inner monologue present

    def test_variety_across_seeds(self):
        p = default_persona()
        texts = {voice.generate("vent", self._snap(), p, [], random.Random(s))
                 for s in range(12)}
        self.assertGreater(len(texts), 3, "同行为不同 seed 应有多变体")

    def test_crashed_voice_is_broken(self):
        p = default_persona()
        t = voice.generate("confront", self._snap(crashed=True), p, [],
                           random.Random(2))
        self.assertTrue(any(k in t for k in
                            ["崩溃边缘", "声音发颤", "一片空白", "几乎说不出话",
                             "碎片", "拼不起来", "感觉不到", "别了"]))

    def test_memory_reference_appears(self):
        p = default_persona()
        mems = [{"text": "母亲在自己 8 岁时离家，之后再无联系",
                 "valence": -0.9}]
        found = False
        for s in range(30):
            t = voice.generate("freeze", self._snap(), p, mems,
                               random.Random(s))
            if "母亲" in t:
                found = True
                break
        self.assertTrue(found, "30 个 seed 内应至少一次引用记忆")

    def test_style_shaping(self):
        quiet = Persona(speech_style="quiet")
        chatty = Persona(speech_style="chatty")
        tq = voice.generate("neutral_act", self._snap(emotion_label="calm"),
                            quiet, [], random.Random(3))
        self.assertTrue(tq.strip())


class TestIntensity(unittest.TestCase):
    def test_cue_scales_intensity(self):
        e1 = appraisal.perceive("你被彻底羞辱了")
        e2 = appraisal.perceive("你被有点羞辱了")
        self.assertGreater(e1.intensity, e2.intensity)
        self.assertGreater(e1.intensity, 1.0, "彻底 应放大强度")
        self.assertLess(e2.intensity, 1.0, "有点 应减弱强度")

    def test_intensity_reaches_engine(self):
        mild = Engine(seed=1)
        strong = Engine(seed=1)
        mild.handle_event("你被批评了")
        strong.handle_event("你被彻底地、当众地批评了")
        self.assertGreater(strong.state.stress, mild.state.stress,
                           "强度线索应传导到应激")


class TestSimilarity(unittest.TestCase):
    def test_similar_texts_score_higher(self):
        a = "朋友借钱不还，还到处说你小气"
        b = "朋友借了钱一直不还，还到处说你小气"
        c = "今天天气很好，你出门买了杯咖啡"
        self.assertGreater(similarity(a, b), similarity(a, c),
                           "语义相近文本应得分更高")


class TestAutopilot(unittest.TestCase):
    def test_daily_drift_writes_memories(self):
        e = Engine(seed=7)
        e.autopilot_enabled = True
        for _ in range(60):
            e.tick(600)                  # 10h of autonomous life
        self.assertGreater(len(e.memory.episodic), 2,
                           "自主生活应产生日常事件记忆")

    def test_auto_sleep_recovers(self):
        e = Engine(seed=8)
        e.autopilot_enabled = True
        e.state.t = 23.0 * 3600          # 23:00
        e.state.sleep_debt = 6.0
        e.tick(3600)                     # 1h tick -> sleep trigger is certain
        self.assertLess(e.state.sleep_debt, 3.0,
                        "深夜+困倦应触发自动睡眠")


class TestMundaneChannel(unittest.TestCase):
    """Chitchat must not deplete the person (regression: it used to drive
    stress/resources/self-control down every turn — a relaxed chat is
    resource-maintaining, not depleting)."""

    def test_chitchat_does_not_deplete(self):
        from human_engine.llm import MockLLM
        e = Engine(seed=11, llm=MockLLM())
        s0 = (e.state.stress, e.state.resources, e.state.self_control)
        for msg in ["你好啊", "你在做什么呢", "今天天气不错"]:
            e.tick(60)
            e.handle_event(msg)
        s1 = (e.state.stress, e.state.resources, e.state.self_control)
        self.assertLessEqual(s1[0], s0[0] + 3, "闲聊不应显著抬升应激")
        self.assertGreaterEqual(s1[1], s0[1], "闲聊不应耗竭资源")

    def test_neutral_chat_has_no_negative_rpe(self):
        from human_engine.llm import MockLLM
        e = Engine(seed=13, llm=MockLLM())
        e.handle_event("随便聊聊")
        self.assertAlmostEqual(e.state.rpe, 0.0, places=3,
                               msg="中性事件不应产生负奖赏预测误差")

    def test_negative_events_still_hit(self):
        from human_engine.llm import MockLLM
        e = Engine(seed=12, llm=MockLLM())
        s0 = e.state.stress
        e.handle_event("你被当众羞辱了")
        self.assertGreater(e.state.stress, s0 + 10, "负面事件冲击必须保留")

    def test_chitchat_pressure_10x10_no_crash(self):
        """10 sessions x 10 turns of small talk: no crash, no stress creep
        (regression: keyword misfires like 打算->threat used to add ~+15)."""
        from human_engine.llm import MockLLM
        chitchat = ["你好啊", "你在做什么呢", "今天天气不错", "你吃饭了吗",
                    "最近忙不忙", "你喜欢听什么音乐", "周末有什么打算",
                    "你养宠物吗", "早点休息", "明天见"]
        for seed in range(10):
            e = Engine(seed=seed, llm=MockLLM())
            s0 = e.state.stress
            for msg in chitchat:
                e.tick(60)
                e.handle_event(msg)
            self.assertFalse(e.state.crashed, f"seed {seed} 闲聊不应崩溃")
            self.assertLess(e.state.stress, s0 + 8,
                            f"seed {seed} 闲聊应激漂移过大: {s0}->{e.state.stress}")
            self.assertGreaterEqual(e.state.resources, 80,
                                    f"seed {seed} 闲聊不应耗竭资源")


class TestIdentity(unittest.TestCase):
    def test_summary_has_identity(self):
        p = default_persona()
        s = p.summary_text()
        self.assertIn(p.name, s)
        self.assertIn("目标", s)
        self.assertIn("口头禅", s)

    def test_mockllm_uses_voice(self):
        e = Engine(seed=9)
        out = e.handle_event("同事在会议上当众羞辱你")
        self.assertTrue(out["action"].strip())
        self.assertNotEqual(out["action"], "（沉默）")


if __name__ == "__main__":
    unittest.main()
