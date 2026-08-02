"""Tests for social relations (spec §2.6) and physiology (spec §2.8)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# tests must never hit the real LLM: force MockLLM
os.environ.pop("HE_API_KEY", None)

import unittest

from human_engine.engine import Engine
from human_engine.persona import Persona
from human_engine.state import State
from human_engine import physiology
from human_engine.relations import Relations


class TestRelations(unittest.TestCase):
    def test_betrayal_crushes_trust_and_activates_attachment(self):
        e = Engine(seed=1)
        r = Relations()
        rel = r.get_or_create("朋友", "friend", 0.0)
        rel.affinity, rel.trust = 0.8, 0.8
        ch = r.apply_event("最好的朋友背叛了你，把你的秘密说出去",
                           "betrayal", -0.85, e.persona, e.state, 0.0)
        self.assertIsNotNone(ch)
        self.assertLess(rel.trust, 0.55, "背叛应重创信任")
        self.assertLess(rel.affinity, 0.75, "背叛应降低好感")
        self.assertGreater(rel.attachment_activation, 0.2, "背叛应激活依恋系统")

    def test_support_event_raises_affinity_and_trust(self):
        e = Engine(seed=2)
        r = Relations()
        rel = r.get_or_create("朋友", "friend", 0.0)
        rel.affinity, rel.trust = 0.5, 0.5
        r.apply_event("朋友在你困难时伸出了援手，陪你到深夜",
                      "help", 0.55, e.persona, e.state, 0.0)
        self.assertGreater(rel.affinity, 0.6)
        self.assertGreater(rel.trust, 0.6)

    def test_attachment_style_modulates_activation(self):
        e = Engine(seed=3)
        for style, lo, hi in [("preoccupied", 0.45, 0.75), ("safe", 0.25, 0.45),
                              ("dismissing", 0.05, 0.2)]:
            p = Persona(attachment=style)
            r = Relations()
            rel = r.get_or_create("伴侣", "partner", 0.0)
            r.apply_event("伴侣说我们分手吧", "abandonment", -0.9, p, e.state, 0.0)
            self.assertGreater(rel.attachment_activation, lo,
                               f"{style} 依恋激活应 > {lo}")
            self.assertLess(rel.attachment_activation, hi,
                            f"{style} 依恋激活应 < {hi}")

    def test_authority_humiliation_raises_power_diff_confront_lowers(self):
        e = Engine(seed=4)
        r = Relations()
        rel = r.get_or_create("老板", "authority", 0.35)
        r.apply_event("老板在会议上当众羞辱你", "humiliation", -0.8,
                      e.persona, e.state, 0.0)
        self.assertGreater(rel.power_diff, 0.4, "权威羞辱应放大权力差")
        r.apply_behavior("confront")
        self.assertLess(rel.power_diff, 0.4, "正面对抗应缩小权力差")

    def test_support_score_grows_with_relations(self):
        r = Relations()
        self.assertAlmostEqual(r.support_score(), 0.0)
        r.get_or_create("朋友", "friend", 0.0)
        r.people["朋友"].affinity, r.people["朋友"].trust = 0.9, 0.9
        self.assertGreater(r.support_score(), 0.2)
        r.people["朋友"].affinity, r.people["朋友"].trust = 0.2, 0.2
        self.assertLess(r.support_score(), 0.1, "坏关系不构成支持")

    def test_engine_pipeline_records_relation_change(self):
        e = Engine(seed=5)
        out = e.handle_event("同事在会议上当众嘲笑你的方案")
        self.assertIsNotNone(out["relation_change"])
        self.assertEqual(out["relation_change"]["name"], "同事")
        rel = e.relations.people["同事"]
        self.assertLess(rel.trust, 0.5)

    def test_seek_support_payoff_scales_with_support(self):
        e_weak = Engine(seed=6)
        e_strong = Engine(seed=6)
        # build strong relations in e_strong
        for ev in ["朋友在你困难时伸出援手",
                   "家人一直支持你",
                   "伴侣说会一直陪着你"]:
            e_strong.handle_event(ev)
        res_w = e_weak.state.resources
        res_s = e_strong.state.resources
        w = e_weak.act("seek_support")
        s = e_strong.act("seek_support")
        gain_w = e_weak.state.resources - res_w
        gain_s = e_strong.state.resources - res_s
        self.assertGreater(gain_s, gain_w, "社会支持高时 seek_support 收益应更大")


class TestPhysiology(unittest.TestCase):
    def test_wake_accumulates_sleep_debt_and_drains_energy(self):
        s = State(t=0.0)
        p = Persona()
        e0, d0 = s.energy, s.sleep_debt
        physiology.update(s, p, 3600.0 * 24)   # one full wake day
        self.assertLess(s.energy, e0, "清醒应消耗精力")
        self.assertGreater(s.sleep_debt, d0, "清醒应累积睡眠债")
        self.assertLessEqual(s.sleep_debt, 8.0, "一天最多欠 8h（sleep_need）")

    def test_sleep_session_recovers(self):
        e = Engine(seed=7)
        s, p = e.state, e.persona
        physiology.update(s, p, 3600.0 * 20)   # 20h awake -> ~6.7h debt
        self.assertGreater(s.sleep_debt, 6.0)
        s.resources = 30.0
        s.self_control = 30.0
        s.stress = 60.0
        r = physiology.sleep_session(s, p, 8.0)
        self.assertAlmostEqual(s.sleep_debt, 0.0, delta=0.5, msg="8h 应还清 6.7h 债")
        self.assertGreater(s.energy, 90.0, "睡眠应回满精力")
        self.assertGreater(s.resources, 60.0, "睡眠应大幅恢复资源（最大恢复源）")
        self.assertGreater(s.self_control, 50.0)
        self.assertLess(s.stress, 50.0, "睡眠应缓解应激")
        self.assertIn("hours", r)

    def test_fatigue_slows_recovery(self):
        e = Engine(seed=8)
        rested = Engine(seed=8)
        e.state.sleep_debt = 8.0   # exhausted
        rested.state.sleep_debt = 0.0
        sc0 = e.state.self_control = 20.0
        rested.state.self_control = 20.0
        for _ in range(60):
            e.tick(60)
            rested.tick(60)
        self.assertLess(e.state.self_control, rested.state.self_control,
                        "睡眠债高 → 自控恢复慢（具身认知）")
        self.assertLess(e.state.self_control, 100, "疲劳者 1h 内不应回满自控")

    def test_engine_tick_sleeping_flag(self):
        e = Engine(seed=9)
        e.state.sleep_debt = 5.0
        e.state.resources = 30.0
        for _ in range(60):
            e.tick(60, sleeping=True)   # 1h of sleep
        self.assertLess(e.state.sleep_debt, 5.0)
        self.assertGreater(e.state.resources, 30.0)


class TestDeterminism(unittest.TestCase):
    def test_relations_evolve_deterministically(self):
        events = ["同事在会议上嘲笑你",
                  "朋友借钱不还",
                  "老板把功劳据为己有",
                  "伴侣说我们分手吧"]
        snapshots = []
        for seed in (10, 10):
            e = Engine(seed=seed)
            for ev in events:
                e.handle_event(ev)
            snapshots.append(e.relations.snapshot())
        self.assertEqual(snapshots[0], snapshots[1], "同 seed 关系演化应完全一致")


if __name__ == "__main__":
    unittest.main()
