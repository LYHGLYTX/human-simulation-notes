"""Consistency tests (spec §6.1): determinism, stability, persona contrast,
timeline coherence, scenario expectations + psychometrics sanity."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

from human_engine.engine import Engine
from human_engine.persona import Persona
from human_engine.scenarios import SCENARIOS, EXPECT, run_scenario
from human_engine import psychometrics as psy


class TestDeterminism(unittest.TestCase):
    def test_same_seed_reproduces_exactly(self):
        a = Engine(seed=42)
        b = Engine(seed=42)
        for ev in SCENARIOS["trauma"]:
            a.handle_event(ev)
            b.handle_event(ev)
        self.assertEqual(a.state.snapshot(), b.state.snapshot())


class TestStability(unittest.TestCase):
    def test_behavior_distribution_stable(self):
        """Same event, 20 seeds: dominant behavior should be stable."""
        from collections import Counter
        kinds = []
        for seed in range(20):
            e = Engine(seed=seed)
            out = e.handle_event("同事在会议上当众羞辱你的方案")
            kinds.append(out["decision"]["behavior"])
        c = Counter(kinds)
        top, count = c.most_common(1)[0]
        self.assertGreaterEqual(count / 20, 0.6,
                                f"行为分布漂移过大: {dict(c)}")


class TestPersonaContrast(unittest.TestCase):
    def test_high_neuroticism_more_stress(self):
        low = Engine(persona=Persona(neuroticism=0.1, resilience=0.8), seed=1)
        high = Engine(persona=Persona(neuroticism=0.9, resilience=0.2), seed=1)
        for ev in SCENARIOS["stress"][:3]:
            low.handle_event(ev)
            high.handle_event(ev)
        self.assertGreater(high.state.stress, low.state.stress,
                           "高神经质应对相同事件产生更高应激")

    def test_high_bas_more_impulsive(self):
        low = Engine(persona=Persona(bas=0.2, neuroticism=0.4), seed=2)
        high = Engine(persona=Persona(bas=0.95, neuroticism=0.4), seed=2)
        low.state.self_control = 25
        high.state.self_control = 25
        low.state.resources = 30
        high.state.resources = 30
        dl = low.decide({"valence": -0.7, "goal_relevance": 0.8,
                         "coping_potential": 0.3, "control": 0.4,
                         "norm_violation": 0.1, "novelty": 0.4,
                         "text": "有人当众羞辱了你"}, flashback=False)
        dh = high.decide({"valence": -0.7, "goal_relevance": 0.8,
                          "coping_potential": 0.3, "control": 0.4,
                          "norm_violation": 0.1, "novelty": 0.4,
                          "text": "有人当众羞辱了你"}, flashback=False)
        self.assertGreaterEqual(dh["impulse"], dl["impulse"])


class TestTimeline(unittest.TestCase):
    def test_trauma_before_changes_reaction(self):
        """The same event lands differently before vs after trauma history."""
        naive = Engine(seed=3)   # no trauma history
        scarred = Engine(seed=3)
        scarred.state.vigilance = 0.7
        scarred.state.depression_tendency = 0.4
        for ev in ["你被最好的朋友背叛了"]:
            naive.handle_event(ev)
            scarred.handle_event(ev)
        self.assertGreater(scarred.state.stress, naive.state.stress,
                           "有创伤史者面对同样背叛应应激更高")


class TestScenarioExpectations(unittest.TestCase):
    def test_daily_no_crash(self):
        e = Engine(seed=4)
        run_scenario(e, "daily")
        self.assertFalse(e.state.crashed)

    def test_extreme_crashes_and_lowers_threshold(self):
        e = Engine(seed=5)
        t0 = e.state.threshold
        run_scenario(e, "extreme")
        self.assertTrue(e.state.crashed)
        self.assertLess(e.state.threshold, t0)

    def test_trauma_raises_vigilance(self):
        e = Engine(seed=6)
        run_scenario(e, "trauma")
        self.assertGreater(e.state.vigilance, 0.1,
                           "创伤场景后警觉度应明显上升")

    def test_stress_raises_stress(self):
        e = Engine(seed=7)
        s0 = e.state.stress
        run_scenario(e, "stress")
        self.assertGreater(e.state.stress, s0)


class TestPsychometrics(unittest.TestCase):
    def test_scales_respond_to_trauma(self):
        naive = Engine(seed=8)
        scarred = Engine(seed=8)
        run_scenario(scarred, "trauma")
        p_naive = psy.full_report(naive)
        p_scarred = psy.full_report(scarred)
        self.assertGreater(p_scarred["pcl5"]["score"], p_naive["pcl5"]["score"])
        self.assertGreaterEqual(p_scarred["stai_s"]["score"],
                                p_naive["stai_s"]["score"])

    def test_phq9_rises_after_repeated_crashes(self):
        from human_engine.stress import recover_from_crash
        e = Engine(seed=9)
        for _ in range(2):
            run_scenario(e, "extreme")
            if e.state.crashed:
                recover_from_crash(e.persona, e.state)
        r = psy.full_report(e)
        self.assertGreater(r["phq9"]["score"], 10, "多次崩溃后 PHQ-9 应显著升高")
        self.assertGreater(r["state"]["crash_count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
