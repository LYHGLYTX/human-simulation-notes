"""Tests for batch simulation (spec §5.3/§6.1): determinism and
theory-grounded statistics."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# batch simulation forces MockLLM internally; belt-and-braces anyway
os.environ.pop("HE_API_KEY", None)

import unittest

from human_engine.simulation import run_batch, compare, PERSONA_PRESETS
from human_engine.scenarios import SCENARIOS


class TestBatchSimulation(unittest.TestCase):
    def test_deterministic(self):
        r1 = run_batch(SCENARIOS["stress"], n=10, seed_base=7)
        r2 = run_batch(SCENARIOS["stress"], n=10, seed_base=7)
        self.assertEqual(r1.as_dict(), r2.as_dict(),
                         "同 seed_base 批量结果应完全一致")

    def test_extreme_always_crashes_daily_never(self):
        extreme = run_batch(SCENARIOS["extreme"], n=15, seed_base=1)
        daily = run_batch(SCENARIOS["daily"], n=15, seed_base=1)
        self.assertGreaterEqual(extreme.crash_rate, 0.8,
                                "极端场景应高崩溃率")
        self.assertEqual(daily.crash_rate, 0.0, "日常场景不应崩溃")
        self.assertGreater(extreme.threshold_drop, 0.0,
                           "崩溃后阈值应永久下降（应激敏化）")

    def test_behavior_distribution_recorded(self):
        r = run_batch(SCENARIOS["stress"], n=5, seed_base=2)
        self.assertGreater(sum(r.behavior_dist.values()), 0,
                           "行为分布不应为空")

    def test_resilient_crashes_less_than_fragile(self):
        res = run_batch(SCENARIOS["stress"], n=15, seed_base=3,
                        persona=PERSONA_PRESETS["resilient"]())
        frag = run_batch(SCENARIOS["stress"], n=15, seed_base=3,
                         persona=PERSONA_PRESETS["fragile"]())
        self.assertLess(res.crash_rate, frag.crash_rate,
                        "高韧性者面对同样压力崩溃率应更低（Bonanno）")

    def test_impulsive_deviates_more(self):
        imp = run_batch(SCENARIOS["stress"], n=15, seed_base=4,
                        persona=PERSONA_PRESETS["impulsive"]())
        default = run_batch(SCENARIOS["stress"], n=15, seed_base=4)
        self.assertGreater(imp.deviant_acts, default.deviant_acts,
                           "高 BAS 冲动者越轨行为应更多（GST 路径）")

    def test_compare_returns_contrast_report(self):
        results = compare({k: f() for k, f in PERSONA_PRESETS.items()},
                          SCENARIOS["stress"], n=5, seed_base=5)
        self.assertEqual(set(results), set(PERSONA_PRESETS))
        self.assertTrue(all(isinstance(r.crash_rate, float)
                            for r in results.values()))


if __name__ == "__main__":
    unittest.main()
