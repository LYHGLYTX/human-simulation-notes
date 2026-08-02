"""Tests for LLM clients: JSON parsing, fallback, auto-select (no network)."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from unittest import mock

from human_engine.llm import OpenAICompatLLM, MockLLM, make_llm, Action
from human_engine.persona import default_persona
from human_engine.state import State
from human_engine import appraisal as A


class FakeResponse:
    def __init__(self, content):
        self._content = content
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def read(self):
        body = json.dumps({"choices": [{"message": {"content": self._content}}]},
                          ensure_ascii=False)
        return body.encode()


class TestOpenAICompatLLM(unittest.TestCase):
    def setUp(self):
        self.llm = OpenAICompatLLM(api_key="test-key", base_url="https://x.test/v1",
                                   model="test-model", proxy="")

    @mock.patch("urllib.request.build_opener")
    def test_perceive_parses_event(self, mock_builder):
        mock_builder.return_value.open.return_value = FakeResponse(
            '{"type": "humiliation", "intensity": 0.9, "subject": "同事", "object": "我"}')
        ev = self.llm.perceive("同事当众羞辱我", default_persona())
        self.assertEqual(ev.type, "humiliation")
        self.assertAlmostEqual(ev.intensity, 0.9)

    @mock.patch("urllib.request.build_opener")
    def test_perceive_unknown_type_falls_back(self, mock_builder):
        mock_builder.return_value.open.return_value = FakeResponse(
            '{"type": "aliens", "intensity": 1}')
        ev = self.llm.perceive("外星人来了", default_persona())
        self.assertEqual(ev.type, "neutral")

    @mock.patch("urllib.request.build_opener")
    def test_perceive_llm_neutral_wins_over_keyword_misfire(self, mock_builder):
        """LLM says neutral; rule says threat via 打 inside 打算 — the LLM's
        semantic judgment must win (no strong negative evidence)."""
        mock_builder.return_value.open.return_value = FakeResponse(
            '{"type": "neutral", "intensity": 0.3}')
        ev = self.llm.perceive("周末有什么打算", default_persona())
        self.assertEqual(ev.type, "neutral",
                         "关键词误伤不得覆盖 LLM 语义判断")

    @mock.patch("urllib.request.build_opener")
    def test_perceive_anti_whitewash_still_works(self, mock_builder):
        """LLM whitewashes a severe event to neutral — strong evidence
        (威胁/羞辱...) overrides it (anti-whitewash intent preserved)."""
        mock_builder.return_value.open.return_value = FakeResponse(
            '{"type": "neutral", "intensity": 1}')
        ev = self.llm.perceive("陌生人在网上威胁要找到你", default_persona())
        self.assertEqual(ev.type, "threat")

    def test_strong_negative_gate(self):
        from human_engine import appraisal as A
        self.assertFalse(A._strong_negative("周末有什么打算"))
        self.assertFalse(A._strong_negative("我先走了"))
        self.assertFalse(A._strong_negative("笑死我了"))
        self.assertTrue(A._strong_negative("有人威胁要打你"))
        self.assertTrue(A._strong_negative("同事当众羞辱我"))
        self.assertTrue(A._strong_negative("他妈的废物"))

    @mock.patch("urllib.request.build_opener")
    def test_appraise_parses_and_clamps(self, mock_builder):
        mock_builder.return_value.open.return_value = FakeResponse(
            '{"novelty": 0.8, "valence": 0.1, "goal_relevance": 0.9, '
            '"coping_potential": 0.2, "norm_violation": 0.0, "control": 0.3}')
        ev = A.Event(type="loss", text="你失去了工作", intensity=1.0)
        appr = self.llm.appraise(ev, default_persona(), State())
        # hybrid: LLM 0.1 vs rule -0.75 triggers the anti-whitewash branch
        # (w_rule=0.75): valence = 0.1*0.25 + (-0.75)*0.75
        self.assertAlmostEqual(appr.novelty, 0.8 * 0.4 + 0.6 * 0.6, delta=0.06)
        self.assertAlmostEqual(appr.valence, 0.1 * 0.25 + (-0.75) * 0.75)

    @mock.patch("urllib.request.build_opener")
    def test_appraise_fallback_on_error(self, mock_builder):
        mock_builder.return_value.open.side_effect = RuntimeError("api down")
        ev = A.Event(type="betrayal", text="朋友背叛了你", intensity=1.0)
        appr = self.llm.appraise(ev, default_persona(), State())
        self.assertLess(appr.valence, 0)   # rule-based fallback still works

    @mock.patch("urllib.request.build_opener")
    def test_generate_with_prose_wrapped_json(self, mock_builder):
        mock_builder.return_value.open.return_value = FakeResponse(
            '好的：{"text": "我...我说不出话。", "tone": "颤抖"}')
        snap = State().snapshot()
        snap["persona_summary"] = "测试人格"
        act = self.llm.generate(snap, [], {"behavior": "freeze"})
        self.assertIsInstance(act, Action)
        self.assertEqual(act.text, "我...我说不出话。")

    def test_make_llm_selects_mock_by_default(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIsInstance(make_llm(), MockLLM)

    def test_make_llm_selects_openai_with_key(self):
        with mock.patch.dict("os.environ", {"HE_API_KEY": "k"}, clear=True):
            self.assertIsInstance(make_llm(), OpenAICompatLLM)

    def test_thinking_param_logic(self):
        # DeepSeek: auto -> thinking disabled sent
        ds = OpenAICompatLLM(api_key="k", base_url="https://api.deepseek.com",
                             model="m", proxy="")
        self.assertEqual(ds.thinking, "disabled")
        # unknown endpoint: param omitted entirely
        other = OpenAICompatLLM(api_key="k", base_url="https://api.x.com/v1",
                                model="m", proxy="")
        self.assertIsNone(other.thinking)
        # forced by env
        with mock.patch.dict("os.environ", {"HE_THINKING": "enabled"}, clear=True):
            forced = OpenAICompatLLM(api_key="k",
                                     base_url="https://api.deepseek.com",
                                     model="m", proxy="")
        self.assertEqual(forced.thinking, "enabled")

    @mock.patch("urllib.request.build_opener")
    def test_chat_payload_includes_thinking(self, mock_builder):
        captured = {}
        def fake_open(req, timeout=90):
            captured["body"] = json.loads(req.data)
            return FakeResponse('{"ok": true}')
        mock_builder.return_value.open.side_effect = fake_open
        llm = OpenAICompatLLM(api_key="k", base_url="https://api.deepseek.com",
                              model="m", proxy="")
        llm._chat("sys", "user")
        self.assertEqual(captured["body"]["thinking"], {"type": "disabled"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
