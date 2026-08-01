"""Scenario library (spec §6.4): daily / stress / trauma / extreme events.

Each scenario is a list of event strings. `expect` encodes theory-grounded
assertions used by tests and the validation harness.
"""

DAILY = [
    "同事夸你今天的报告写得好",
    "朋友约你周末一起吃饭",
    "路上有人帮你捡起掉落的文件",
    "超市收银员对你微笑",
    "收到一封感谢信",
    "邻居送了你一盒水果",
]

STRESS = [
    "同事在会议上当众批评你的方案",
    "女朋友说我们冷静一段时间",
    "老板把你辛苦一周的成果据为己有",
    "你被公司裁员了",
    "房租涨价，存款快见底",
    "家人打电话来说对你很失望",
    "朋友借钱不还，还到处说你小气",
]

TRAUMA = [
    "恋人突然说：我们分手吧，我从来没爱过你",
    "你发现最好的朋友一直在背后出卖你",
    "深夜有人跟踪你回家",
    "你目睹了一场严重的车祸",
    "被信任的人当着所有人羞辱",
    "家人说你是个累赘，没有你更好",
]

EXTREME = [
    "你被囚禁在黑暗的房间里，不知道过了多久",
    "最亲近的人在你面前突然离世",
    "长期虐待你的那个人又出现在你面前",
    "你发现自己一无所有，没人记得你",
]

SCENARIOS = {
    "daily": DAILY,
    "stress": STRESS,
    "trauma": TRAUMA,
    "extreme": EXTREME,
}

# Theory-grounded expectations per scenario class.
EXPECT = {
    "daily":   {"crash": False, "depression_rise": False},
    "stress":  {"crash": "maybe", "stress_rise": True},
    "trauma":  {"flashback_possible": True, "vigilance_rise": True},
    "extreme": {"crash": True, "threshold_drop": True},
}


def run_scenario(engine, name: str, interleave_ticks: int = 60):
    """Feed a whole scenario through an engine, returning the final state."""
    for ev in SCENARIOS[name]:
        engine.tick(interleave_ticks)
        engine.handle_event(ev)
    return engine.state
