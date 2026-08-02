"""Voice: state-driven behavior expression — the "alive" text layer.

Replaces single-line templates with a small generative system:
behavior x emotion x speech style x mannerisms x memory references.
Offline (MockLLM) path gains human texture; state consistency rules keep
the text honest (crashed people don't speak in full calm sentences).

All choices use the shared seeded rng -> deterministic per seed.
"""
from __future__ import annotations

import random

# ---------------------------------------------------------------------------
# third-person action descriptions per behavior; {m} = mannerism slot
DESC: dict[str, list[str]] = {
    "confront": [
        "你盯着对方，一字一句地说：{m}你再说一遍。",
        "你深吸一口气，把话当面挑明，说完自己先愣住了。",
        "你第一次没有躲，声音不大，但每个字都很清楚。",
        "你站住了，回头，把憋了很久的话一次说完了。",
        "你拍了下桌子，又立刻后悔——但话已经收不回来了。",
    ],
    "avoid": [
        "你低下头，装作没听见，快步走开。",
        "你说“嗯，知道了”，然后转身离开。",
        "你笑了笑，把话题岔开，像什么都没发生。",
        "你借口有事，逃一样地离开了现场。",
        "你把手机翻了个面，假装在看别的东西。",
    ],
    "vent": [
        "你把门猛地关上，把桌上的东西摔在地上。",
        "你砸了手边的杯子，碎片溅了一地，然后蹲下来发呆。",
        "你把枕头砸在墙上，又捡起来，抱在怀里。",
        "你一个人吼了几句，声音在空房间里弹回来。",
    ],
    "seek_support": [
        "你拿起手机，犹豫很久，给唯一信任的人发了条消息。",
        "你拨出一个号码，响了两声又挂断，最后发了条语音。",
        "你终于开口，把这件事说给了那个人听，说到一半就说不下去了。",
        "你写了一段很长的字，删删改改，最后只发了三个字：在吗。",
    ],
    "impulsive_attack": [
        "你冲上去推了他一把，脑子里只有一个念头：让他闭嘴。",
        "你一把抓住他的衣领，声音嘶哑：{m}别说了！",
        "你随手抄起东西砸过去，砸没砸中自己也不知道。",
        "你狠狠踹了一脚旁边的椅子，指着他说：够了！",
    ],
    "impulsive_selfharm": [
        "你把自己关进房间，不想让任何人看见。（模拟研究用途：如你在现实中有类似困扰，请寻求专业帮助）",
        "你用力攥紧拳头，直到疼痛盖过脑子里那个声音。（模拟研究用途，非诊断）",
        "你盯着镜子里的自己，很久，然后移开了视线。（模拟研究用途）",
    ],
    "freeze": [
        "你站在原地，脑子一片空白，什么也说不出来。",
        "你张了张嘴，却没有发出声音，整个人僵住了。",
        "你感觉世界突然安静下来，只剩下自己的心跳。",
        "你一动不动，像被按了暂停键。",
    ],
    "fight": [
        "你爆发了，砸了手边的东西，吼出压抑很久的话。",
        "积攒的一切在那一刻全涌了出来，你摔了门，又摔了椅子。",
        "你像换了一个人，声音大得连自己都不认识。",
    ],
    "dissociate": [
        "世界突然变得很远，你看着自己像在看别人。",
        "你听见自己在说话，但感觉那声音不是你的。",
        "周围的一切像隔了一层水，你在这层水的外面。",
    ],
    "neutral_act": [
        "你平静地处理了这件事，像处理一件普通的小事。",
        "你点点头，把事情记下来，继续做手头的事。",
        "你没什么特别的反应，只是把这件事放在心里。",
        "你照常做完了今天该做的事。",
    ],
}

# ---------------------------------------------------------------------------
# first-person inner monologue pools per emotion label
INNER: dict[str, list[str]] = {
    "anger": ["心里烧着一团火", "牙根咬得发紧", "血往头顶涌"],
    "fear": ["腿在发抖", "脑子里只有“逃”一个字", "心提到了嗓子眼"],
    "anxious": ["心里七上八下", "手指不自觉地绞在一起", "总觉得要出事"],
    "sadness": ["胸口闷得喘不过气", "鼻子一酸", "喉咙像被什么堵住了"],
    "shame": ["脸上发烫，想找个地缝钻进去", "不敢看任何人的眼睛", "觉得自己像个笑话"],
    "distress": ["头嗡嗡地响", "什么都抓不住的感觉", "想喊又喊不出来"],
    "joy": ["嘴角忍不住往上翘", "心里亮了一下", "想跟谁说点什么"],
    "calm": ["心里没什么波澜", "习惯了", "就这样吧"],
    "relief": ["绷着的那根弦松了下来", "长长地呼出一口气"],
    "hopeful": ["好像还有一点指望", "想再试一次"],
    "contempt": ["懒得看他一眼", "心里冷笑了一声"],
    "neutral": ["说不上什么感觉", "只是把这件事记住了"],
}

# speech-style sentence shapers
STYLE_SHAPER = {
    "quiet":  {"suffix": ["……", "（声音很轻）", "（说完就低下了头）"], "gap": "。"},
    "blunt":  {"suffix": ["。", "。就这？", "。算了。"], "gap": "。"},
    "chatty": {"suffix": ["。", "，说实话", "，你知道吗，"], "gap": "，"},
    "poetic": {"suffix": ["。", "，像一片叶子落进水里。", "。风很冷。"], "gap": "。"},
}

# crashed / extreme-state fragments (short, broken)
CRASHED_PREFIX = ["（崩溃边缘）", "（声音发颤）", "（一片空白）", "（几乎说不出话）"]

MEMORY_REF = [
    "眼前忽然闪过：{mem}",
    "脑子里不受控制地想起：{mem}",
    "那一瞬间，{mem}的事又涌了上来",
]


def generate(behavior: str, snapshot: dict, persona, memories: list,
             rng: random.Random | None = None) -> str:
    """Produce a behavior utterance consistent with the current state."""
    rng = rng or random
    label = snapshot.get("emotion_label", "neutral")
    pad = snapshot.get("pad", (0.0, 0.0, 0.0))
    crashed = snapshot.get("crashed", False)
    style = getattr(persona, "speech_style", "quiet")

    # --- description ----------------------------------------------------
    pool = DESC.get(behavior, DESC["neutral_act"])
    desc = rng.choice(pool)

    # mannerism slot or append
    mannerisms = getattr(persona, "mannerisms", None) or []
    if "{m}" in desc:
        desc = desc.replace("{m}", rng.choice(mannerisms) if mannerisms else " ")
    elif mannerisms and rng.random() < 0.4:
        desc += rng.choice(mannerisms)

    # --- inner monologue -------------------------------------------------
    inner_pool = INNER.get(label, INNER["neutral"])
    inner = rng.choice(inner_pool)
    if crashed:
        inner = rng.choice([
            "脑子里全是碎片，拼不起来",
            "什么都感觉不到了",
            "只剩下一个念头：别了",
        ])

    # --- style shaping ---------------------------------------------------
    shaper = STYLE_SHAPER.get(style, STYLE_SHAPER["quiet"])
    if rng.random() < 0.5:
        desc += rng.choice(shaper["suffix"])
    if crashed:
        desc = rng.choice(CRASHED_PREFIX) + desc
    elif pad[1] > 0.4:                      # high arousal: raw, breathless
        desc = desc.rstrip("。") + "！"
    elif pad[1] < -0.2 and style == "quiet":
        desc = desc.rstrip("。") + "……"

    text = f"{desc}（{inner}）"

    # --- memory reference -------------------------------------------------
    if memories and rng.random() < 0.35:
        mem = rng.choice(memories)
        mtext = str(mem.get("text", "")).strip()
        if mtext and len(mtext) <= 40:
            text += rng.choice(MEMORY_REF).format(mem=mtext)
    return text
