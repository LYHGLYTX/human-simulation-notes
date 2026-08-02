#!/usr/bin/env python3
"""Interactive CLI for the human engine (M1).

Usage:
    python -m human_engine.cli            # interactive
    echo "你被当众羞辱了" | python -m human_engine.cli --once
    python -m human_engine.cli --demo     # run a scripted scenario
"""
from __future__ import annotations

import sys

from .engine import Engine
from .llm import MockLLM


def print_state(e: Engine):
    print("  ", e.state.short_summary())
    print(f"    PAD={tuple(round(v, 2) for v in e.state.pad)} "
          f"mood={tuple(round(v, 2) for v in e.state.mood_pad)} "
          f"阈值={e.state.threshold:.1f} wear={e.state.wear:.0f} "
          f"内疚={e.state.guilt:.2f} 羞耻={e.state.shame:.2f}")


def handle(e: Engine, line: str) -> bool:
    line = line.strip()
    if not line:
        return True
    if line.startswith("/"):
        cmd = line[1:].split()[0]
        if cmd in ("q", "quit", "exit"):
            return False
        if cmd in ("state", "s"):
            print_state(e)
        elif cmd in ("mem", "m"):
            for it in e.memory.episodic[-8:]:
                print(f"    [{it.t:+.0f}s] {it.text} (v={it.valence:+.1f} "
                      f"a={it.arousal:.1f} imp={it.importance:.2f})")
            print(f"    创伤碎片 {len(e.memory.trauma)} 条:",
                  [f.trigger_words for f in e.memory.trauma])
        elif cmd in ("appr",):
            p = e.state.last_pipeline
            if p:
                print("    appraisal:", {k: round(v, 2) for k, v in p["appraisal"].items()})
        elif cmd == "sleep":
            r = e.sleep(8.0)
            print(f"  睡了 8 小时。精力={r['energy']:.0f} 资源={r['resources']:.0f} "
                  f"自控={r['self_control']:.0f} 睡眠债={r['sleep_debt']:.1f}h "
                  f"应激={r['stress']:.0f}"
                  + (f" 归档记忆{r['archived']}条" if r.get("archived") else ""))
            if r.get("insight"):
                print(f"  睡梦中浮现一个念头：{r['insight']}")
        elif cmd == "recover":
            from .stress import recover_from_crash
            if e.state.crashed:
                recover_from_crash(e.persona, e.state)
                print("  崩溃逐渐平息…（阈值永久下降了）")
            else:
                print("  当前未崩溃。")
        elif cmd == "help":
            print("  输入事件文本；命令：/state /mem /appr /sleep /recover /quit")
        return True

    e.tick(60)  # advance time a bit before handling
    out = e.handle_event(line)
    print(f"  [{out['event_type']}] {line}")
    print(f"  → {out['action']}")
    if out["flashback"]:
        print("  ⚠ 闪回：创伤记忆被强制触发！")
    print_state(e)
    return True


def main(argv: list[str] | None = None):
    argv = argv or sys.argv[1:]
    e = Engine(seed=7)
    print("=== human_engine M1 (MockLLM) ===")
    print(e.persona.summary_text())
    print("输入事件，或 /help 查看命令。")

    if "--demo" in argv:
        events = [
            "同事在会议上当众嘲笑你的方案，说你'根本不行'",
            "女朋友发消息说：我们分手吧，我受够了",
            "朋友借钱不还，还到处说你小气",
            "老板把你辛苦一周的成果据为己有",
            "陌生人在网上威胁要找到你",
            "你收到一封道歉信，说当年的事是他们错了",
        ]
        for ev in events:
            if not handle(e, ev):
                break
        return 0

    if "--once" in argv:
        line = next((a for a in argv if not a.startswith("--")), None)
        if line:
            handle(e, line)
        return 0

    try:
        while True:
            line = input("你 > ")
            if not handle(e, line):
                break
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    sys.exit(main())
