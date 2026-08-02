"""Batch simulation (spec §5.3 / §6.1): run event sequences N times and
aggregate statistics — crash rate, recovery, behavior distribution, scales.

Deterministic: run i uses seed (seed_base + i); same inputs -> same outputs.
Produces the contrast material needed for M6 expert evaluation (same events x
different personas -> anonymized logs).

CLI:
    python -m human_engine.simulation --scenario extreme --n 50
    python -m human_engine.simulation --scenario stress --persona high_resilience
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from statistics import mean, stdev

from .engine import Engine
from .persona import Persona, default_persona
from .scenarios import SCENARIOS, run_scenario
from .llm import MockLLM
from . import psychometrics as psy


@dataclass
class BatchResult:
    n: int
    crash_rate: float               # runs that crashed at least once
    final_crashed_rate: float       # still crashed at the end (unrecovered)
    recovered_rate: float           # crashed but recovered by the end
    crash_at: list                  # event index (1-based) of first crash
    behavior_dist: dict             # behavior -> count (all decisions)
    flashbacks: int                 # total flashback events across runs
    deviant_acts: int               # total deviant behaviors
    threshold_drop: float           # mean threshold drop among crash runs
    final: dict                     # mean ± std of key states
    scales: dict                    # PCL-5 / PHQ-9 mean ± std

    def as_dict(self) -> dict:
        return asdict(self)


def _stats(values: list) -> str:
    if not values:
        return "0"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{mean(values):.1f}±{stdev(values):.1f}"


def run_batch(events: list[str], n: int = 20, persona: Persona | None = None,
              seed_base: int = 0, interleave_ticks: int = 60,
              llm=None) -> BatchResult:
    """Run the same event sequence n times (one engine per run), aggregate.

    Defaults to MockLLM: batch statistics measure the deterministic
    dynamics, not LLM expression quality. Pass llm=... to use a real model.
    """
    persona = persona or default_persona()
    # per-run MockLLM keeps each run's random stream independent (a shared
    # instance would chain runs through engine.llm.rng binding)
    per_run_mock = llm is None
    crashed_at: list[int] = []
    behaviors: Counter = Counter()
    flashbacks = 0
    deviant = 0
    n_crashed = 0
    n_final_crashed = 0
    drops: list[float] = []
    finals: dict[str, list] = {"stress": [], "resources": [],
                               "depression": [], "vigilance": [],
                               "threshold": [], "crash_count": []}
    scales: dict[str, list] = {"pcl5": [], "phq9": [], "stai_s": []}

    for i in range(n):
        # fresh MockLLM per run (seed isolation); Engine(llm=None) would
        # auto-select a real LLM via .env, which batch must never do
        e = Engine(persona=persona, seed=seed_base + i,
                   llm=MockLLM() if per_run_mock else llm)
        initial_threshold = e.state.threshold
        run_crashed = False
        for j, ev in enumerate(events, start=1):
            e.tick(interleave_ticks)
            out = e.handle_event(ev)
            decision = out["decision"]
            behaviors[decision.get("behavior", "neutral_act")] += 1
            if decision.get("deviant"):
                deviant += 1
            if out["flashback"]:
                flashbacks += 1
            if not run_crashed and e.state.crashed:
                run_crashed = True
                crashed_at.append(j)
                drops.append(round(initial_threshold - e.state.threshold, 1))
        if run_crashed:
            n_crashed += 1
        if e.state.crashed:
            n_final_crashed += 1
        r = psy.full_report(e)
        finals["stress"].append(r["state"]["stress"])
        finals["resources"].append(r["state"]["resources"])
        finals["depression"].append(r["state"]["depression_tendency"])
        finals["vigilance"].append(r["state"]["vigilance"])
        finals["threshold"].append(r["state"]["threshold"])
        finals["crash_count"].append(r["state"]["crash_count"])
        scales["pcl5"].append(r["pcl5"]["score"])
        scales["phq9"].append(r["phq9"]["score"])
        scales["stai_s"].append(r["stai_s"]["score"])

    return BatchResult(        n=n,
        crash_rate=round(n_crashed / n, 3),
        final_crashed_rate=round(n_final_crashed / n, 3),
        recovered_rate=round(max(0, n_crashed - n_final_crashed) / n, 3),
        crash_at=crashed_at,
        behavior_dist=dict(behaviors.most_common()),
        flashbacks=flashbacks,
        deviant_acts=deviant,
        threshold_drop=round(mean(drops), 1) if drops else 0.0,
        final={k: _stats(v) for k, v in finals.items()},
        scales={k: _stats(v) for k, v in scales.items()},
    )


def compare(personas: dict[str, Persona], events: list[str], n: int = 20,
            seed_base: int = 0) -> dict[str, BatchResult]:
    """Same events x different personas -> contrast report (M6 material)."""
    return {name: run_batch(events, n=n, persona=p, seed_base=seed_base)
            for name, p in personas.items()}


# --- CLI ---------------------------------------------------------------
PERSONA_PRESETS = {
    "default": default_persona,
    "resilient": lambda: Persona(neuroticism=0.35, resilience=0.85,
                                 bis=0.45, bas=0.5, attachment="safe",
                                 schemas={k: 0.15 for k in
                                          ("abandonment", "mistrust",
                                           "defectiveness", "failure")}),
    "fragile": lambda: Persona(neuroticism=0.9, resilience=0.1, bis=0.8,
                               bas=0.7, attachment="fearful",
                               schemas={"abandonment": 0.8, "mistrust": 0.75,
                                        "defectiveness": 0.6, "failure": 0.7}),
    "impulsive": lambda: Persona(neuroticism=0.4, resilience=0.5, bis=0.2,
                                 bas=0.95, attachment="dismissing"),
}


def _report(name: str, r: BatchResult) -> str:
    return (
        f"[{name}] n={r.n} 崩溃率={r.crash_rate:.0%} "
        f"结束时崩溃={r.final_crashed_rate:.0%} 恢复率={r.recovered_rate:.0%}\n"
        f"    首崩时点={r.crash_at[:10]}{'…' if len(r.crash_at) > 10 else ''} "
        f"平均阈值降={r.threshold_drop}\n"
        f"    行为分布={dict(sorted(r.behavior_dist.items(),
                                    key=lambda x: -x[1])[:5])}\n"
        f"    闪回={r.flashbacks} 越轨={r.deviant_acts}\n"
        f"    终态 stress={r.final['stress']} resources={r.final['resources']} "
        f"抑郁={r.final['depression']}\n"
        f"    量表 PCL-5={r.scales['pcl5']} PHQ-9={r.scales['phq9']} "
        f"STAI={r.scales['stai_s']}"
    )


def main(argv: list[str] | None = None):
    argv = argv or sys.argv[1:]
    scenario = "extreme"
    n = 20
    persona_name = "default"
    seed = 0
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--scenario" and i + 1 < len(argv):
            scenario = argv[i + 1]; i += 2
        elif a == "--n" and i + 1 < len(argv):
            n = int(argv[i + 1]); i += 2
        elif a == "--persona" and i + 1 < len(argv):
            persona_name = argv[i + 1]; i += 2
        elif a == "--seed" and i + 1 < len(argv):
            seed = int(argv[i + 1]); i += 2
        elif a == "--compare":
            # run all presets and compare
            events = SCENARIOS.get(scenario, SCENARIOS["extreme"])
            print(f"=== 批量模拟对比: 场景[{scenario}] × {n} runs × "
                  f"{len(PERSONA_PRESETS)} 人格 ===\n")
            results = compare({k: f() for k, f in PERSONA_PRESETS.items()},
                              events, n=n, seed_base=seed)
            for name, r in results.items():
                print(_report(name, r))
                print()
            return 0
        else:
            i += 1

    events = SCENARIOS.get(scenario, SCENARIOS["extreme"])
    persona = PERSONA_PRESETS.get(persona_name, default_persona)()
    print(f"=== 批量模拟: 场景[{scenario}] ({len(events)} 事件) × {n} runs "
          f"× 人格[{persona_name}] seed_base={seed} ===")
    r = run_batch(events, n=n, persona=persona, seed_base=seed)
    print(_report(persona_name, r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
