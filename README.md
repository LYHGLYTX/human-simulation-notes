# Human Simulation Notes & Projects

> 模拟人类的个人笔记与项目合集 — a personal collection of notes, experiments, and projects on simulating human behavior, cognition, and interaction.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE)

## Overview

This repository is a living archive of my ongoing exploration into what it means to **simulate a human** — from behavioral modeling and cognitive architectures to conversational agents, persona design, and human–AI interaction.

It is not a finished framework. It is a workbench: half-formed ideas, structured notes, runnable experiments, and the occasional finished prototype, all organized so they can grow into something more.

## Contents

```
.
├── notes/          # 笔记：research/（按主题）+ periods/（按 年/月/日 的时期日志）
│   ├── research/   #   主题研究笔记（emotion, memory, trauma, ethics…）
│   └── periods/    #   时期日志（2026/08/01.md）
├── projects/       # 模拟程序与原型（simulation programs & agents）
├── experiments/    # 小实验与基准（quick experiments）
└── resources/      # 参考资料（papers, books, tools）
```

| Directory    | What lives here |
|--------------|-----------------|
| `notes/research/` | Topic-organized notes: theory, literature, design specs, architecture blueprints |
| `notes/periods/`  | Time-ordered journal: `年/月/日` (e.g. `2026/08/01.md`) — daily thoughts & progress |
| `projects/`  | Self-contained simulation programs with their own README |
| `experiments/` | Minimal, reproducible experiments — quick hypotheses tested in code |
| `resources/` | Curated papers, books, tools, and references |

## Goals

- **Understand**: How can we decompose human behavior into models that are both useful and honest about their limits?
- **Build**: Incremental prototypes that put those models to work — chatbots, simulations, agents.
- **Document**: Every step, so future-me (and you) can trace *why* something was built the way it was.

## Getting Started

This is a notes-and-code collection rather than a single application, so each subdirectory is self-contained:

```bash
# Explore the notes
open notes/

# Run a specific project — see its own README
cd projects/<project-name> && cat README.md
```

## Contributing & Feedback

This is primarily a personal archive, but questions, ideas, and constructive critique are welcome via GitHub Issues.

## License

Unless otherwise noted, content in this repository is shared under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — personal and non-commercial use with attribution. Code samples carry their own headers where a different license applies.
