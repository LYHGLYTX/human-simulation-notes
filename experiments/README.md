# experiments/

小实验与基准 — small runnable experiments & benchmarks.

用于快速验证假设的一次性/半成品实验：某个情绪模型参数的测试、某个记忆检索策略的对比、prompt 效果的 A/B 等。

每个实验一个目录，包含：实验说明（假设、方法）、代码、结果记录。

结构建议：

```
experiments/
└── <experiment-name>/
    ├── README.md     # 假设、方法、结论
    └── run.py        # 可复现的代码
```

实验成熟后可以晋升到 `projects/`。
