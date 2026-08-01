# human-engine

模拟人类精神状态的**核心引擎**（M1 里程碑，MockLLM 离线版）。

设计依据：`../../notes/research/design/spec.md`（spec v0.1）
理论依据：`../../notes/research/<主题>/theories.md`（10 模块理论全集）

## 架构

```
human_engine/
├── persona.py     # 静态人格（大五/黑暗三合一/BIS-BAS/依恋/图式/人生年表）
├── state.py       # 动态状态（PAD 情绪/心境/应激/资源/自控/内疚羞耻/生理）
├── emotion.py     # ALMA 三层 + PAD 弛豫 + 情绪标签映射
├── memory.py      # 自传体记忆（遗忘曲线检索）+ 创伤碎片通道
├── stress.py      # 应激累积/崩溃触发/不可逆后果（敏化+磨损）
├── morality.py    # 冲动生成/规范对抗/道德脱离/内疚羞耻回路
├── appraisal.py   # 事件结构化 + Scherer 检查表（规则版）
├── llm.py         # LLMClient 协议 + MockLLM（可换真实 LLM）
├── engine.py      # tick 主循环 + 8 步事件管线 + 决策
└── cli.py         # 交互式 CLI
```

## 运行

```bash
cd projects/human-engine
python -m human_engine.cli --demo        # 脚本演示场景
python -m human_engine.cli               # 交互模式（输入事件，/help 看命令）
python -m human_engine.cli --once "你被当众羞辱了"
python -m unittest tests/test_engine.py  # 测试
```

## 交互示例

```
你 > 同事在会议上当众嘲笑你的方案，说你"根本不行"
  [humiliation] 同事在会议上当众嘲笑你的方案，说你"根本不行"
  → 你低下头，装作没听见，快步走开。（心里烧着一团火）
  [t=60s] 情绪=anger(P-0.32 A+0.47 D-0.11) 应激=38/阈值79 资源=62 自控=63 阶段=alert
```

## 当前能力（M1）

- ✅ 完整 8 步事件管线（感知→appraisal→情绪→应激→记忆→决策→行为→反馈）
- ✅ ALMA 三层情绪动力学（人格→心境→情绪，PAD 弛豫）
- ✅ 自传体记忆 + 遗忘曲线 + 心境一致性检索
- ✅ 创伤通道（感官碎片 + 触发词闪回）
- ✅ 崩溃触发 + 不可逆后果（阈值永久下降、异稳态磨损）
- ✅ 越轨回路（冲动→规范对抗→道德脱离→内疚/羞耻）
- ⏳ 真实 LLM 接入（OpenAI 兼容，M3）

## 下一步（见 spec §7）

- M2：创伤通道深化（闪回频率/处理/再巩固）
- M3：接真实 LLM（appraisal 与生成）
- M4/M5：游戏化 + 细致前端
