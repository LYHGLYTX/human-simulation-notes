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
python -m human_engine.web               # 🎮 游戏化 Web UI → http://localhost:8000
python -m unittest tests/test_engine.py  # 测试
```

## 🎮 游戏化 Web UI（阶段 2，零依赖）

浏览器打开 http://localhost:8000：

- 左栏：角色状态——表情、情绪标签、压力/精力/资源/自控/冲动/内疚/羞耻状态条、PAD 三轴、心境基线、GAS 阶段
- 右栏：给 TA 发事件 → TA 自动反应（气泡）→ **你选择 TA 的应对方式**（3–4 个选项，金色=引擎推荐；红色=越轨选项）
- 选择后果即时生效：越轨可能带来内疚/羞耻，也可能习得性道德脱离；压力爆表 → 💥 崩溃横幅（阈值永久下降）
- 支持：😴 睡觉（恢复资源）、↺ 重置、状态条 3s 自动刷新

## 交互示例

```
你 > 同事在会议上当众嘲笑你的方案，说你"根本不行"
  [humiliation] 同事在会议上当众嘲笑你的方案，说你"根本不行"
  → 你低下头，装作没听见，快步走开。（心里烧着一团火）
  [t=60s] 情绪=anger(P-0.32 A+0.47 D-0.11) 应激=38/阈值79 资源=62 自控=63 阶段=alert
```

## 🔌 接入真实 LLM（M3，OpenAI 兼容）

无需改代码——引擎自动检测环境变量 `HE_API_KEY`：

```bash
export HE_API_KEY=sk-xxxx                 # 必填
export HE_BASE_URL=https://api.openai.com/v1   # 或 DeepSeek / 硅基流动 / vLLM 等
export HE_MODEL=gpt-4o-mini               # 按服务商选择
export HE_PROXY=http://127.0.0.1:2080     # 外网 API 需要代理时设置；国内 API 可留空
python -m human_engine.cli --demo         # 现在走真实 LLM
```

- **perceive**：LLM 做事件结构化（12 类事件）
- **appraise**：LLM 按 Scherer 六维检查表输出评价（JSON），**失败自动降级为规则版**，模拟不会中断
- **generate**：注入人格+状态+相关记忆+行为决策，生成第一人称反应
- 未设置 `HE_API_KEY` 时自动用 MockLLM（离线可跑）

## 当前能力（M1–M3）

- ✅ 完整 8 步事件管线（感知→appraisal→情绪→应激→记忆→决策→行为→反馈）
- ✅ ALMA 三层情绪动力学（人格→心境→情绪，PAD 弛豫）
- ✅ 自传体记忆 + 遗忘曲线 + 心境一致性/反刍检索
- ✅ 创伤通道：概率化闪回、回避预增、倾诉/直面处理（consolidate）
- ✅ 崩溃触发 + 不可逆后果（阈值永久下降、异稳态磨损、习得性无助）
- ✅ 越轨回路：冲动→规范对抗→道德脱离→内疚/羞耻；匿名机会因素
- ✅ 真实 LLM 接入（OpenAI 兼容，可插拔+自动降级）

## 下一步（见 spec §7）

- 关系图（好感/信任/权力差）、睡眠-生理循环深化
- 细致前端（M4/M5）与验证迭代（M6）
