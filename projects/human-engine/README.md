# human-engine

模拟人类精神状态的**核心引擎**（M1 里程碑，MockLLM 离线版）。

设计依据：`../../notes/research/design/spec.md`（spec v0.1）
理论依据：`../../notes/research/<主题>/theories.md`（10 模块理论全集）

## 架构

```
human_engine/
├── persona.py           # 静态人格（大五/黑暗三合一/BIS-BAS/依恋/图式/人生年表）
├── persona_generation.py # 访谈生成个体（1,000 People 范式，离线规则版）
├── state.py             # 动态状态（PAD/心境/应激/资源/信念/生理/学习信号）
├── emotion.py           # ALMA 三层 + PAD 弛豫 + WASABI 次情绪 + FLAME 延迟爆发
├── memory.py            # 记忆（遗忘曲线检索/创伤通道/反思洞察/归档/自编辑）
├── stress.py            # 应激累积/崩溃触发/不可逆后果（敏化+磨损）
├── morality.py          # 冲动生成/规范对抗/道德脱离/内疚羞耻回路
├── coping.py            # EMA coping 策略选择（problem/emotion/avoidance）
├── relations.py         # 社会关系图（好感/信任/权力差/依恋激活 + 社会支持）
├── physiology.py        # 生理层（精力/睡眠债/昼夜节律/睡眠结算）
├── appraisal.py         # 事件结构化 + Scherer 检查表（规则版）
├── llm.py               # LLMClient 协议 + MockLLM（可换真实 LLM）
├── engine.py            # tick 主循环 + 8 步事件管线 + 决策（EFE 启发式）
└── cli.py               # 交互式 CLI
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

## 当前能力（M1–M3 + 关系/生理层 + 深化 pass）

- ✅ 完整 8 步事件管线（感知→appraisal→情绪→应激→记忆→决策→行为→反馈）
- ✅ ALMA 三层情绪动力学 + WASABI 主次情绪（认知深度减慢弛豫） + FLAME 延迟爆发
- ✅ 自传体记忆 + 遗忘曲线 + 反思洞察（自我叙事）+ 记忆压力归档 + 同主题自编辑（Mem0）
- ✅ 创伤通道：概率化闪回、回避预增、倾诉/直面处理（consolidate）
- ✅ 崩溃触发 + 不可逆后果（阈值永久下降、异稳态磨损、习得性无助→冻结）
- ✅ 越轨回路：冲动→规范对抗→道德脱离→内疚/羞耻；匿名机会因素
- ✅ 真实 LLM 接入（OpenAI 兼容，可插拔+自动降级）
- ✅ 社会关系图：好感/信任/权力差/依恋激活；社会支持作为 COR 资源
- ✅ 生理层：精力/睡眠债/昼夜节律/疲劳；睡眠 = 最大恢复源 + 记忆巩固窗口
- ✅ 信念层（PsychSim 式）：信念随经历漂移，给 appraisal 上色（威胁高估/无助感）
- ✅ EMA coping 策略选择（问题聚焦/情绪聚焦/回避，受习得性无助压制）
- ✅ 决策含主动推理启发式（心境作先验偏好 + 新颖性探索项）
- ✅ 学习：RPE 奖赏期望适应（PVLV-lite）+ 事件序列预测误差（htm-lite）
- ✅ 访谈生成个体（1,000 People 范式：问卷答案 → 人格/图式/人生年表）
- ✅ 临床量表（PCL-5/PHQ-9/PANAS/STAI）+ 场景库断言验证

## 🧪 验证（阶段 4：一致性 + 量表 + 场景库）

```bash
cd projects/human-engine
python -m unittest discover -s tests   # 32 个测试
```

- **一致性**：同 seed 完全复现（确定性）；同事件 20 seeds 主行为占比 ≥60%（不漂移）；高神经质→更高应激、高 BAS→更高冲动（人格对比方向正确）；创伤史→同样背叛事件应激更高（时间线效应）
- **场景库**（`human_engine/scenarios.py`）：日常/压力/创伤/极端四类，含理论预期断言（daily 不崩溃、extreme 必崩溃且阈值永久下降…）
- **量表**（`human_engine/psychometrics.py`）：简化 PCL-5（PTSD）/PHQ-9（抑郁）/PANAS/STAI-S，从连续状态推导——多次崩溃后 PHQ-9 显著升高、创伤场景后 PCL-5 显著升高（研究用途，非诊断）

```python
from human_engine.engine import Engine
from human_engine.scenarios import run_scenario
from human_engine.psychometrics import full_report
e = Engine(seed=1)
run_scenario(e, "trauma")
print(full_report(e))
```

## 下一步（见 spec §7）

- 细致前端（M4/M5：状态表/推理链/量表/批量模拟）
- 专家盲评（M6）
- 明确不做（记录在 note）：Nengo 全脑模拟、concordia GM 架构（超出轻量引擎范围）
