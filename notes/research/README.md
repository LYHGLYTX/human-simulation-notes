# research/

常规研究笔记 — topic-organized research notes.

按主题分类的知识沉淀：理论笔记、文献综述、设计 spec、架构蓝图、实验总结。

**命名规则**：每个主题一个子目录，`kebab-case`：

```
research/
├── emotion/        # 情绪理论与计算模型（Appraisal、PAD、ALMA…）
├── memory/         # 记忆系统（多系统记忆、遗忘曲线、agent 记忆架构）
├── trauma/         # 创伤与应激（PTSD 模型、崩溃动力学）
├── personality/    # 人格（大五、黑暗三合一、依恋风格）
├── morality/       # 道德与越轨（道德脱离、GST、攻击模型）
├── decision/       # 决策（前景理论、双系统、主动推理）
├── cognition/      # 认知与架构（注意、工作记忆、ACT-R/Soar）
├── neuroscience/   # 神经科学（预测编码、自由能、情绪回路）
├── psychopathology/# 心理病理（抑郁、焦虑、人格障碍、自杀、成瘾）
└── social/         # 社会心理（归因、从众、群体、关系）
```

每个主题目录内有 `theories.md`（该领域**理论全集**：所有主流理论/模型/文献，一条一行）+ 各专题展开文档（`emotion/appraisal-theory.md` 等）。
