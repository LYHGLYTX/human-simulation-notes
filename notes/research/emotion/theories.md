# 情绪理论全集（Emotion Theories）

> 理论地图：覆盖情绪心理学与计算情感建模的所有主流理论。
> 条目格式：**名称**（作者 年份，出处）— 一句话核心。→ 对模拟的用途。
> 精读某条时在本目录新建独立 .md 展开。

## 1. 情绪的本质：离散 vs 维度

| 理论 | 出处 | 核心 | 模拟用途 |
|---|---|---|---|
| **基本情绪理论** | Ekman 1992, *Cognition and Emotion* | 6–7 种跨文化基本情绪（快乐/悲伤/愤怒/恐惧/厌恶/惊讶+轻蔑），各有表情与生理模式 | 情绪标签的离散集合（事件→标签映射） |
| **分化情绪理论** | Izard 1977, *Human Emotions* | 10 种分化情绪，情绪是动机系统 | 情绪-动机耦合 |
| **情感神经科学** | Panksepp 1998, *Affective Neuroscience*（书） | 7 大原始情感系统：SEEKING/RAGE/FEAR/CARE/PANIC/PLAY/LUST | 情绪系统的神经层基座 |
| **环状模型（circumplex）** | Russell 1980, *JPSP* | 情绪=效价×唤醒二维环形空间 | 情绪状态坐标（2D） |
| **PAD 模型** | Mehrabian & Russell 1974（书） | 愉悦-唤醒-支配三维 | **我们的状态机坐标**（3D） |
| **正负激活模型** | Watson & Tellegen 1985, *JPSP* | 情绪=正激活×负激活两正交维度 | 心境测量（PANAS 量表基础） |
| **情绪语义空间** | Cowen & Keltner 2017, *PNAS* | 27 种不同情绪体验的连续语义地图（视频诱发） | LLM 情绪标注的高维参考 |

## 2. 评价理论（Appraisal Theory）— 情绪如何产生

| 理论 | 出处 | 核心 | 模拟用途 |
|---|---|---|---|
| **评价即情绪启动** | Arnold 1960, *Emotion and Personality*（书） | 情绪始于对刺激"好/坏"的直觉评价 | 评价层的最早雏形 |
| **Lazarus 认知-动机-关系理论** | Lazarus 1991, *Emotion and Adaptation*（书） | 初级评价（目标相关性/一致性/自我卷入）+次级评价（责备/应对潜力/未来期望） | **LLM appraisal 抽取的字段来源** |
| **多级顺序检查模型（CPM）** | Scherer 2001, *Appraisal Processes in Emotion* | 五级顺序检查：新颖性→内在愉悦→目标相关性→应对潜力→规范兼容性 | **LLM 检查表蓝本（我们已采用）** |
| **评价维度理论** | Smith & Ellsworth 1985, *JPSP* | 6 维度（愉快/预期努力/确定性/注意/控制/责任）区分情绪 | appraisal→情绪映射表 |
| **Roseman 评价理论** | Roseman 1991, *Cognition and Emotion* | 动机状态×情境×概率×控制×合法性 → 17 种情绪 | 规则式情绪推导 |
| **情绪法则** | Frijda 1986, *The Emotions*（书） | 情绪=对意义的行动倾向（approach/avoid/attack…）；"法则"约束情绪触发 | **情绪→行为倾向接口** |
| **OCC 模型** | Ortony, Clore & Collins 1988（书） | 事件/智能体/对象三类焦点→22 种情绪+强度公式 | 情绪类型学（已列入清单） |

## 3. 情绪与认知/记忆的交互

| 理论 | 出处 | 核心 | 模拟用途 |
|---|---|---|---|
| **情感优先假说** | Zajonc 1980, *American Psychologist* | 情绪反应先于并独立于认知加工 | 快速情绪通道（先于 appraisal 的直觉反应） |
| **情绪即信息** | Schwarz & Clore 1983, *JPSP* | 情绪作为判断的信息源（心情→世界评价） | 情绪→认知偏差调制 |
| **心境一致性记忆** | Bower 1981, *JPSP* | 当前心境增强一致情绪的回忆 | 记忆检索的情绪偏置（已采用） |
| **情绪调节过程模型** | Gross 1998, *Review of General Psychology* | 情境选择/修正→注意部署→认知改变→反应调制 | 应对策略库（压抑/重评/转移） |
| **拓展-构建理论** | Fredrickson 2001, *Review of General Psychology* | 积极情绪拓展认知-行为库并构建资源 | 积极情绪的资源积累效应 |
| **体细胞标记假说** | Damasio 1994, *Descartes' Error*（书） | 躯体状态标记引导决策（"直觉"） | 情绪→决策的快速通道 |
| **情绪-评价倾向** | Lerner & Keltner 2000, *JPSP* | 具体情绪（恐惧/愤怒）对风险判断的特定影响 | 情绪→风险偏好调制 |
| **风险即情绪** | Loewenstein et al. 2001, *Psychological Bulletin* | 风险反应主要由即时情绪驱动而非认知评估 | 恐慌/冲动决策 |

## 4. 经典情绪生理理论（历史脉络）

| 理论 | 出处 | 核心 |
|---|---|---|
| 詹姆斯-兰格理论 | James 1884 / Lange 1885 | 生理变化在先，情绪是对生理的知觉 |
| 坎农-巴德理论 | Cannon 1927 / Bard | 丘脑同时产生生理与情绪 |
| 两因素理论 | Schachter & Singer 1962, *Psychological Review* | 生理唤醒+认知标签=情绪（情绪归因） |

## 5. 计算情绪模型（直接可实现的算法）

| 模型 | 出处 | 核心 | 模拟用途 |
|---|---|---|---|
| **Cathexis** | Velásquez 1997, MIT | 多个情绪原型并行激活+衰减+阈值 | 最早的多原型情绪实现 |
| **FLAME** | El-Nasr 2000 | 模糊逻辑评价+双衰减+延迟爆发 | 延迟爆发=创伤延迟反应（已列入） |
| **ALMA** | Gebhard 2005 | Emotion/Mood/Personality 三层 PAD | **我们的状态机骨架** |
| **WASABI** | Becker-Asano 2008 | PAD 弛豫方程+主/次情绪 | 弛豫动力学（已列入） |
| **EMA** | Marsella & Gratch 2009 | appraisal→coping 因果结构评价 | 情绪→应对接口（已列入） |
| **PARLE-E** | Bui 2004 | 概率评价+动态贝叶斯情绪更新 | 不确定性下的情绪更新 |
| **PAD 弛豫方程族** | Mehrabian 1996 | PAD 值按 personality 基线弛豫 | 情绪惯性/恢复参数 |
