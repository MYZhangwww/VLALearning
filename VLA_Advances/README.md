# VLA 技术进展学习文档索引 (Index of VLA Advances)

本文档汇总了截止 2026 年 2 月 Vision-Language-Action Models (VLA) 领域的 11 项关键技术演进。

VLA 是 VLM 的自然延伸——VLM 让模型"看懂世界"，VLA 则让模型"动手改变世界"。

---

## Part 4: 视觉-语言-动作模型 (VLA Foundation)

### 17. [VLA 概述：从感知到行动的范式转变](./17_VLA_Overview.md)
*   **演进点**：机器人控制从手工策略 → 学习策略 → 基座模型策略。
*   **核心价值**：理解 VLA 在 LLM → VLM → VLA 技术栈中的位置，以及它为什么是具身智能 (Embodied AI) 的核心。
*   **关键概念**：感知-决策-执行闭环、基座模型范式。

### 18. [RT-1 到 RT-2：VLA 的诞生](./18_RT1_RT2_Foundation.md)
*   **演进点**：从专用 Robotics Transformer 到 VLM 驱动的 VLA。
*   **核心价值**：RT-2 首次证明 **互联网规模的视觉-语言知识可以直接迁移到机器人控制**，开创了 VLA 这一模型类别。
*   **代表模型**：RT-1, RT-2, RT-2-X。

### 19. [VLA 架构深度解析：如何把 VLM 变成机器人控制器](./19_VLA_Architecture.md)
*   **演进点**：VLA 的五大架构范式——自回归、扩散、强化、混合、专用。
*   **核心价值**：深入拆解 VLA 的输入-处理-输出流水线，理解视觉编码器、语言模型、动作解码头三大模块的协作。
*   **代表模型**：RT-2, OpenVLA, Octo, π0。

### 20. [动作表示：从离散 Token 到扩散生成](./20_Action_Tokenization.md)
*   **演进点**：动作空间的编码方式——离散化 Bin → FAST 频域压缩 → 扩散去噪。
*   **核心价值**：动作表示是 VLA 区别于 VLM 的核心差异，不同编码方式直接决定了模型的精度、速度和泛化能力。
*   **代表方法**：RT-2 Binning, FAST, Diffusion Policy, Action Chunking。

### 21. [OpenVLA：开源 VLA 的里程碑](./21_OpenVLA.md)
*   **演进点**：从闭源 55B RT-2-X 到开源 7B OpenVLA，性能反超 16.5%。
*   **核心价值**：首个可在消费级 GPU 上微调的开源 VLA，极大降低了机器人基座模型的研究门槛。
*   **代表模型**：OpenVLA, OpenVLA-OFT。

### 22. [π0 系列：通用机器人基座模型](./22_Pi0_Generalist_Policy.md)
*   **演进点**：从单任务策略 → 多任务泛化 → 开放世界泛化 (π0 → π0.5)。
*   **核心价值**：Physical Intelligence 的 π0 系列是目前最接近"通用机器人大脑"的模型，能在从未见过的家庭环境中完成复杂长程任务。
*   **代表模型**：π0, π0-FAST, π0.5。

## Part 5: VLA 关键技术 (VLA Key Technologies)

### 23. [数据与跨具身泛化：Open X-Embodiment](./23_Data_Cross_Embodiment.md)
*   **演进点**：从单机器人数据集到跨具身统一数据集。
*   **核心价值**：Open X-Embodiment 汇聚 22 个机器人平台的百万级演示数据，证明了**不同机器人的数据可以互相受益**，奠定了 VLA 的"ImageNet 时刻"。
*   **代表数据集**：Open X-Embodiment, DROID, BridgeData V2。

### 24. [世界模型与机器人规划](./24_World_Models_Planning.md)
*   **演进点**：从反应式控制到预测式规划——让机器人"先在脑中预演"。
*   **核心价值**：视频世界模型让 VLA 可以预测未来画面和奖励，在虚拟空间中规划最优动作序列。
*   **代表方法**：UniPi, iVideoGPT, World-VLA-Loop, ViPRA。

### 25. [Sim-to-Real：从仿真到真实世界](./25_Sim2Real_Transfer.md)
*   **演进点**：解决 VLA 最大瓶颈——真实数据稀缺和安全风险。
*   **核心价值**：通过仿真训练 + 域随机化 + 师生蒸馏，实现从虚拟环境到真实机器人的零样本迁移。
*   **代表方法**：Sim2Real-VLA, Isaac Lab, Domain Randomization。

### 26. [VLA 部署：实时控制与边缘推理](./26_Edge_Deployment.md)
*   **演进点**：7B 参数的模型如何在机器人上实现 30Hz 实时控制？
*   **核心价值**：从量化、异步推理到轻量化模型，解决 VLA 从实验室到真实机器人的"最后一公里"问题。
*   **代表方法**：EdgeVLA, LiteVLA, VLASH, BLURR, π0-FAST@30Hz。

### 27. [VLA SOTA 模型全景对比 (截至 2026.02)](./27_VLA_SOTA_Review.md)
*   **演进点**：当前最强 VLA 模型的横向对比与选型指南。
*   **核心价值**：从架构、数据、性能、开源状态等维度全面对比 RT-2-X, OpenVLA, π0.5, Octo 等模型。
*   **代表模型**：RT-2-X, OpenVLA-OFT, π0.5, Octo, GR-2。
