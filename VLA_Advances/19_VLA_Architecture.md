# 19. VLA 架构深度解析：如何把 VLM 变成机器人控制器

## 1. VLA 的通用架构模板

几乎所有 VLA 模型都遵循相同的三段式流水线：

```
┌──────────────┐   ┌──────────────────────┐   ┌───────────────────┐
│ 视觉编码器    │   │ 骨干网络 (Backbone)    │   │ 动作解码头         │
│ Vision Encoder│──→│ (通常是 LLM)          │──→│ Action Head        │
│ ViT / CLIP   │   │ Llama / PaLM / Qwen  │   │ AR / Diffusion     │
└──────────────┘   └──────────────────────┘   └───────────────────┘
        ↑                    ↑                          ↓
   [RGB 图像]         [语言指令]                [机器人动作序列]
                    [本体感受(可选)]
```

每段的职责：
*   **视觉编码器**：把 RGB 图像变成视觉 Token 序列
*   **骨干 LLM**：融合视觉 + 语言 + 本体感受，进行多模态推理
*   **动作解码头**：把 LLM 的隐藏状态转换为机器人可执行的动作

## 2. 五大 VLA 架构范式

根据 2025 年的综合综述，VLA 模型可分为五大范式：

### (a) 自回归范式 (Autoregressive VLA)

**核心思想**：把动作当成 Token，像生成文本一样逐个生成。

```
输入: [Image Tokens] [Text Tokens] → LLM → [Action Token 1] [Action Token 2] ... [Action Token 7]
```

**代表模型**：RT-2, OpenVLA

**优点**：
*   架构简单，直接复用 VLM 的自回归框架
*   天然支持语言指令和动作的联合建模

**缺点**：
*   逐 Token 生成速度慢
*   离散化精度有限（每维 256 Bin = ~0.4% 精度）
*   无法建模动作维度间的**联合分布**

### (b) 扩散范式 (Diffusion VLA)

**核心思想**：LLM 输出特征，接一个扩散模型来生成连续动作。

```
输入: [Image] [Text] → LLM → [Hidden State] → 🌊 Diffusion Head → [连续动作序列]
```

**代表模型**：π0, Octo

**优点**：
*   输出连续值，精度极高
*   能建模多模态分布（同一任务可能有多种正确动作）
*   Action Chunking：一次生成整段动作序列

**缺点**：
*   扩散去噪需要多步迭代，推理速度慢
*   训练和调参比自回归复杂

### (c) 强化学习范式 (RL-based VLA)

**核心思想**：用 VLM 作为奖励模型或价值函数，配合 RL 训练策略。

**代表方法**：VLM-as-Reward, World-VLA-Loop

**优点**：
*   无需人工演示数据（可以自主探索）
*   可以优化长期目标

**缺点**：
*   训练不稳定，样本效率低
*   需要环境交互（或仿真器）

### (d) 混合范式 (Hybrid VLA)

**核心思想**：组合上述范式的优势。

**代表模型**：π0-FAST（自回归 + FAST 编码）、Discrete Diffusion VLA

**典型组合**：
*   自回归 VLM + 扩散动作头（π0）
*   自回归框架 + 频域动作编码（π0-FAST）
*   离散扩散 + Transformer 骨干（Discrete Diffusion VLA）

### (e) 专用范式 (Specialized VLA)

**核心思想**：为特定任务或场景设计专门的架构。

**代表模型**：导航专用 VLA、双臂操作 VLA

## 3. 视觉编码器的选择

| 编码器 | 预训练方式 | 用于 VLA 的代表 | 特点 |
| :--- | :--- | :--- | :--- |
| **CLIP ViT** | 对比学习 (图文对齐) | RT-2 | 语义丰富，但缺少空间细节 |
| **SigLIP** | Sigmoid 对比学习 | OpenVLA | 比 CLIP 训练更稳定 |
| **DINOv2** | 自监督学习 (MAE) | OpenVLA | 空间细节丰富，适合精细操作 |
| **DINOv2 + SigLIP** | 双编码器融合 | **OpenVLA** | 🔑 语义 + 空间 = 最佳组合 |
| **ViT (from scratch)** | 联合训练 | Octo | 灵活但需要更多数据 |

> **关键洞察**：OpenVLA 的 DINOv2 + SigLIP 双编码器策略成为了 VLA 视觉编码的标准做法——DINOv2 提供空间精度（物体在哪里），SigLIP 提供语义理解（物体是什么）。

## 4. 骨干 LLM 的角色

LLM 在 VLA 中不仅仅是"语言模型"，它扮演的是**多模态融合与推理引擎**：

| 功能 | 说明 |
| :--- | :--- |
| **多模态融合** | 将视觉 Token、语言 Token、本体感受 Token 在统一的注意力中交互 |
| **指令理解** | 解析自然语言指令的含义和意图 |
| **世界知识** | 利用预训练知识理解物体属性（"玻璃杯是易碎的"） |
| **推理** | 多步推理能力（"要打开抽屉，先要松开手里的东西"） |
| **时序建模** | 通过历史帧理解动作的时间依赖关系 |

常用 LLM 骨干：

| LLM | 参数量 | 用于 VLA |
| :--- | :--- | :--- |
| Llama 2 7B | 7B | OpenVLA |
| PaLI-X | 55B | RT-2 |
| PaLM-E | 12B | RT-2 |
| Qwen2 | 多种 | 最新 VLA 工作 |

## 5. 动作解码头的比较

| 动作头类型 | 输出方式 | 精度 | 速度 | 多模态分布 | 代表 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **分类头 (Binning)** | 每维 256 类 | 中 | 快 | ❌ | RT-1, RT-2 |
| **自回归 Token** | 逐 Token 生成 | 中 | 慢 | ❌ | OpenVLA |
| **MLP 回归** | 直接输出连续值 | 高 | 最快 | ❌ | 简单 BC |
| **扩散头 (Diffusion)** | 迭代去噪 | **最高** | 慢 | **✅** | π0, Octo |
| **FAST Token** | 频域压缩 Token | 高 | 快 | ❌ | π0-FAST |

## 6. 实际架构案例

### RT-2 架构

```
[Image 300x300] → ViT-G (4B) → 视觉 Token
                                    ↓
[Text Instruction] → Tokenizer → 文本 Token → PaLI-X (55B) → 自回归解码
                                                               ↓
                                              "1 128 91 241 5 101 128"
                                                               ↓
                                              解码为 7 维连续动作
```

### OpenVLA 架构

```
[Image 224x224] → DINOv2 + SigLIP (双编码器) → 融合视觉 Token
                                                      ↓
[Text Instruction] → Tokenizer → 文本 Token → Llama 2 (7B) → 自回归解码
                                                                ↓
                                                     7 个动作 Token (每维 256 Bin)
                                                                ↓
                                                      解码为 7 维连续动作
```

### π0 架构

```
[Multi-View Images] → ViT → 视觉 Token
                                ↓
[Text Instruction] → Tokenizer → 文本 Token → Pre-trained VLM → Hidden State
                                                                     ↓
[Proprioception] → MLP → 本体 Token ──────────────────────────→ Diffusion Head
                                                                     ↓
                                                          Action Chunk (未来 N 步动作)
```

## 7. 关键架构趋势

1.  **双视觉编码器成为标配**：DINOv2 (空间) + SigLIP/CLIP (语义) 的组合被广泛采用。
2.  **扩散头取代分类头**：精度更高，能建模多模态分布，适合灵巧操作。
3.  **Action Chunking (动作分块)**：一次预测未来 N 步动作（如 N=16），而非逐步预测，提高控制平滑度和推理效率。
4.  **多视角输入**：从单一腕部摄像头 → 腕部 + 全局两个摄像头 → 更多视角，提高空间感知。
5.  **MoE 架构渗透**：Qwen3-Omni 等 MoE 模型开始进入 VLA 领域，以较少的激活参数实现更大模型容量。

> **一句话总结**：VLA 架构 = VLM 架构 + 动作解码头。视觉编码器看世界，LLM 理解并推理，动作头将意图转化为物理动作——三者协同让 AI 从"看懂"走向"做到"。
