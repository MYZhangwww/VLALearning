# LLM 技术进展学习文档索引 (Index of LLM Advances)

本文档汇总了过去 5 年 Large Language Models (LLM) 领域的 10 项关键技术演进，分为 **Part 1 (基础架构)** 和 **Part 2 (效率与 Scaling)** 两部分。

---

## Part 1: 基础架构演进 (Fundamental Advances)

### 1. [LayerNorm → RMSNorm](./01_LayerNorm_to_RMSNorm.md)
*   **演进点**：从 Layer Normalization 到 Root Mean Square Normalization。
*   **核心价值**：去除了 Mean 统计量，仅保留缩放不变性。**计算效率更高**，且在深层网络训练中具有**更好的梯度稳定性**。
*   **代表模型**：LLaMA, Gopher, Chinchilla.

### 2. [APE → RoPE](./02_APE_to_RoPE.md)
*   **演进点**：从绝对位置嵌入 (Absolute Positional Embedding) 到旋转位置编码 (Rotary Positional Embedding)。
*   **核心价值**：通过复数旋转引入**相对位置信息**，具有更好的**长度外推性** (Extrapolation)。
*   **代表模型**：LLaMA, PaLM, GLM, Qwen.

### 3. [FFN → Gated FFN (SwiGLU)](./03_FFN_to_GatedFFN.md)
*   **演进点**：从标准 ReLU/GELU FFN 到 Gated Linear Units (GLU) 变体。
*   **核心价值**：引入**门控机制** (Gating)，增加了模型的非线性表达能力，通常能带来显著的 Performance 提升。
*   **代表模型**：PaLM, LLaMA, Mixtral.

### 4. [Attention → QK-Norm + 稳定性设计](./04_Attention_Stability.md)
*   **演进点**：应对 Attention Logits 爆炸导致的训练不稳定。
*   **核心价值**：通过 **Query-Key Normalization** 或 **Logits Scaling**，防止 Attention Score 过大，稳定训练梯度。
*   **代表模型**：ViT-22B, GLM-130B.

### 5. [Register / Memory Tokens + StreamLLM](./05_Register_Memory_Token.md)
*   **演进点**：利用额外的 Token 来存储全局信息或作为 "Attention Sink"。
*   **核心价值**：
    *   **Register Token**：消除 Vision Transformer 中的伪影。
    *   **StreamLLM**：利用 **Attention Sink (首 Token)** 机制，实现**无限长度的流式推理**，防止上下文窗口滑动导致的崩盘。
*   **代表模型**：DINOv2, StreamLLM (Llama-2 optimized).

---

## Part 2: 效率与 Scaling 进阶 (Efficiency & Scaling)

### 6. [MHA → MQA / GQA (Inference Acceleration)](./06_MQA_GQA.md)
*   **演进点**：从 Multi-Head Attention 到 Multi-Query / Grouped-Query Attention。
*   **核心价值**：大幅减少 **KV Cache** 的显存占用（最高压缩 32 倍），解决 Memory Bandwidth 瓶颈，从 Memory Bound 转为 Compute Bound。
*   **代表模型**：LLaMA-2 (70B), LLaMA-3, Mistral.

### 7. [RoPE → ALiBi (Positional Extrapolation)](./07_ALiBi.md)
*   **演进点**：通过线性偏置 (Linear Bias) 替代显式位置编码。
*   **核心价值**：具有极强的**长度外推能力**，在短序列上训练的模型可直接推理长序列。
*   **代表模型**：MPT-7B, BLOOM.

### 8. [Standard Attention → FlashAttention (IO-Aware)](./08_FlashAttention.md)
*   **演进点**：IO 感知的精确注意力加速算法。
*   **核心价值**：通过 **Tiling (分块)** 和 **Recomputation**，将显存复杂度从 $O(N^2)$ 降为 **$O(N)$**，并极大减少 HBM 访问。是长文本训练的基石。
*   **代表模型**：All Modern LLMs (DeepSpeed, Megatron, Pytorch 2.0).

### 9. [Dense → Sparse MoE (Mixture of Experts)](./09_MoE.md)
*   **演进点**：从稠密网络到稀疏专家网络。
*   **核心价值**：**稀疏激活 (Sparse Activation)**。在保持推理计算量 (FLOPs) 不变的情况下，极大增加模型参数量 (Capacity)。需解决 **Load Balancing** 和 **Stability** 问题。
*   **代表模型**：GPT-4, Mixtral 8x7B, DeepSeekMoE.

### 10. [RLHF (PPO) → DPO (Direct Preference Optimization)](./10_DPO.md)
*   **演进点**：从复杂的 PPO 强化学习到直接偏好优化。
*   **核心价值**：**无需 Reward Model**，无需 Sampling。通过数学转换将 RL 问题变为稳定的监督学习问题。但在复杂推理上可能仍需 Online RL 辅助。
*   **代表模型**：Llama-3-Instruct, Mistral-Instruct, Zephyr.
