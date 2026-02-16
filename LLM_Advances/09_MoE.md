# 09. Dense -> Sparse MoE (Mixture of Experts)

随着模型规模从百亿 (10B) 向千亿 (100B) 甚至万亿 (1T) 迈进，训练和推理成本呈指数级增长。为了在增加模型容量的同时保持计算效率，**Mixture of Experts (MoE)** 架构成为了 Scaling 的关键技术（如 GPT-4, Mixtral 8x7B, DeepSeekMoE）。

## 1. 核心思想：稀疏激活 (Sparse Activation)
传统的 Dense 模型（稠密模型）对于每个输入 token，都会激活网络中的所有参数。这导致计算量（FLOPs）与参数量成正比。

**MoE 模型** 将 FFN 层替换为一组 **Experts（专家网络）**。
- 对于每个 token，只有一小部分 Expert 被激活（例如 8 个中的 2 个）。
- 这意味着：**总参数量极大（High Capacity），但每次推理的计算量极小（Active Parameters << Total Parameters）。**

## 2. 关键组件
一个典型的 MoE 层包含：
1.  **Experts**：$N$ 个独立的 FFN 网络（通常 $N=8, 16, 64...$）。
2.  **Gate / Router (路由网络)**：一个轻量级的线性层，负责根据输入 token 的特征，决定将其分配给哪 $k$ 个 Expert。

$$
y = \sum_{i=1}^{k} g(x)_i \cdot E_i(x)
$$
其中 $g(x)$ 是 Router 输出的权重（通常经过 Softmax），$E_i(x)$ 是第 $i$ 个专家的输出。

## 3. 挑战与解决方案
虽然 MoE 理论上很美，但训练极其困难：

### (a) 负载不均衡 (Load Imbalance)
- **问题**：Router 可能由于初始化随机性，倾向于把所有 token 都发给同一个 Expert（Expert Collapse）。导致该专家过载，其他专家空闲，甚至变成死神经元。
- **解决**：引入 **Load Balancing Loss (辅助损失)**，惩罚分配不均的情况，强制 Router "雨露均沾"。

### (b) 显存开销
- **问题**：虽然计算量小，但总参数量巨大，显存占用极大。
- **解决**：需要高效的并行策略（Expert Parallelism），将不同的 Expert 分布在不同的 GPU 上。

### (c) 训练稳定性
- **问题**：动态路由是个离散决策过程，梯度难以传播。
- **解决**：通常使用 Softmax 的 Top-K 加权求和，使路由过程可微。

## 4. 典型模型
- **Switch Transformer (Google)**: 极端的 Top-1 Routing。
- **Mixtral 8x7B (Mistral AI)**: 高效的 Top-2 Routing，每个 token 激活 2 个专家。效果超越 LLaMA-2 70B，但推理速度和显存占用远小于 70B Dense 模型。
- **DeepSeekMoE**: 细粒度专家（Fine-Grained Experts）+ 共享专家（Shared Experts），进一步提升参数效率。

## 5. 总结

| 特性 | Dense Model | Sparse MoE Model |
| :--- | :--- | :--- |
| **参数量** | $P$ | $N \times P$ (大得多) |
| **计算量 (FLOPs)** | $P$ | $\approx P$ (保持不变) |
| **激活方式** | 全激活 | **稀疏激活 (Sparse Is Better)** |
| **优势** | 训练稳定，部署简单 | 在同等算力下，极大提升上限 |
| **代表作** | GPT-3, LLaMA-2 | GPT-4, Mixtral 8x7B, DeepSeek |
