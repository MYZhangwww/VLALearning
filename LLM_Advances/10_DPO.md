# 10. RLHF (PPO) -> DPO (Direct Preference Optimization)

在 LLM 的对齐（Alignment）阶段，**RLHF (Reinforcement Learning from Human Feedback)** 曾是业界标准（如 InstructGPT/ChatGPT）。然而，RLHF 流程复杂、训练不稳。**DPO (Direct Preference Optimization)** 的出现（Rafailov et al., 2023）将这一过程简化为直接的监督学习，彻底改变了 LLM 的对齐范式。

## 1. 背景：RLHF 的复杂性 (Pipelines)
标准的 PPO-based RLHF 需要维护 **4 个模型** 并不是一件容易的事：
1.  **Reference Model** (冻结的 SFT 模型，防止遗忘)
2.  **Reward Model** (训练好的奖励模型)
3.  **Policy Model** (当前正被训练的 LLM)
4.  **Critic/Value Model** (用于 PPO 价值估计)

**痛点**：
- **资源消耗极大**：同时加载 4 个模型，显存爆炸。
- **超参敏感**：PPO 对超参数极其敏感，容易出现 Reward Hacking 或训练发散。

## 2. DPO 的核心思想
DPO 发现了一个数学上的巧妙转换：**最优策略（Optimal Policy）可以用奖励函数（Reward Function）的显式形式来表示。**

这意味着：我们不需要显式地训练一个 Reward Model，然后用 RL 去优化它。我们可以直接用**Human Preference Data（偏好数据，即 $x, y_w, y_l$）** 来通过类似 Cross-Entropy 的损失函数直接优化 LLM。

### 核心公式
$$
\mathcal{L}_{\text{DPO}} = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_{\theta}(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

其中：
- $y_w$ 是胜出（Winner）的回复，$y_l$ 是失败（Loser）的回复。
- $\beta$ 是控制偏离参考模型程度的超参。
- $\pi_{\theta}$ 是当前策略，$\pi_{\text{ref}}$ 是 SFT 后的参考策略。

**直观理解**：
DPO 使得模型生成 $y_w$ 的概率相对于参考模型**升高**，同时生成 $y_l$ 的概率相对于参考模型**降低**。

## 3. DPO 的优势
1.  **无需 Reward Model**：省去了单独训练 RM 的步骤（虽然隐式地包含在公式里）。
2.  **无需 Sampling**：不需要像 PPO 那样在训练过程中不断生成样本（Sampling），DPO 直接在静态数据集上计算似然概率。
3.  **极简稳定性**：本质上是一个二分类任务（Binary Classification），训练过程和 SFT 一样稳定，显存占用仅略高于 SFT。

## 4. 变体与发展
DPO 的成功引发了大量后续研究，进一步简化或增强对齐过程：
- **KTO (Kahneman-Tversky Optimization)**: 不需要成对数据 ($y_w, y_l$)，只需要单点数据的 Good/Bad 标签。
- **IPO (Identity Preference Optimization)**: 解决了 DPO 可能出现的过拟合（Overfitting）问题。
- **SimPO (Simple Preference Optimization)**: 甚至去掉了 Reference Model，进一步节省显存，速度更快。

## 5. 总结

| 特性 | RLHF (PPO) | DPO (Direct Preference) |
| :--- | :--- | :--- |
| **训练范式** | 强化学习 (RL) | 监督学习 (Supervised Learning) |
| **所需模型数** | 4 个 (Policy, Ref, Reward, Critic) | 2 个 (Policy, Ref) |
| **稳定性** | 极差，超参敏感 | 极好，类似 SFT |
| **计算开销** | 极高 (由于 Sampling) | 较低 |
| **效果** | SOTA (曾经) | 与 PPO 持平甚至更好 |
| **主流框架** | TRL, DeepSpeed-Chat | LLaMA-Factory, TRL, Axolotl |
