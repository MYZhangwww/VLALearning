# 08. Standard Attention -> FlashAttention (IO-Aware Exact Attention)

随着 Context Length 的增长（从 2k 到 32k, 100k, 1M），Attention 的 $O(N^2)$ 显存和计算复杂度成为最大瓶颈。**FlashAttention** (Dao et al., 2022/2023) 的出现彻底改变了这一局面，它不是近似算法，而是通过**IO 感知（IO-Awareness）**实现的**精确注意力**加速。

## 1. 背景：HBM vs SRAM
GPU 的存储结构分为多级：
- **HBM (High Bandwidth Memory)**：显存（如 A100 的 80GB）。容量大，但读写速度慢。
- **SRAM (Static RAM)**：计算单元附近的缓存（如 L1/Shared Memory）。速度极快，但容量极小（每 SM 仅 ~100KB）。

**标准 Attention 实现的问题**：
在计算 $S = QK^T$, $P = \text{softmax}(S)$, $O = PV$ 时，PyTorch 会频繁地将巨大的 $S$ 和 $P$ 矩阵（尺寸 $N \times N$）在 HBM 和 SRAM 之间读写。这种频繁的 **Memory Access (IO)** 是主要的性能瓶颈，而非计算本身。

## 2. FlashAttention 核心机制
FlashAttention 使用 **Tiling (分块)** 和 **Recomputation (重计算)** 技术来优化 IO。

### (a) Tiling (分块计算)
它将 $Q, K, V$ 切分成小块（Block），使其可以完全装入高速的 **SRAM**。
在 SRAM 内部计算 Attention 分数，更新输出，然后只将最终结果写回 HBM。
**关键点**：由于 $N \times N$ 的中间矩阵 $S$ 和 $P$ 根本不需要完整地写入 HBM，这减少了 **90% 以上** 的显存读写量。

### (b) Recomputation (反向传播优化)
在前向传播时，为了节省显存，FlashAttention **不存储**巨大的 Attention Map ($N \times N$)。
在反向传播时，它根据 Output 和 Block 的统计量，重新快速计算一遍 Attention Score。
虽然增加了一些计算量（FLOPs），但由于减少了最慢的 HBM 读写，总速度反而更快。

## 3. FlashAttention-2 & 3
- **FlashAttention-2** (2023): 进一步优化了并行策略（将 Q 和 KV 的循环并行化），减少了非 MatMul 操作的开销，在 A100 上达到了理论峰值 FLOPs 的 50-70%。
- **FlashAttention-3** (2024): 针对 Hopper 架构 (H100) 的 FP8 和 Tensor Core 进行了极致优化。

## 4. 影响与意义
1.  **训练速度提升**：通常加速 2-4 倍。
2.  **长文本成为可能**：原本受限于 $O(N^2)$ 显存的 4k 长度，现在可以轻松扩展到 32k、128k 甚至更长，因为显存占用从 $O(N^2)$ 降到了 $O(N)$（线性显存）。
3.  **标准化**：现在几乎所有的主流大模型训练框架（Megatron-LM, Deepspeed, PyTorch SDPA）都默认集成了 FlashAttention。

## 5. 总结

| 特性 | Standard Attention | FlashAttention |
| :--- | :--- | :--- |
| **计算结果** | 精确 | 精确 (无损) |
| **显存瓶颈** | $O(N^2)$ (存储 Attention Map) | **$O(N)$ (线性)** |
| **IO 访问** | 频繁读写 HBM | 极致减少 HBM 访问 |
| **核心技巧** | 无 | Tiling (分块) + Recomputation |
| **影响** | 限制了长文本发展 | 开启了 Long Context 时代 |
