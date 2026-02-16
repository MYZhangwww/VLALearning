# VLM 技术进展学习文档索引 (Index of VLM Advances)

本文档汇总了截止 2026 年 2 月 Vision-Language Models (VLM) 领域的 **11 项**关键技术演进。

---

## Part 3: 多模态架构与训练 (VLM Architecture & Training)

### 11. [Connector: Q-Former → MLP](./11_Connector.md)
*   **演进点**：模态连接器的简化之路。
*   **核心价值**：从复杂的 Query Transformer 回归到简单的 **MLP Projection**，证明了在 LLM 足够强的情况下，保留原始视觉特征比压缩特征更好。
*   **代表模型**：LLaVA, GPT-4V.

### 12. [Resolution: Fixed Res → AnyRes](./12_Resolution_AnyRes.md)
*   **演进点**：从固定分辨率 (Resize) 到 动态高分辨率切片 (Dynamic Tiling)。
*   **核心价值**：通过 **AnyRes (Crop & Fuse)** 策略，让 VLM 能够看清高清图片中的细粒度物体、OCR 文字和图表，打破了 ViT 的分辨率限制。
*   **代表模型**：LLaVA-NeXT, InternVL.

### 13. [Architecture: V-Encoder → Pure Decoder](./13_Decoder_Only_VLM.md)
*   **演进点**：从双塔结构 (ViT + LLM) 到 纯解码器架构 (Pure Decoder)。
*   **核心价值**：移除 Vision Encoder，直接将像素片段 (Patch) 线性投影进入 LLM。支持原生任意分辨率，架构极简，利于端侧部署和文档理解。
*   **代表模型**：Fuyu-8B, Molmo, Nougat.

### 14. [Training: Img-Txt Align → Visual Instruction Tuning](./14_Visual_Instruction_Tuning.md)
*   **演进点**：从对齐预训练 (CLIP) 到 视觉指令微调 (Visual SFT)。
*   **核心价值**：利用纯文本 GPT-4 生成多模态对话数据，将 VLM 从"看图说话"进化为"听懂指令"的 **Visual Chatbot**。含 RLHF 详解。
*   **代表模型**：LLaVA, InstructBLIP.

### 15. [Reasoning: VQA → Visual CoT](./15_Visual_CoT.md)
*   **演进点**：从直接回答 (VQA) 到 视觉思维链 (Visual Chain-of-Thought)。
*   **核心价值**：强制模型输出 `观察 -> 分析 -> 结论` 的推理过程，显著提升了在几何、逻辑、科学问题上的准确率，减少了通过 Shortcut 猜答案的现象。
*   **代表模型**：LLaVA-1.5, GPT-4o, Gemini 1.5.

### 16. [SOTA Models: Qwen2-VL vs Gemini 1.5 vs GPT-4o (+ 后续 Gemini 3 / GPT-5.2 / Qwen3)](./16_SOTA_Models_Review.md)
*   **演进点**：当前最强 VLM 模型架构深度解析 + 后续模型全面对比。
*   **核心价值**：
    *   **Qwen2-VL → Qwen3**: 动态分辨率 + 开源全模态 + 音频 SOTA。
    *   **Gemini 1.5 → 3 Pro**: 1M 上下文 + 帧级视频推理，感知之王。
    *   **GPT-4o → 5.2**: 端到端语音 + 自验证机制，推理之王。
*   **代表模型**：Qwen3, Gemini 3 Pro, GPT-5.2.

---

## 补充专题 (Supplementary Topics)

以下 5 篇文档补充了 VLM 进展中同样重要但未被上述主线覆盖的关键技术：

### S1. [视觉编码器演进：CLIP → SigLIP → DINOv2 → SigLIP 2](./VLM_S1_Vision_Encoder_Evolution.md)
*   **演进点**：VLM 的"眼睛"从单一对比学习走向多目标联合训练。
*   **核心价值**：CLIP 提供语义，DINOv2 提供空间感知，**双编码器融合**成为 VLA 标配。SigLIP 2 (400M) 以多目标训练打败 InternViT (6B)，证明**训练方法 > 模型规模**。
*   **代表模型**：CLIP, SigLIP 2, DINOv2, Prismatic VLMs.

### S2. [视觉接地：从"看图说话"到"指哪说哪"](./VLM_S2_Visual_Grounding.md)
*   **演进点**：VLM 从纯文本输出 → 输出空间坐标（边界框/点坐标）。
*   **核心价值**：Grounding 是 **VLM → VLA 的桥梁**——机器人操作、UI 自动化、医学影像都需要精确定位。
*   **代表模型**：Kosmos-2, Ferret, CogAgent, Set-of-Mark.

### S3. [视频理解：从单张图片到时间流](./VLM_S3_Video_Understanding.md)
*   **演进点**：从单图理解 → 多帧分析 → 数小时长视频理解。
*   **核心价值**：帧采样、Token 压缩、M-RoPE 时间位置编码让 VLM 理解动态世界。**视频理解是 VLM 静态感知到 VLA 动态行动的关键桥梁**。
*   **代表模型**：Video-LLaVA, Qwen2-VL, Gemini 3 Pro.

### S4. [多模态幻觉：VLM 最致命的缺陷](./VLM_S4_Hallucination.md)
*   **演进点**：识别和对抗 VLM 特有的"编造不存在内容"问题。
*   **核心价值**：对比解码 (VCD/ICD/LCD) 提供**无需重训的即插即用**方案；RLHF/DPO 从训练层面根治。在 VLA 中，幻觉从"说错话"升级为"做错事"。
*   **评测基准**：POPE, CHAIR, MMHal-Bench.

### S5. [统一视觉模型：从"只看"到"又看又画"](./VLM_S5_Unified_Vision.md)
*   **演进点**：VLM 从纯理解 → 理解+生成的统一模型。
*   **核心价值**：通过视觉 Tokenization (VQ-VAE) 将图像纳入自回归框架，**一个模型同时理解和生成图像**。为 VLA 的世界模型和视觉目标生成奠定基础。
*   **代表模型**：Chameleon, Emu2, Show-o, Janus.
