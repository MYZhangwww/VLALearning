## 2026-02-16

### 1. 我要学习大语言模型最近5年的进展。主要有以下要点, 请为每个要点写一篇 详细的MarkDown 文档方便我来学习
1. layerNorm -> RMSNorm
2. APE -> RoPE
3. FFN->Gated FFN
4. Attention -> QK-Norm + 稳定性设计
5. 引入 Register/Memory Token


请为我解释为什么 RMS Norm 其数值稳定性在深层网络中表现也十分优异，有助于缓解梯度消失/爆炸问题？ 并且RMS Norm 跟 PreNorm 和 PostNorm 有什么关系？ 把这一部分的内容也放入第一篇LLM 技术进展文档里

请为我详细解释 第三篇 Gated FFN & SwiGLU 中 SwiGLU 的计算公公式每一项的详细含义与实现方法，并为我逐点详细介绍 SwiGLU为什么有效？提供更丰富的讲解和案例。把这一部分的内容也补充进入第三篇LLM 技术进展文档里

请解释Memory Tokens (LLMs) 的具体实现方式，补充Recurrent Memory Transformer (RMT)： 和 StreamLLM (Attention Sinks) 的具体实现方式，并列举出应用实例。 把这一部分的内容也补充进入第五篇LLM 技术进展文档里

请解释详细第五篇技术文档里： LLM 在流式多轮对话中，显存（KV Cache）会爆满，但简单的“滑动窗口”会导致模型崩盘 的原因。以及解释 直接丢弃最早的 token（滑动窗口），LLM 的 Perplexity 会瞬间暴涨的原因。 在StreamLLM  里 的Rolling Cache 方法里，当新 token 进来时，丢弃 Rolling Cache 中最旧的，但在计算 Attention 时，位置编码（Positional Encoding）需要进行特殊处理（例如相对位置不变），请详细解释。 把这一部分的内容也补充进入第五篇LLM 技术进展文档里


出了这里的五篇LLM技术进展之外，截止到今天还有哪些LLM重要进展，请列出请补充在一个MarkDown 文档里面， 并对每一个要点，类似这五篇LLM技术文档，生成一个MarkDown 文档详细介绍。