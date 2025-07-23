---
marp: true
paginate: true
footer: 2025-07-23
style: |
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
math: mathjax
---

# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

通过优化访存提高 Attention 算子计算效率

---

# Background

- 标准的 self-attention $O = softmax(QK^T)V$ 会分成三步计算（3-pass）
  - $S = QK^T$
  - $P = softmax(S)$
  - $O = PV$
- 空间复杂度 $O(N^2)$，当 N 较大时，访存压力快速增加

---

# Online Softmax

Safe softmax:
$softmax({x_1,...,x_N}) = {\frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}} = {\frac{e^{x_i - m}}{\sum_{j=1}^{N} e^{x_j - m}}}, m = max_{i=1}^{N}(x_i)$

$$
3-pass
\begin{cases}
m_i = max(m_{i-1}, x_i) \\ 
d_i = d_{i-1} + e^{x_i - m_N} \\
a_i = \frac{e_{x_i - m_N}}{d_N} \\
\end{cases}
$$


Online softmax:
$$
2-pass
\begin{cases}
m_i = max(m_{i-1}, x_i), d'_i = d'_{i-1} * e^{m_{i - 1} - m_i} + e^{x_i - m_i} \\
a_i = \frac{e_{x_i - m_N}}{d'_N} \\
\end{cases}
$$

不保证中间结果 (partial result) 的正确性，每次计算时更新之前的结果：
$d_i \neq d'_i (i < N), d_N = d'_N$

---

# FlashAttention-v1 2022

$$
2-pass
\begin{cases}
m_i &= max(m_{i-1}, x_i), d'_i = d'_{i-1} * e^{m_{i - 1} - m_i} + e^{x_i - m_i} \\
o_i &= \sum_{j=1}^i a_j V[j,:]  \\
    &= \sum_{j=1}^i(\frac{e^{x_j - m_N}}{d'_N}V[j,:])
\end{cases}
$$

$$
1-pass
\begin{cases}
o'_i &= \sum_{j=1}^i(\frac{e^{x_j - m_i}}{d'_i}V[j,:]) \\
     &= o'_{i-1} \frac{d'_{i-1} e^{m_{i-1} - m_i}}{d'_i} + \frac{e^{x_i - m_i}}{d'_i} V[i,:]
\end{cases}
$$

不保证中间结果 (partial result) 的正确性，每次计算时更新之前的结果：
$o_j \neq o'_j (j < i), o_i = o'_i$

For more details: [From Online Softmax to FlashAttention](https://link.zhihu.com/?target=https%3A//courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)

Attention 空间复杂度降低为 $O(N)$

---

<div style="font-size: x-large">

- FlashAttention-v1 2022
  - 针对训练场景下 attention 优化
  - 3x training speed up (GPT-2 with 1K seq_len)，
- FlashAttention-v2 2023
  - 减少非 matmul 计算量， 增加 Q 在 N 维度的并行
  - 1.3-1.5x faster in forward pass, 2x faster in backward pass than FA1
- FlashAttention-v3 2024
  - 针对 Hopper 架构优化
  - 1.5-2.0x faster than FA2 on Hopper GPU
- FlashDecoding 2023
  - 针对推理场景（KV Cache）优化，split_k 思想增加 K/V 的并行度
  - 50x faster than FA2 (128K context), 8x end-to-end speed up
- FlashDecoding++ 2025 MLSys
  - 预先统计 softmax 输入的范围，用先验值 $\phi$ 取代 max
  - 1.37x faster than FlashDecoding

</div>