# MiniMax M3 Sparse Attention — Triton Baseline

从 vLLM 中提取的 **MiniMax M3 稀疏注意力 Triton kernel 独立项目**，用于性能基准测试和算法研究。

## 项目结构

```
minimax_m3_sparse_baseline/
├── kernels/
│   ├── __init__.py          # 统一导出
│   ├── common.py            # 共享常量 (SPARSE_BLOCK_SIZE=128, PDL 检测等)
│   ├── sparse_attn.py       # 稀疏 GQA Attention kernel (prefill + decode split-K + merge)
│   └── index_topk.py        # Lightning Indexer kernel (score + topk + decode)
├── benchmark.py             # 性能基准测试
├── test_correctness.py      # 正确性验证
├── requirements.txt
└── README.md
```

## 算子概述

### 1. 稀疏注意力 (sparse_attn)

| 函数 | 阶段 | 用途 |
|------|------|------|
| `minimax_m3_sparse_attn` | Prefill | GQA block-sparse attention，只对 indexer 选出的 top-k 块计算 |
| `minimax_m3_sparse_attn_decode` | Decode | Split-K block-sparse attention + merge（flash-decoding 风格） |

底层 Triton kernel：
- `_gqa_sparse_fwd_kernel` — prefill kernel，grid `(max_query_len, num_kv_heads, batch)`
- `_gqa_sparse_decode_kernel` — decode split-K kernel
- `_merge_topk_attn_out_kernel` — 合并 split-K 输出

### 2. Lightning Indexer (index_topk)

| 函数 | 阶段 | 用途 |
|------|------|------|
| `minimax_m3_index_score` | Prefill | Index query 与 index-K cache 的 block-wise max-score 计算 |
| `minimax_m3_index_topk` | Prefill | 从 score 矩阵中选出 top-k 块 (bitonic sort) |
| `minimax_m3_index_decode` | Decode | 融合的 score + split-K topk (包含 partial + merge) |

## 正确性验证

在 **NVIDIA H100 80GB HBM3 (SM 9.0)** + **Triton 3.6.0** 环境下的测试结果：

| 测试 | 状态 | 关键指标 |
|------|------|----------|
| INDEXER | ✓ | 62.5% valid topk entries, 100% in-range block ids |
| PREFILL | ✓ | max abs diff: 0.00195, mean abs diff: 0.00012 |
| DECODE | ✓ | decode vs prefill max diff: 0.00098, mean diff: 0.00012 |

> **说明**：PREFILL 的 max relative diff 较高 (343) 是因为某些输出值接近 0 时分母极小，实际绝对误差仅 ~0.002（bf16 精度正常范围）。DECODE 与 PREFILL 之间的微小差异来自 Split-K merge 的数值舍入。

## 性能基准测试

以下数据在 **NVIDIA H100 80GB HBM3** 上测得，配置为 `num_heads=32, num_kv_heads=8, head_dim=128, topk=8`（`--mode full`，即 `"large"` 配置），dtype=bfloat16。

### Prefill 性能

| seq_len | SparseAttn | IndexScore | IndexTopK | Total Pipeline | TFLOPs/s |
|---------|-----------|------------|-----------|----------------|----------|
| 1024   | 0.322ms   | 0.052ms    | 0.059ms   | **0.432ms**    | 6.7      |
| 2048   | 0.880ms   | 0.048ms    | 0.056ms   | **0.984ms**    | 4.9      |
| 4096   | 1.980ms   | 0.064ms    | 0.056ms   | **2.099ms**    | 4.3      |

**分析**：

- **SparseAttn kernel 时延**随 seq_len 线性增长（~2x per 2x seq_len），原因是 KV 块数翻倍，每个 query 需要计算的 token 数也翻倍。grid 为 `(max_query_len, num_kv_heads, batch)` = `(N, 8, 1)`，query 维度并行度充足。
- **IndexScore** 近乎常数 (~0.05-0.06ms)，因为其计算量与 block 数（而非 token 数）成正比，block 数从 8→32 变化不大。
- **IndexTopK** 常数 ~0.056ms，bitonic sort 开销与 block 数关系较弱。
- **TFLOPs/s 随 seq_len 增加而下降**（6.7→4.9→4.3）：理论上计算量翻倍应保持 TFLOPs/s 不变，下降说明 kernel 在更长序列下受 memory-bound 影响加剧（加载更多 KV cache 数据）。
- **CUDA Profiler 数据**：`_gqa_sparse_fwd_kernel` 独占 CUDA 时间（100%），per-iter CUDA 时间从 seq_len=1024 的 323.6μs 增长到 4096 的 1.985ms，与 wall-clock 基本一致，说明 CPU launch overhead 很低。

### Decode 性能

| batch | SparseAttnDecode | IndexDecode | Total/iter | Throughput (tokens/s) |
|-------|-----------------|-------------|------------|----------------------|
| 1     | 0.105ms         | 0.170ms     | 0.275ms    | 9,518                |
| 8     | 0.105ms         | 0.170ms     | 0.275ms    | 76,317               |
| 32    | 0.105ms         | 0.170ms     | 0.275ms    | 303,435              |
| 64    | 0.107ms         | 0.171ms     | 0.278ms    | 600,758              |

**分析**：

- **SparseAttnDecode 时延 batch-invariant**：无论 batch=1 还是 64，SparseAttn 都在 ~0.105ms。原因：Split-K kernel 的 grid 取决于 `(num_kv_heads, num_splits, batch)` 但每个 block 的工作量很小且 GPU 并行度足够；真正的工作量与 batch 内总 query 数有关（即 batch * decode_qlen），qlen=1 时总量极小。
- **IndexDecode 是主要瓶颈**：0.170ms 占据了整体时延的 ~62%。index 查表+SVD 还原+topk 的固定开销在 decode 阶段无法摊销。
- **吞吐量近似线性扩展**：batch=1→64，吞吐量从 9.5k 增长到 600k tokens/s（~63x），接近理想的 64x。轻微的非线性来自 batch=64 时 SparseAttn 从 0.105→0.107ms 的微小增加。
- **CUDA Profiler 数据**：`_gqa_sparse_decode_kernel` 的 CUDA 时间从 batch=1 的 4.6μs/iter 增长到 batch=64 的 44.9μs/iter（~10x），占 CUDA 总时间的 79→93%。`_merge_topk_attn_out_kernel` 恒定 ~2.3-3.6μs/iter。

### Profiler overhead 对比

| 阶段 | 无 profiler | 有 profiler | 开销 |
|------|-----------|------------|------|
| Prefill (4096) | 1.980ms/iter | 1.997ms/iter | ~1% |
| Decode (batch=64) | 0.107ms/iter | 0.332ms/iter | ~3x |

> Decode 阶段 profiler 开销极高（~3x），因为每次 kernel launch 都需要记录 CUDA event，而 decode kernel 执行时间极短（~5-45μs），event recording 开销相对显著。**性能分析时建议关闭 profiling 获取真实吞吐**。

### 性能热力图总结

```
Prefill pipeline breakdown (seq_len=4096):
  SparseAttn  ████████████████████████████████ 94.3% (1.980ms)
  IndexScore  █ 3.0% (0.064ms)
  IndexTopK   █ 2.7% (0.056ms)

Decode pipeline breakdown (batch=64):
  IndexDecode       ██████████████████████████ 61.5% (0.171ms)
  SparseAttnDecode  ██████████████ 38.5% (0.107ms)
```

## 在远程服务器上运行

### 环境准备（目标: H100, SM90, CUDA）

```bash
# 1. 上传项目到服务器
scp -r minimax_m3_sparse_baseline user@server:/home/liuyuxin/

# 2. SSH 到服务器
ssh user@server

# 3. 进入项目目录
cd /home/liuyuxin/minimax_m3_sparse_baseline

# 4. 创建 conda 环境 (Python 3.10+)
conda create -n m3_baseline python=3.10 -y
conda activate m3_baseline

# 5. 安装 PyTorch (CUDA 12.x 对应 H100)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 6. 安装 Triton
pip install triton

# 7. 验证安装
python -c "
import torch
import triton
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name()}')
print(f'Triton: {triton.__version__}')
"
```

### 运行测试

```bash
# 正确性测试（快速验证 kernel 结果正确）
python test_correctness.py

# 快速 benchmark（5 次迭代，小配置）
python benchmark.py --mode quick

# 完整 benchmark（多种 batch/seq_len 组合）
python benchmark.py --mode full

# 带 CUDA profiler 的 benchmark
python benchmark.py --mode full --profile
```

## 关键参数说明

| 参数 | 典型值 | 含义 |
|------|--------|------|
| `num_heads` | 32 | 注意力头总数 |
| `num_kv_heads` | 8 | KV 头数 (GQA) |
| `head_dim` | 128 | 每个头的维度 |
| `topk_blocks` | 4/8 | 每个 query 选中的 KV 块数 |
| `init_blocks` | 0/1 | 强制选中的初始块数 |
| `local_blocks` | 1/3 | 强制选中的局部块数 |
| `SPARSE_BLOCK_SIZE` | 128 | 每个稀疏块大小 (= KV page size) |

## 设计说明

- **1024\*1024 Base-2 softmax**: 使用 `exp2`/`log2` 而非标准 `exp`/`log`（MiniMax M3 特有），通过整数移位操作加速 softmax 计算。
- **Paged KV cache**: KV 通过 `block_table` 进行逻辑块→物理页映射，支持变长序列和内存碎片化场景。
- **PDL (Programmatic Dependent Launch)**: H100 (SM90+) 上自动启用，允许一个 kernel 的完成直接触发下一个 kernel（无需 CPU 介入），降低 launch latency。代码通过 `common.py` 中的 compute capability 检测自动判断。
- **Split-K**: decode 阶段 query 数极少（qlen=1），传统 grid 只有 `(num_kv_heads, batch)` 两个维度，SM 利用率不足。Split-K 将 K 维度切分成多个 splits，每个 split 独立计算 partial softmax，最后通过 merge kernel 归约，显著提升 GPU 占用率。
- **Lightning Indexer**: 使用额外的 compact index-K cache（dim=128 的压缩表示）计算 block-wise 相似度分数，避免在完整 head_dim 上扫描全部 KV block，减少 prefill 阶段的选择开销。

## 许可

原始代码来自 [vllm-project/vllm](https://github.com/vllm-project/vllm)，Apache-2.0 许可。
