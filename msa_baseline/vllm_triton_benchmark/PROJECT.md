# vLLM MSA (Multi-head Sparse Attention) Triton Benchmark 项目详解

> 项目路径: `/home/test_dcy/MSA/vllm_triton_benchmark/`

---

## 目录

1. [项目背景](#1-项目背景)
2. [项目结构](#2-项目结构)
3. [架构设计](#3-架构设计)
4. [核心算子详解](#4-核心算子详解)
5. [Prefill vs Decode](#5-prefill-vs-decode)
6. [正确性测试](#6-正确性测试)
7. [性能基准](#7-性能基准)
8. [运行方法](#8-运行方法)
9. [Autotune 机制](#9-autotune-机制)
10. [与官方 MSA 对比](#10-与官方-msa-对比)
11. [多实现支持](#11-多实现支持)
12. [环境说明](#12-环境说明)

---

## 1. 项目背景

从 vLLM MiniMax M3 模型中提取的块稀疏注意力 Triton 算子。
核心管线：Index 打分 → Top-K 选块 → 块稀疏注意力，覆盖 Prefill 和 Decode 两阶段。

---

## 2. 项目结构

```
vllm_triton_benchmark/
│
├── test_vllm_msa.py              # vllm_msa 正确性测试（含参考实现）
├── benchmark_vllm_msa.py         # vllm_msa 性能基准
│
├── pipeline.py                   # 管线接口（切换实现只改这里）
├── reference.py                  # PyTorch 参考（与实现无关）
│
├── vllm_msa/                     # vLLM 版 MSA 实现
│   ├── __init__.py               # 公开 API
│   ├── index_topk.py             # Scoring + Bitonic Top-K（Prefill & Decode）
│   ├── sparse_attn.py            # Block-sparse GQA attention（Prefill & Decode）
│   └── common/
│       └── utils.py              # 工具函数（current_platform, round_up）
│
├── PROJECT.md / README.md        # 文档
└── requirements.txt
```

### 文件职责

| 文件 | 职责 | 耦合度 |
|------|------|:--:|
| `test_vllm_msa.py` | 正确性测试 + 参考实现 | 紧耦合 vllm_msa |
| `benchmark_vllm_msa.py` | Prefill/Decode 性能基准 | 紧耦合 vllm_msa |
| `pipeline.py` | 薄封装层 | 唯一的实现耦合点 |
| `reference.py` | PyTorch 参考实现 | 无耦合 |
| `vllm_msa/` | vLLM 版具体实现 | — |

---

## 3. 架构设计

```
┌─────────────────────────────────┐
│  test_vllm_msa.py               │  ← 直接 import vllm_msa
│  benchmark_vllm_msa.py          │  ← 直接 import vllm_msa
├─────────────────────────────────┤
│  pipeline.py   reference.py     │  ← 解耦层（未来切换实现用）
├─────────────────────────────────┤
│  vllm_msa/     (future) xxx/    │  ← 各实现版本
└─────────────────────────────────┘
```

---

## 4. 核心算子详解

### Prefill 管线（3 步）

| 步骤 | 函数 | 内核 | 说明 |
|:--:|------|------|------|
| 1 | `minimax_m3_index_score` | `_index_block_score_kernel` | Index Q @ K → block max-pool |
| 2 | `minimax_m3_index_topk` | `_topk_index_kernel` | Bitonic Sort + forced init/local |
| 3 | `minimax_m3_sparse_attn` | `_gqa_sparse_fwd_kernel` | base-2 softmax GQA |

### Decode 管线（2 步融合）

| 步骤 | 函数 | 内核链 | 说明 |
|:--:|------|------|------|
| 1 | `minimax_m3_index_decode` | 3 个 kernel | split-K 打分 + 分块 topk + 合并 |
| 2 | `minimax_m3_sparse_attn_decode` | 2 个 kernel | split-K attention + merge |

---

## 5. Prefill vs Decode

| | Prefill | Decode |
|--|---------|--------|
| 场景 | 批量处理多个输入 token | 每请求 1 个新 token |
| 并行 | 按 query token | split-K（按选中块）|
| 步数 | 3 步独立 | 2 步融合 |
| total_q | batch × seq_len | batch × decode_qlen |
| per-step 列 | Score, TopK, Attn | IdxDec, AttnDec |

---

## 6. 正确性测试

验证标准：`cos_sim >= 0.9999`（采用官方 MSA 约定）

```bash
python test_vllm_msa.py
```

参考实现（`test_vllm_msa.py` 内置）使用 PyTorch for-loop，数学完全等价但未优化。
仅验证有效位置（`~isnan & abs > 1e-8`）。

---

## 7. 性能基准

### CLI 参数

| 参数 | 默认 | 说明 |
|------|:--:|------|
| `--decode` | off | Decode 模式 |
| `--decode-qlen N` | 1 | Decode 每请求 query token 数 |
| `--with-ref` | off | PyTorch 参考计时 |
| `--per-step` | off | Per-step 分解耗时 |
| `--warmup N` | 10 | 预热次数 |
| `--rep N` | 50 | 测量次数 |

### 测试形状

在 `benchmark_vllm_msa.py` 中修改 `SHAPES` 列表，四元组 `(batch, seq_len, num_kv_heads, num_heads)`。

### KV Cache

```python
KV_DTYPES = [torch.bfloat16, torch.float8_e4m3fn]
```

### 输出列

**Prefill**: Total(ms), P50(ms), Score(ms), TopK(ms), Attn(ms), Ref(ms), Speedup
**Decode**: Total(ms), P50(ms), IdxDec(ms), AttnDec(ms)

结果追加写入 `msa_e2e_benchmark_results.csv`

---

## 8. 运行方法

```bash
conda activate msa-cuda
cd /home/test_dcy/MSA/vllm_triton_benchmark

# 正确性
python test_vllm_msa.py

# Prefill 性能
python benchmark_vllm_msa.py --per-step --with-ref

# Decode 性能
python benchmark_vllm_msa.py --decode --per-step --warmup 5 --rep 20
```

---

## 9. Autotune 机制

所有 kernel 的 autotune 功能完整可用。

| Kernel | 装饰器 | Config |
|--------|--------|:--:|
| `_topk_index_kernel` (Prefill) | `@heuristics + @autotune` | 6 个 |
| `_topk_index_partial_kernel` (Decode) | `@heuristics + @autotune` | 5 个 |
| 其他 | `@triton.jit` 或 `@heuristics` | 无 |

---

## 10. 与官方 MSA 对比

| 维度 | 本项目 | 官方 MSA |
|------|:---:|:---:|
| GPU | H100 (SM90) | Blackwell (SM100) |
| Softmax | base-2 (exp2/log2) | natural (exp/ln) |
| Index 打分 | max-pool over block | 外部 indexer |
| Top-K | Bitonic Sort, topk=32 | sparse_topk_select, topk=16 |
| Sparse Attn | Flash Attention | CuTe-DSL |
| Decode | split-K | 分页 FP8 |
| 验证 | cos_sim >= 0.9999 | cos_sim >= 0.9999 |

算法管道等价，主差异在 softmax 底数。

---

## 11. 多实现支持

### 当前实现：`vllm_msa`

```python
# pipeline.py
from vllm_msa import minimax_m3_index_score as index_score, ...
```

### 添加新实现

1. 在项目根目录创建新包（如 `official_msa/`）
2. 修改 `pipeline.py` 的 import 指向新包
3. `test_vllm_msa.py` 和 `benchmark_vllm_msa.py` 无需修改

或者创建 `test_official_msa.py` / `benchmark_official_msa.py` 直接依赖新实现。

---

## 12. 环境说明

| 环境 | Triton | Autotune | 状态 |
|------|--------|:--:|:--:|
| `msa-cuda` | 标准 3.6.0 | ✅ | 推荐 |
| `gems-msa-tle` | FlagTree 修改版 | ❌ `_topk_index_kernel` 失败 | 不兼容 |

`gems-msa-tle` 的 Triton autotuner 被 FlagGems 修改（增加 `auto_adjust_block_sizes` 钩子），
会破坏 `_topk_index_kernel` 的 `tl.static_assert(BLOCK_SIZE_K > BLOCK_SIZE_T)` 约束。
详见 `BUG_REPORT.md`。

---

*环境: Conda `msa-cuda` | Triton 3.6.0 (标准) | PyTorch 2.10.0+cu128 | H100*
