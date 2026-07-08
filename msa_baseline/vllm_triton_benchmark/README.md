# vLLM Triton MSA Benchmark

MiniMax M3 块稀疏注意力（Multi-head Sparse Attention）Triton 算子端到端基准测试项目。

从 vLLM 提取核心 MSA 算子，支持 Prefill 和 Decode 两阶段的正确性测试与性能基准。

## 目录结构

```
vllm_triton_benchmark/
├── test_vllm_msa.py              # vllm_msa 正确性测试（cos_sim >= 0.9999）
├── benchmark_vllm_msa.py         # vllm_msa 性能基准（Prefill + Decode）
├── pipeline.py                   # MSA 管线接口（切实现只改这里）
├── reference.py                  # PyTorch 参考实现（与实现无关）
├── vllm_msa/                     # vLLM 版 MSA 实现
│   ├── __init__.py
│   ├── index_topk.py             # Index scoring + Bitonic Top-K
│   ├── sparse_attn.py            # Block-sparse GQA attention
│   └── common/
│       └── utils.py              # 工具函数（current_platform, round_up）
├── PROJECT.md                    # 详细项目文档
├── README.md
└── requirements.txt
```

## 硬件要求

- GPU: NVIDIA H100 (SM90) 或更高

## 环境

```bash
conda activate msa-cuda
```

> 标准 Triton 编译器即可。不可使用 `gems-msa-tle`（FlagTree 修改版 autotuner 不兼容）。

## 快速开始

```bash
# 正确性
python test_vllm_msa.py

# 性能（Prefill）
python benchmark_vllm_msa.py

# 性能（Decode）
python benchmark_vllm_msa.py --decode --per-step

# 全功能
python vllm_triton_benchmark/benchmark_vllm_msa.py --per-step --with-ref --warmup 5 --rep 20
```

## 性能基准参数

| 参数 | 默认 | 说明 |
|------|:--:|------|
| `--decode` | off | Decode 模式 |
| `--decode-qlen N` | 1 | Decode 每请求 query token 数 |
| `--with-ref` | off | PyTorch 参考计时（仅 Prefill）|
| `--per-step` | off | Per-step 分解耗时 |
| `--warmup N` | 10 | 预热次数 |
| `--rep N` | 50 | 测量次数 |

## 输出列

| 列 | 条件 | 含义 |
|----|------|------|
| Total(ms) / P50(ms) | 始终 | 端到端延迟 |
| Score / TopK / Attn | Prefill + --per-step | 三步各自耗时 |
| IdxDec / AttnDec | Decode + --per-step | 两步各自耗时 |
| Ref(ms) / Speedup | Prefill + --with-ref | 参考耗时与加速比 |

## Prefill vs Decode

| | Prefill | Decode |
|--|---------|--------|
| 场景 | 批量处理多个 token | 每请求 1 个新 token |
| 并行 | 按 query token | split-K |
| 步数 | 3 | 2（融合）|
| total_q | batch × seq_len | batch × decode_qlen |

## 管线

```
Q, KV Cache, Index Cache
  → index_score / index_decode  (打分 + 选块)
  → sparse_attn / sparse_attn_decode  (块稀疏注意力)
  → output
```

## 加新实现

在项目根目录放新包（如 `official_msa/`），修改 `pipeline.py` 的 import 即可切换。`reference.py` 与实现无关，无需修改。
