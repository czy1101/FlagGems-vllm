# sglang_msa_triton

MiniMax-M3 稀疏注意力算子，从 sglang 提取的独立版本。

**Triton kernel 逻辑与原 sglang 实现逐字节一致。**

## 目录结构

```
sglang_msa_triton/
├── sglang_msa/                     # 算子包（与 minimax_sparse_ops 同结构）
│   ├── __init__.py
│   ├── minimax_sparse.py          # 复合 API: minimax_sparse_prefill/decode
│   ├── common/
│   │   ├── utils.py               # bitonic sort, allocator, get_cu_seqblocks, fp8 check
│   │   └── index.py               # topk_index_reduce (GQA index head 归约)
│   ├── decode/
│   │   ├── flash_with_topk_idx.py  # Decode 索引打分 + top-k 选择 (4 Triton kernel)
│   │   └── topk_sparse.py          # Decode 稀疏 GQA attention (1 kernel + merge)
│   └── prefill/
│       ├── flash_with_topk_idx.py  # Prefill 索引打分 + top-k 选择 (2 Triton kernel)
│       └── topk_sparse.py          # Prefill 稀疏 GQA attention (1 kernel)
├── test_sglang_msa.py             # 正确性测试
├── benchmark_sglang_msa.py         # 性能基准
├── requirements.txt
└── README.md
```

## 使用

```python
from sglang_msa import (
    flash_decode_with_topk_idx,       # Decode: 索引打分 + top-k
    flash_decode_with_gqa_share_sparse, # Decode: 稀疏 GQA attention
    flash_prefill_with_topk_index,     # Prefill: 索引打分 + top-k
    flash_prefill_with_gqa_share_sparse, # Prefill: 稀疏 GQA attention
    minimax_sparse_decode,             # Decode 完整管线
    minimax_sparse_prefill,            # Prefill 完整管线
    topk_index_reduce,                 # GQA head 归约
    get_cu_seqblocks,                  # 计算 cu_seqblocks
)
```

## 运行

```bash
# 正确性测试
python test_sglang_msa.py

# 性能基准
python benchmark_sglang_msa.py

# 分步耗时
python benchmark_sglang_msa.py --per-step

# 自定义 warmup/rep
python benchmark_sglang_msa.py --warmup 20 --rep 200
```
