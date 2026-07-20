# FlagGems-vLLM MSA

面向 H100（SM90）的 MiniMax M3 MSA 算子、测试与 benchmark。核心实现位于
`src/flaggems_vllm/ops/MSA`，保留原 `triton_h100_v0` 的 vLLM Paged KV
Cache 接口，不依赖 vLLM runtime 即可测试本实现，并提供 vLLM、sglang
两种不同接入方式的对比。

vLLM 和 sglang baseline 都是可选的：如果同级仓库或指定路径不可用，benchmark
会打印提示并自动退化为 FlagGems 独跑；它们可用时仍按原逻辑完成对比。

## 与旧 triton_h100 的关键区别

旧目录使用 continuous K/V；本目录使用分页缓存：

```text
kv_cache:       [num_blocks, 2, 128, num_kv_heads, head_dim]
index_kv_cache: [num_blocks, 128, head_dim]
block_table:    [batch, max_blocks]
```

逻辑块 `block` 通过 `page = block_table[request, block]` 找到物理页，K/V
分别位于 `kv_cache[page, 0]` 和 `kv_cache[page, 1]`。

## 文件

- `src/flaggems_vllm/ops/MSA/index_topk.py`、`sparse_attn.py`：MSA Triton 算子。
- `tests/test_MSA/ref_torch.py`、`test_correctness.py`：Paged KV 的 PyTorch
  参考和端到端 prefill/decode 正确性测试。测试会随机打散物理页，覆盖真实
  block table 寻址。
- `bench_vs_vllm.py`：直接加载 vLLM 源码中的算子文件，不触发完整 vLLM
  runtime 导入。
- `bench_vs_sglang.py`：按 sglang 的 `req_to_token + slot_ids` 接口接入。
- `bench_compare.py`：本实现、vLLM、sglang 三列统一对比。

两种 adapter 都会在第一次 warmup 调用时完成布局/元数据准备并缓存，正式计时只
覆盖各自 MSA pipeline，不把 cache 格式转换重复计入每次算子延迟。

## 服务器运行

从 `FlagGems-vllm` 仓库根目录以模块方式运行：

```bash
python -m pytest tests/test_MSA/test_correctness.py

# 只测本实现，先确认环境与 shape
python -m benchmark.test_MSA.bench_vs_vllm --no-vllm --shape 8,1024,8,48 --per-step

# vLLM 对比；路径指向 vLLM 源码根目录
# 如果 vLLM 不存在，会自动退化为 FlagGems 独跑
python -m benchmark.test_MSA.bench_vs_vllm --vllm-path /path/to/vllm-main --all-shapes

# sglang 对比；默认使用 FlagGems-vllm 同级的 sglang/python
# 如果 sglang 不存在，会自动退化为 FlagGems 独跑
python -m benchmark.test_MSA.bench_vs_sglang --all-shapes

# 三方统一表格
PYTHONPATH=/path/to/sglang/python:$PYTHONPATH \
python -m benchmark.test_MSA.bench_compare \
  --vllm-path /path/to/vllm-main --all-shapes

# Decode
python -m benchmark.test_MSA.bench_compare --decode --all-shapes \
  --vllm-path /path/to/vllm-main
```

sglang 的高层 decode 接口一次只接受每个 request 一个 query token，因此三方
decode 对比要求 `--decode-qlen 1`。测试 speculative decode（`qlen > 1`）时可用
vLLM 对比并传 `--no-sglang`。

## 本地与服务器验证边界

Triton 编译、数值正确性和 H100 性能必须在服务器执行。建议先跑正确性测试，
全部通过后再采集 benchmark；性能测试默认 BF16、head dim 128、warmup 5、
repeat 30。
