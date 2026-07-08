# Bug Report: FlagTree-modified Triton `auto_adjust_block_sizes` breaks `_topk_index_kernel`

> 日期: 2026-07-08 | 最后更新: 2026-07-08 | 状态: 根因已确认

---

## 1. Bug 概述

`_topk_index_kernel` 在 FlagGems/FlagTree 修改过的 Triton 编译器（`gems-msa-tle` 环境）中，autotune 功能失效，编译阶段抛出：

```
CompileTimeAssertionFailure: at 23:4:
    tl.static_assert(BLOCK_SIZE_K > BLOCK_SIZE_T)
```

**根因：** FlagTree 新增的 `analyze_kernel_dependencies` 静态分析将 `BLOCK_SIZE_K` 错误映射到 `init_blocks`（值为 1），触发 `adjust_block_size_tl_load` 将全部 6 个 config 的 `BLOCK_SIZE_K` 缩减为 1，导致 `1 > 32 (BLOCK_SIZE_T)` 为 False。

标准 Triton（`msa-cuda` 环境）无此功能，一切正常。

---

## 2. 环境对比

| | gems-msa-tle | msa-cuda |
|--|:--:|:--:|
| Triton | 3.6.0 (FlagTree 修改) | 3.6.0 (标准) |
| PyTorch | 2.10.0+cu128 | 2.10.0+cu128 |
| CUDA | 12.8 | 12.8 |
| autotuner.py | 10 处 FlagTree 修改 | 0 处修改 |
| adjust_kernel_param.py | 1340 行（FlagTree 新增） | 不存在 |

---

## 3. 排查过程

### 3.1 初步排除

| 检查项 | 结论 |
|--------|------|
| `next_power_of_2(32)` | 正常返回 32 |
| autotuner 传入的 `BLOCK_SIZE_T` | 确认为 32 |
| 6 个 autotune config 的 `BLOCK_SIZE_K` | 全部 ≥ 64，全部 > 32 |
| 磁盘缓存干扰 | 清除后仍失败 |
| 跨环境对比 | `msa-cuda` 正常，`gems-msa-tle` 失败 |
| 其他 autotune kernel | `_topk_index_partial_kernel` (Decode) 在 FlagTree 下正常 |

### 3.2 决定性证据：`auto_adjust_block_sizes` 将所有 BLOCK_SIZE_K 改为 1

#### 3.2.1 为什么之前的 monkey-patch 误判了

之前 monkey-patch `_bench` 打印的是 `config.kwargs["BLOCK_SIZE_K"]`。但 FlagTree autotuner 中实际传给 kernel 的是 `current` 字典（`dict(meta, **config.all_kwargs())`），而非 `config` 对象。**`auto_adjust_block_sizes` 修改的是 `current`，原 `config` 对象不受影响。** 因此打印 config 看到的是原值 2048，但 kernel 实际收到的是被修改后的值。

#### 3.2.2 直接调用 `auto_adjust_block_sizes`，确凿证据

在 FlagTree 环境下，手动构造与 autotuner 完全相同的参数直接调用：

```python
nargs = {topk: 32, init_blocks: 1, ...}, BLOCK_SIZE_T=32
# 6 个 config 的 BLOCK_SIZE_K: 2048, 1024, 512, 256, 128, 64
auto_adjust_block_sizes(nargs, jit_fn, configs, current, config)
```

**实测输出：**

```
BLOCK_SIZE_K: 2048 ->    1  *** MODIFIED ***
BLOCK_SIZE_K: 1024 ->    1  *** MODIFIED ***
BLOCK_SIZE_K:  512 ->    1  *** MODIFIED ***
BLOCK_SIZE_K:  256 ->    1  *** MODIFIED ***
BLOCK_SIZE_K:  128 ->    1  *** MODIFIED ***
BLOCK_SIZE_K:   64 ->    1  *** MODIFIED ***
```

全部 6 个 config 的 BLOCK_SIZE_K 被修改为 1 → `1 > 32` 为 False → static_assert 失败。

#### 3.2.3 黑盒验证：关闭 `auto_adjust_block_sizes` 后正常

FlagTree 通过 `FLAGTREE_AABS` 环境变量（默认 `True`）控制此功能：

```python
# triton/knobs.py:377
adjust_block_size: env_bool = env_bool("FLAGTREE_AABS", True)
```

设置 `FLAGTREE_AABS=0` 后，正确性测试通过：

```bash
FLAGTREE_AABS=0 python test_vllm_msa.py
# Test Passed! cos_sim = 0.999998 >= 0.9999
```

**因果链闭环。**

#### 3.2.4 排除 AOT 编译假说

AOT 假说认为 FlagTree 预编译时 BLOCK_SIZE_T 为未解析占位符。但 `_topk_index_partial_kernel`（同结构：heuristics + autotune + jit + static_assert）在 FlagTree 下正常 → 假说被排除。

### 3.3 故障传播链

```
autotuner._bench() 被调用 6 次（每个 config 一次）
  │
  ├─ current = dict(meta={BLOCK_SIZE_T:32}, **config.all_kwargs())
  │     → current = {BLOCK_SIZE_K: 2048, BLOCK_SIZE_T: 32, ...}
  │     → 注意：config 对象本身未修改，只有 current 被修改
  │
  ├─ analyze_kernel_dependencies(jit_fn)
  │     → load_map = {'BLOCK_SIZE_K': 'init_blocks'}   ← 误判根源
  │
  ├─ adjust_block_size_tl_load(bs_name="BLOCK_SIZE_K",
  │       ts_name="init_blocks", min_bs=1)
  │     → bs=2048, ts=nargs["init_blocks"]=1
  │     → if bs > ts: 2048 > 1 → True
  │     → updated_bs = next_power_of_2(1) = 1
  │     → update_bs(..., current, "BLOCK_SIZE_K", 1)
  │     → current = {BLOCK_SIZE_K: 1, BLOCK_SIZE_T: 32, ...}
  │
  ├─ kernel_call() → self.fn.run(**current)
  │     → 编译: tl.static_assert(1 > 32) → ❌ CompileTimeAssertionFailure
  │
  └─ 异常被 _bench 的 try-except 捕获 → rett = [inf, inf, inf]
       │
       └─ 6 个 config 全部 inf → autotuner 被迫选一个 → 最终调用仍失败
```

---

## 4. 精确根因定位

3 个 FlagTree 新增函数在 `triton/runtime/adjust_kernel_param.py` 中，共约 1340 行代码。

### ① `analyze_kernel_dependencies`（FlagTree 新增）

- 静态分析 kernel IR，构建 `load_map` 字典
- 对 `_topk_index_kernel`，输出 `load_map = {'BLOCK_SIZE_K': 'init_blocks'}`
- **问题**：将 `BLOCK_SIZE_K`（排序 tile 尺寸，应在 64-2048 间）错误映射到 `init_blocks`（值为 1 的强制初始块计数参数）

### ② `auto_adjust_block_sizes`（第 1283 行）

- 入口函数，看到 `load_map` 非空，遍历调用 `adjust_block_size_tl_load`

### ③ `adjust_block_size_tl_load`（第 1171 行）— 直接肇事函数

```python
def adjust_block_size_tl_load(nargs, current, config, bs_name, ts_name, min_bs=1):
    bs = current[bs_name]               # BLOCK_SIZE_K = 2048
    ts = nargs[ts_name]                 # init_blocks = 1
    if bs > ts:                         # 2048 > 1 → True
        updated_bs = next_power_of_2(ts) # next_power_of_2(1) = 1
        if updated_bs < min_bs:
            updated_bs = min_bs          # 1 < 1 → False
        update_bs(..., bs_name, updated_bs)  # 将 BLOCK_SIZE_K 改为 1
```

**逻辑**：如果 block size 大于对应的 tensor size，就将其缩减。但映射关系错误导致按 `init_blocks=1` 缩减了 `BLOCK_SIZE_K`。

**触发条件**：`FLAGTREE_AABS=True`（默认），设置为 `0` 即跳过。

---

## 5. FlagTree 对 autotuner.py 的 10 处修改

| 行号 | 改动 | 类别 |
|:--:|------|------|
| 16 | `from .adjust_kernel_param import auto_adjust_block_sizes` | 新增导入 |
| 37-38 | `self.shared_config_pre_hook` | 新增属性 |
| 99 | `self.seen_tuned_metas = {}` | 去重缓存 |
| 147-163 | `auto_adjust_block_sizes()` + seen_tuned_metas 去重 | **根因** |
| 185 | `rett = self.do_bench(...)` | 返回值处理 |
| 189-192 | `seen_tuned_metas` 存储 | 去重缓存 |
| 237 | `self.seen_tuned_metas = {}` | 重置 |
| 286-289 | `pruned_configs = copy.deepcopy(self.configs)` | 深拷贝 |

其中第 147-163 行为根因，其余 9 处为辅助逻辑。

---

## 6. 受影响范围

| Kernel | FlagTree | 标准 Triton |
|--------|:--:|:--:|
| `_topk_index_kernel` (Prefill Top-K) | ❌ | ✅ |
| `_topk_index_partial_kernel` (Decode Top-K) | ✅ | ✅ |
| 其他 kernel（score/attn 等） | ✅ | ✅ |

---

## 7. 解决方案

**推荐**：使用标准 Triton 环境 `msa-cuda`。

**在 FlagTree 环境中**：设置 `FLAGTREE_AABS=0` 关闭 `auto_adjust_block_sizes`：

```bash
FLAGTREE_AABS=0 python test_vllm_msa.py
FLAGTREE_AABS=0 python benchmark_vllm_msa.py
```

或在脚本 `import vllm_msa` 之前 monkey-patch：

```python
import triton.knobs
triton.knobs.autotuning.adjust_block_size = False
```

---

## 8. 定位状态

- [x] 确认问题仅在 FlagTree 修改版 Triton 中出现
- [x] 直接调用 `auto_adjust_block_sizes` 确认 BLOCK_SIZE_K → 1
- [x] 追踪到 `adjust_block_size_tl_load` 为直接肇事函数
- [x] 追踪到 `analyze_kernel_dependencies` 错误映射为根因
- [x] 排除 AOT 编译假说
- [x] 黑盒验证 `FLAGTREE_AABS=0` 修复问题
- [ ] `_topk_index_partial_kernel` 为何不受影响（不影响根本结论）

---

*报告基于 Triton 3.6.0 两个环境的实际对比分析。*
