# PyPTO → PTO-ISA → Binary 全链路中的算子融合（VF Fusion）

> **参考来源：**
> - [PTOAS/docs/PTO_IR_manual.md](https://github.com/PTO-ISA/PTOAS/blob/main/docs/PTO_IR_manual.md)
> - [PTO-ISA/pto-isa](https://github.com/PTO-ISA/pto-isa)
> - 代码示例：[`code/vf_fusion_example.py`](../code/vf_fusion_example.py)
> - IR 示例：[`code/ir_examples/before_fusion.mlir`](../code/ir_examples/before_fusion.mlir)、[`code/ir_examples/after_fusion.mlir`](../code/ir_examples/after_fusion.mlir)

---

## 结论速览

| 编译阶段 | 是否有融合 Pass | 融合类型 | 备注 |
|----------|----------------|----------|------|
| PyPTO（Python 前端） | ✅ 逻辑层图融合（Graph-level） | 用户手工或 Schedule 策略融合 tile 循环 | 生成更紧凑的 Level-1 IR |
| PTOAS Level-1 → Level-2 | ✅ **VF Fusion Pass**（`VectorFusionPass`） | 连续 Vector 算子融合 | 核心融合 Pass，复用 tile_buf，消除中间写回 |
| PTOAS Level-2 → Level-3 | ✅ 依赖分析驱动的隐式融合 | InsertSync 会合并可并发的流水线段 | 非独立 fusion pass，但通过事件合并实现融合效果 |
| BiSheng 编译器（C++ → binary） | ✅ 编译器级向量化与指令融合 | 昇腾硬件 intrinsics 融合 | 后端优化，不在 PTO 层可见 |

---

## 背景：为什么需要算子融合？

在昇腾 NPU 的 Tile 计算中，每个 `pto.tile_buf` 位于片上的特定 Buffer（`vec`/`mat`/`acc` 等）。如果不做融合，两个相邻的 Vector 操作（如 `tadd` 后接 `tmul`）会产生：

```
tadd → 结果写到 tile_buf_C（片上）
tmul → 从 tile_buf_C 读（片上），结果写 tile_buf_D
```

这看起来都在片上，但仍然会带来**不必要的 Buffer 占用**和**可能的流水线气泡**。VF Fusion 的目标是将这类 chain 融合为单次计算，消除中间的 tile_buf 分配。

---

## 一、PyPTO 前端层的融合（Graph-level Fusion）

PyPTO 是 Python 前端，直接构建 PTO Bytecode（对应 Level-1 IR）。在这一层，**用户可以通过 Schedule 策略控制融合粒度**。

### 1.1 不做融合时（两个独立 tile 操作）

```python
# code/vf_fusion_example.py 中 unfused_example() 函数
result_add = pto.tadd(a, b)       # 独立算子，结果是独立 SSA 值
result_mul = pto.tmul(result_add, c)  # 消费 result_add
```

生成的 Level-1 IR（见 `code/ir_examples/before_fusion.mlir`）：
```mlir
%1 = pto.tadd %a, %b : !pto.tile<16x16xf16>
%2 = pto.tmul %1, %c  : !pto.tile<16x16xf16>
```

### 1.2 用户层融合（通过 Compute Region）

PyPTO 支持将多个操作放在同一个 `compute_region` 中，编译器会将其识别为可融合的 op 集合：

```python
# code/vf_fusion_example.py 中 fused_example() 函数
with pto.compute_region():
    result_add = pto.tadd(a, b)
    result_mul = pto.tmul(result_add, c)
    pto.tstore(result_mul, output)
```

这会在 Level-1 IR 中生成 `pto.compute_region { ... }` 块，提示 PTOAS 进行整块融合。

---

## 二、PTOAS 中的 VF Fusion Pass（核心）

PTOAS 是整个链路中**融合 Pass 最集中**的地方。它基于 MLIR 的 Pass infrastructure 构建，在 Level-1 → Level-2 降层过程中包含以下与融合相关的 Pass：

### 2.1 VectorFusionPass（`--pto-vf-fusion`）

**这是最主要的融合 Pass**，对应 VF Fusion（Vector Fusion）。

**触发条件：**
- 两个相邻的 Vector 类 op（`pto.tadd`、`pto.tmul`、`pto.tactivate` 等）
- 第一个 op 的输出 tile_buf 只被第二个 op 消费（无其他 use）
- 两个 op 的 tile_buf shape/dtype/loc 兼容

**融合效果（见 `code/ir_examples/after_fusion.mlir`）：**

融合前（Level-2）：
```mlir
// 两个独立的 tile_buf，两次写、一次读
%tmp = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, ...>
pto.tadd ins(%a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%tmp : !pto.tile_buf<...>)

pto.tmul ins(%tmp, %c : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%result : !pto.tile_buf<...>)
```

融合后（VF Pass 产出）：
```mlir
// %tmp 被消除，tadd 的结果直接 in-place 传递给 tmul
// PTOAS 生成 pto.fused_vec_op 或直接 inline 计算
pto.fused_vec_op {
  pto.tadd ins(%a, %b : ...) outs(%fused_buf : ...)
  pto.tmul ins(%fused_buf, %c : ...) outs(%result : ...)
} : !pto.tile_buf<...>
```

> **注意**：`pto.fused_vec_op` 是 PTOAS 内部 IR 节点，不对外暴露；最终在 EmitC 阶段被 lowering 为单条 C++ intrinsic 调用链。

### 2.2 TileBufferReusePass（隐式融合）

在 Level-2 中，`PlanMemory` Pass 执行后会尝试对生命周期不重叠的 `tile_buf` 进行地址复用（Alias）。这等价于**通过内存共享实现的隐式融合**：

- 如果 `%tmp` 的生命周期在 `tmul` 的 `outs` 开始时已结束，则两者可以共享同一片上地址
- 从运行时角度看，等价于融合了内存访问路径

### 2.3 InsertSync Pass 对融合的影响

`InsertSync` Pass（Level-2 → Level-3）在分析流水线依赖时，会识别**可以合并的事件同步点**。对于融合后的 op 序列，由于中间 tile_buf 被消除，原本需要两对 `record_event`/`wait_event` 的位置可以合并为一对，减少同步开销。

---

## 三、PTOAS Pass Pipeline 中融合 Pass 的位置

完整的 PTOAS Pass 执行顺序（Level-1 → binary）：

```
Level-1 PTO IR
    │
    ├─ [可选] --pto-compute-region-fusion   ← 合并 compute_region 块
    │
    ├─ --pto-vf-fusion                      ← ★ VF Fusion：Vector 算子链融合
    │
    ├─ --pto-plan-memory (PlanMemory)       ← 缓冲区分配（含 TileBufferReuse）
    │   └─ TileBufferReusePass             ← ★ 隐式融合：生命周期重叠分析
    │
Level-2 PTO IR（tile_buf，有融合后的 op，无事件）
    │
    ├─ --pto-insert-sync (InsertSync)       ← 流水线依赖分析（合并同步点）
    │
Level-3 PTO IR（显式事件同步）
    │
    ├─ --pto-emitc                          ← EmitC Lowering → C++ intrinsics
    │
C++ 源文件（pto-isa intrinsics）
    │
    └─ BiSheng 编译器                        ← 硬件级指令融合（编译器后端）
NPU binary
```

---

## 四、从 PyPTO 用户视角触发 VF Fusion

在 PyPTO 中，有两种方式确保 VF Fusion 被触发：

### 方式 1：直接链式调用（编译器自动识别）

```python
# 见 code/vf_fusion_example.py: auto_fusion_example()
# 当 tadd 的输出只被 tmul 消费时，VF Fusion Pass 自动触发
x = pto.tadd(a, b)   # x 只有一个 use
y = pto.tmul(x, c)   # 消费 x
```

编译时加 `--pto-vf-fusion` flag（默认开启）：
```bash
ptoas --pto-vf-fusion input.pto -o output.ll
```

### 方式 2：使用 compute_region 显式声明融合意图

```python
# 见 code/vf_fusion_example.py: explicit_fusion_example()
with pto.compute_region(name="add_mul_fused"):
    x = pto.tadd(a, b)
    y = pto.tmul(x, c)
    pto.tstore(y, dst)
```

### 方式 3：关闭融合（调试用）

```python
# 见 code/vf_fusion_example.py: debug_no_fusion()
# 编译时关闭 VF Fusion，用于对比性能或调试中间 IR
# ptoas --no-pto-vf-fusion input.pto -o output.ll
```

---

## 五、VF Fusion 的限制条件

以下情况 VF Fusion **不会**触发：

| 限制条件 | 原因 |
|----------|------|
| 中间 tile_buf 有多个 use | fusion 后 use 语义改变 |
| 两个 op 的 tile_buf loc 不兼容 | 如 `vec` + `mat` 不可直接融合 |
| 跨越控制流边界（if/loop） | PTOAS 保守处理，不跨 region 融合 |
| 中间有显式的 `pto.tstore` | store 到 GM 后值被外部消费，不能消除 |
| Level-3 模式（`--pto-level=level3`） | 用户手动管理事件，融合 pass 可能被跳过 |

---

## 六、与 BiSheng 后端融合的分工

PTO 层的 VF Fusion 和 BiSheng 编译器的融合是**两个不同层次**的融合，不重叠：

| 层次 | 融合对象 | 融合粒度 | 可见性 |
|------|----------|----------|--------|
| PTOAS VF Fusion | `pto.tile` / `pto.tile_buf` op | Tile 粒度（16×16 块） | PTO IR 可见 |
| BiSheng 后端融合 | C++ intrinsic 调用 / SIMD 指令 | 标量/向量指令粒度 | 机器码层可见 |

---

## 延伸阅读

- [`docs/pto-ir-levels.md`](pto-ir-levels.md) — Level-1/2/3 IR 层次与降层流程
- [PTOAS README](https://github.com/PTO-ISA/PTOAS/blob/main/README.md) — PTOAS Pass 列表与编译选项
- [PTO_IR_manual.md §Fusion](https://github.com/PTO-ISA/PTOAS/blob/main/docs/PTO_IR_manual.md) — VF Fusion 正式规范
- [PTO-ISA pto-isa](https://github.com/PTO-ISA/pto-isa) — C++ intrinsics，VF Fusion 后最终调用的目标函数
- 代码示例：[`code/vf_fusion_example.py`](../code/vf_fusion_example.py)
