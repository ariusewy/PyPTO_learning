# PTO IR 的 Level-1、Level-2、Level-3 区别

> **参考来源：** [PTOAS/docs/PTO_IR_manual.md](https://github.com/PTO-ISA/PTOAS/blob/main/docs/PTO_IR_manual.md)、[PTO-ISA/pto-isa](https://github.com/PTO-ISA/pto-isa)

---

## 背景

**PTO（Parallel Tile Operation）** 是昇腾 CANN 定义的一套面向 Tile 的虚拟指令集架构（Virtual ISA）。
**PTOAS（PTO Assembler & Optimizer）** 基于 LLVM/MLIR 构建，是连接上层 AI 框架与底层 NPU/CPU 硬件的专用编译器工具链。

PTO IR 采用**分层、多级（hierarchical multi-level）** 的中间表示栈，从高到低分为三个抽象层次：

```
Level-1（SSA-centric IR）          ← 最高抽象，由框架/前端产生
       ↓ PTO-AS buffer allocation pass
Level-2（DPS tile-buffer IR）      ← 显式缓冲区，用户/框架管理分配
       ↓ PTO-AS insert-sync pass
Level-3（Low-level scheduling IR） ← 显式事件/流水线同步，最靠近硬件
       ↓ 后端代码生成（EmitC / C++ intrinsics）
目标代码（C++ → BiSheng 编译器 → NPU 机器码）
```

> **当前状态：** `PTO_IR_manual.md` 中说明，公开文档重点覆盖 Level-2 与 Level-3；Level-1 的公开接口仍在积极设计中，将在未来版本中正式规范。

---

## Level-1：SSA-centric IR（以 SSA 值为中心的 IR）

### 核心特征

- **Tile 是 SSA 值**：操作的结果类型为 `pto.tile`，采用 SSA（Static Single Assignment）语义，像普通 MLIR 值一样传递和使用。
- **无需手动分配缓冲区**：不需要用户显式调用 `pto.alloc_tile`，缓冲区的分配和内存规划由 PTO-AS 的 Pass（`PlanMemory`）在降层过程中自动完成。
- **高层、目标无关**：是离用户最近的抽象层，易于框架集成。PyPTO、TileLang 等框架在 Python 端直接构建的 PTO Bytecode 对应这一层。

### 适用场景

- AI 框架的后端代码生成（如 PyPTO、tilelang-ascend、PTO-DSL 的高层接口）
- 用户不想关心缓冲区放置和流水线同步的场景（交由编译器自动处理）

### 示意代码风格（概念性）

```mlir
// Level-1：pto.tile 是 SSA 值，无需显式 alloc
%result = pto.tadd %a, %b : !pto.tile<16x16xf16>
```

---

## Level-2：DPS tile-buffer IR（以目的地传递风格为中心的 IR）

### 核心特征

- **Tile 是显式缓冲区对象**：操作的操作数和结果类型为 `pto.tile_buf`，采用 DPS（Destination-Passing Style），即"输出缓冲区由调用者显式传入"，而非由操作产生新的 SSA 值。
- **用户/框架显式管理缓冲区生命周期**：需要通过 `pto.alloc_tile` 声明 Tile 缓冲区，并在操作的 `ins`/`outs` 中明确指定源和目标缓冲区。
- **内存放置信息已确定**：`pto.tile_buf` 的类型中包含 `loc`（如 `vec`/`mat`/`left`/`right`/`acc`/`bias`）、数据类型、shape、valid region、layout、fractal 等完整元数据。
- **与 Level-1 的关键区别**：将"缓冲区分配"与"流水线调度"解耦。缓冲区分配是 NP-hard 问题，流水线调度也是 NP-hard 问题——Level-2 让用户/框架承担分配责任，PTO-AS Pass 专注于调度。

### 适用场景

- 对缓冲区布局有明确要求的专家级开发者
- 需要复用缓冲区以优化片上内存使用的场景
- PTO-DSL（`ptodsl/compiler/ir.py` 中 `to_ir_module` 生成的代码）默认使用此层级

### 示意代码

```mlir
// Level-2：显式分配缓冲区，使用 ins/outs DPS 风格
%a0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                     v_row=16, v_col=16, blayout=row_major,
                                     slayout=none_box, fractal=512, pad=0>
%a1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                     v_row=16, v_col=16, blayout=row_major,
                                     slayout=none_box, fractal=512, pad=0>

pto.tload ins(%pv0 : !pto.partition_tensor_view<16x16xf16>)
          outs(%a0 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, ...>)

pto.tadd ins(%a0, %a1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst  : !pto.tile_buf<...>)
```

---

## Level-3：Low-level scheduling IR（低级调度 IR）

### 核心特征

- **流水线/事件同步完全显式且由用户管理**：需要在操作之间手动插入 `pto.record_event` 和 `pto.wait_event` 来建立跨流水线的依赖关系（如 DMA 流水线与矩阵计算流水线之间的同步）。
- **直接控制执行顺序和操作间依赖**：可以精确控制昇腾 NPU 上不同硬件流水线（`PIPE_MTE2`/`PIPE_M`/`PIPE_V` 等）之间的执行顺序。
- **与 Level-2 的关键区别**：在 Level-2 基础上，禁用了 PTO-AS 的自动同步插入 Pass（`InsertSync`），改由用户手动声明事件。
- **最靠近硬件**：是 PTO IR 最低抽象层，代码生成前的最终形式。

### 适用场景

- 需要极致性能调优、手动控制流水线并发的专家级场景
- 手写高性能 Kernel（如手工 GEMM、Flash Attention 的流水线设计）
- 使用 `ptoas --pto-level=level3` 选项时，PlanMemory 和 InsertSync 均被禁用

### 示意代码

```mlir
// Level-3：手动在 DMA load 和 Vector compute 之间插入事件同步
pto.tload ins(%pv : !pto.partition_tensor_view<16x16xf16>)
          outs(%buf : !pto.tile_buf<loc=vec, ...>)

// 记录：从 GM 加载完毕，通知 Vector 计算单元
pto.record_event [#pto.pipe_event_type<EVENT_LOAD_FROM_GM>,
                  #pto.pipe_event_type<EVENT_COMPUTE_VEC>,
                  #pto.event<EVENT_ID0>]

// Vector 计算单元等待 DMA 完成
pto.wait_event [#pto.pipe_event_type<EVENT_LOAD_FROM_GM>,
                #pto.pipe_event_type<EVENT_COMPUTE_VEC>,
                #pto.event<EVENT_ID0>]

pto.tadd ins(%buf0, %buf1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%result     : !pto.tile_buf<...>)
```

---

## 三级对比总结

| 维度 | Level-1 | Level-2 | Level-3 |
|------|---------|---------|---------|
| **Tile 表示** | `pto.tile`（SSA 值） | `pto.tile_buf`（显式缓冲区，DPS） | `pto.tile_buf`（同 Level-2） |
| **缓冲区分配** | 自动（PTO-AS PlanMemory Pass） | 用户显式 `pto.alloc_tile`（无需 PlanMemory） | 用户显式管理（PlanMemory 禁用） |
| **流水线同步** | 自动（PTO-AS InsertSync Pass） | 自动（PTO-AS InsertSync Pass） | 用户手动 `record_event`/`wait_event`（InsertSync 禁用） |
| **抽象程度** | 最高 | 中等 | 最低 |
| **控制粒度** | 最粗（框架友好） | 中等（专家友好） | 最细（性能极致调优） |
| **典型用户** | PyPTO、TileLang、PTO-DSL 等框架 | 手写 Kernel 的专家开发者 | 需极致流水线优化的 Kernel 开发者 |
| **PTOAS 选项** | 默认（PlanMemory + InsertSync） | 默认（仅 InsertSync，跳过 PlanMemory） | `--pto-level=level3`（禁用 PlanMemory 和 InsertSync） |
| **公开状态** | 接口仍在设计中 | 已公开文档化 | 已公开文档化 |

---

## 降层流程

```
用户代码 / AI 框架（PyPTO、TileLang、PTO-DSL...）
    ↓ 生成
Level-1 PTO IR (.pto 文件)
    ↓ PTO-AS: PlanMemory Pass（分配 tile_buf，确定片上地址）
Level-2 PTO IR（tile_buf，无事件）
    ↓ PTO-AS: InsertSync Pass（分析流水线依赖，自动插入 record_event/wait_event）
Level-3 PTO IR（tile_buf + 显式事件同步）
    ↓ PTO-AS: 代码生成（EmitC Lowering → C++ intrinsics）
C++ 源文件（调用 pto-isa C++ 头文件中的 intrinsics）
    ↓ BiSheng 编译器
NPU 可执行文件 / .so 动态库
```

---

## 延伸阅读

- [PTOAS README](https://github.com/PTO-ISA/PTOAS/blob/main/README.md) — PTOAS 工具链构建与使用
- [PTO_IR_manual.md](https://github.com/PTO-ISA/PTOAS/blob/main/docs/PTO_IR_manual.md) — PTO IR 完整参考手册
- [PTO-ISA README（中文）](https://github.com/PTO-ISA/pto-isa/blob/main/README_zh.md) — PTO Tile Library 概览
- [PTO-ISA 虚拟 ISA 手册](https://github.com/PTO-ISA/pto-isa/blob/main/docs/PTO-Virtual-ISA-Manual_zh.md) — 虚拟 ISA 契约文档
- [PTO-DSL](https://github.com/huawei-csl/pto-dsl) — Pythonic 接口与 JIT 编译器，展示了完整的 Level-1→C++ 编译流程
