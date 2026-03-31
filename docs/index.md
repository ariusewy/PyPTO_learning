# PyPTO 学习文档

欢迎来到 PyPTO 编译器学习文档。本文档站点整合了三个子项目的中文文档，帮助你理解从 Python DSL 到 NPU 执行的完整编译流程。

## 📚 仓库地址

| 仓库 | 地址 |
|------|------|
| **学习仓库** | https://github.com/ariusewy/PyPTO_learning |
| PyPTO 原始仓库 | https://github.com/hw-native-sys/pypto.git |
| pto-as (PTOAS) 原始仓库 | https://github.com/zhangstevenunity/PTOAS.git |
| pto-isa 原始仓库 | https://gitcode.com/cann/pto-isa.git |

## 💡 问答建议

**推荐使用 GitHub Copilot 进行代码问答**：

1. 在 GitHub 仓库页面，点击右上角的 **Copilot** 图标
2. 直接用自然语言提问，例如：
   - "TADD 指令是如何实现的？"
   - "MemoryReuse Pass 的算法是什么？"
   - "解释一下 PTO IR 的 Level-1/2/3 区别"
3. Copilot 可以直接分析仓库中的代码，给出更准确的回答

> 相比于通用的 AI 助手，Copilot 能直接读取仓库代码上下文，回答更精准。

---

## 三个子项目的角色

| 项目 | 角色 | 输入 | 输出 | 核心职责 |
|------|------|------|------|----------|
| **PyPTO** | 前端编译器 | Python DSL | PTO-ISA MLIR (`.pto`) | IR 变换、tile 级优化、内存分配 |
| **pto-as** | MLIR 后端 | PTO-ISA MLIR (`.pto`) | C++ 代码（调用虚拟指令） | 布局推断、同步插入、代码发射 |
| **pto-isa** | 虚拟指令库 | — | C++ 头文件 | 虚拟指令实现（展开为 CCE intrinsics） |

**关键理解**：

- **pto-as 输出的是 C++ 代码**，其中调用的是 **pto-isa 定义的虚拟指令 API**
- **pto-isa 是纯头文件库**，在 C++ 编译时将虚拟指令展开为 CCE intrinsics
- **Bisheng** 是昇腾 CANN 的编译器框架，处理 C++ 前端、模板展开和优化
- **CCEC** 是 Bisheng 内部的 NPU 后端编译器，负责最终的硬件代码生成
- 虚拟指令通过 C++ 模板机制在 **编译时内存中** 展开，不产生额外的中间文件

---

## 编译栈层次总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            编译栈完整层次                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Layer 5: Python DSL (PyPTO)                                               │
│           └─ @pl.program, pl.matmul, pl.tile.load                          │
│           └─ 用户友好，高层抽象                                              │
│                                                                             │
│  Layer 4: PyPTO IR + 19 Passes                                             │
│           └─ 自定义不可变树 IR，编译器自动优化                                │
│                                                                             │
│  Layer 3: PTOCodegen + ptoas                                               │
│           └─ 生成 PTO-ISA MLIR → C++ 代码（调用虚拟指令）                     │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  Layer 2: 手写 PTO Kernel  ◀── kernels/manual/                             │
│           └─ 直接使用 pto::TLOAD, pto::TMATMUL 等虚拟指令                   │
│           └─ 绕过 DSL/IR/Codegen，最大控制力                                 │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  Layer 1: pto-isa 虚拟指令库                                                │
│           └─ TLOAD_TILE_IMPL → copy_gm_to_cbuf (CCE intrinsics)           │
│                                                                             │
│  Layer 0: Bisheng + CCEC 编译器                                             │
│           └─ 模板展开 → LLVM IR → NPU Binary                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 为什么前端不用 MLIR？

| 特性 | PyPTO IR（不可变树） | MLIR（可变 SSA 图） |
|------|---------------------|-------------------|
| 修改方式 | 创建新节点，旧节点不变 | 原地修改 Operation |
| Python 集成 | 简单，nanobind 即可 | 复杂，需要完整 MLIR binding |
| 调试 | 容易，节点永不变化 | 较难，需追踪状态变化 |

**关注点分离**：PyPTO 专注于 **高层 tile 级优化** （循环切分、CV 分离、内存复用），MLIR 专注于 **低层后端优化** （bufferization、memory planning、代码发射）。

### 为什么后端用 MLIR？

- **成熟的后端基础设施**：Bufferization、EmitC、Pass 管理器、类型系统
- **与其他工具链的互操作性**：PTO-ISA MLIR 可被其他 MLIR 工具处理
- **形式化的 IR 规范**：ODS 提供严格的 IR 规范，自动生成 parser/printer/verifier

---

## 完整编译流水线

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PyPTO (前端)                                    │
│                                                                             │
│  Python DSL ──AST Parse──▶ Tensor IR ──19 Passes──▶ Tile IR ──PTOCodegen──▶ │
│  (pl.load,      (ast.py)   (tensor.matmul)   (tile.load,    (pto_codegen)   │
│   pl.matmul)                               tile.matmul)                     │
│                                                    │                        │
│                                                    ▼                        │
│                                           PTO-ISA MLIR (.pto)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼ .pto 文件
┌─────────────────────────────────────────────────────────────────────────────┐
│                              pto-as (MLIR 后端)                              │
│                                                                             │
│  PTO-ISA MLIR ──InferLayout──▶ DPS IR ──PlanMemory──▶ With Sync ──EmitC──▶ │
│  (Level 1/2)    (ND/DN/NZ)    (Level 2)  (addr分配)   (SET_FLAG)  (C++代码)  │
└─────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼ C++ 代码 + pto-isa 头文件
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Bisheng + CCEC 编译器框架                               │
│                                                                             │
│  1. Bisheng C++ 前端: 模板实例化，虚拟指令 → CCE intrinsics                  │
│  2. CCEC 后端: 指令选择、寄存器分配、代码生成 → .o 目标文件                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                              NPU Binary 运行在 Ascend AICore
```

---

## 各阶段简要说明

### 阶段 1：Python DSL → Tensor IR

**发生位置**：`pypto/python/pypto/language/`

**做什么**：`@pl.program` 装饰器捕获类定义，`ast.parse()` 解析为 Python AST，`ASTParser` 构造 PyPTO IR。

**生成什么**：`ir.Program`、`ir.Function`、`ir.Call`（算子调用）、`ir.TensorType`。

### 阶段 2：19 个 Pass（IR 变换）

**发生位置**：`pypto/src/ir/transforms/`

**关键 Pass**：

| Pass | 核心变换 |
|------|----------|
| ConvertToSSA | 循环变量用 `iter_arg` 表示 |
| ConvertTensorToTileOps | `tensor.matmul` → `tile.load` + `tile.matmul` |
| InferTileMemorySpace | 推断 tile 放在哪个 buffer (Vec/Mat/Left/Right/Acc) |
| ExpandMixedKernel | InCore 函数 → AIC (Cube) + AIV (Vector) |
| MemoryReuse | 生命周期不重叠的共享 MemRef |
| AllocateMemoryAddr | 物理地址分配 |

详细说明见 [Pass 文档](pypto/dev/passes/00-pass_manager.md)。

### 阶段 3：PTOCodegen（Tile IR → PTO-ISA MLIR）

**发生位置**：`pypto/src/codegen/pto/pto_codegen.cpp`

**做什么**：遍历优化后的 IR 树，生成 PTO-ISA MLIR 文本。

**关键映射**：

| PyPTO Tile 操作 | PTO-ISA MLIR 操作 |
|-----------------|------------------|
| `tile.load` | `pto.partition_view` + `pto.tload` |
| `tile.move` | `pto.tmov` |
| `tile.matmul` | `pto.tmatmul` |

详细说明见 [Codegen 文档](pypto/dev/codegen/00-pto_codegen.md)。

### 阶段 4：ptoas（MLIR 后端 Pass）

**发生位置**：`pto-as/lib/PTO/Transforms/`

**关键 Pass**：

| Pass | 作用 |
|------|------|
| InferPTOLayout | 推断 tile 的物理布局（ND/DN/NZ fractal 格式） |
| PTOConvertToDPS | SSA tile 值 → Destination-Passing Style buffer 语义 |
| PlanMemory | buffer 物理偏移分配（精确到字节） |
| PTOInsertSync | 插入 `SET_FLAG`/`WAIT_FLAG` 流水线同步 |
| PTOToEmitC | 生成 C++ 代码 |

**输出**：C++ 代码，调用 pto-isa 虚拟指令 API。

详细说明见 [PTO IR 手册](pto-as/PTO_IR_manual.md)。

### 阶段 5：pto-isa + Bisheng/CCEC

虚拟指令通过 C++ 模板机制在编译时展开：

```
pto::TLOAD_TILE_IMPL(tile, global_tensor)
       │
       ▼ 模板实例化 (Bisheng C++ 前端)
    copy_gm_to_cbuf(dst, src, sid, nBurst, lenBurst, ...)
       │
       ▼ CCEC 后端
    NPU 二进制指令编码 (.o)
```

**不产生中间文件**，全部在编译器内存中完成。

详细说明见下方 [Bisheng 与 CCEC 编译器详解](#bisheng-与-ccec-编译器详解)。

---

## 两种编译策略对比

| 方面 | Default (PTO 路径) | CCE 路径 |
|------|-------------------|----------|
| **流程** | PyPTO → .pto → ptoas → .cpp | PyPTO → CCECodegen → .cpp |
| **中间格式** | PTO-ISA MLIR | 无 |
| **同步插入** | ptoas | PyPTO |
| **布局推断** | ptoas | PyPTO |
| **适用场景** | 新架构，精细流水线控制 | 兼容旧代码路径 |

---

## Bisheng 与 CCEC 编译器详解

### 两个编译器的关系

| 方面 | Bisheng (框架层) | CCEC (后端层) |
|------|------------------|---------------|
| **层级** | 编译器框架/驱动 | 内部后端 |
| **触发** | `bisheng` 命令 | `bisheng -xcce` 自动启用 |
| **输入** | C++ 源码 + PTO 虚拟指令 | LLVM IR |
| **输出** | LLVM IR | Ascend NPU 目标代码 (.o) |
| **主要工作** | C++ 前端、模板展开、LLVM 优化 | 指令选择、寄存器分配、代码生成 |

### 编译命令示例

```bash
# 编译 NPU Kernel
bisheng -c -xcce -O2 --cce-aicore-only \
  --cce-aicore-arch=dav-c310-vec \     # 目标架构 (A3)
  -std=c++17 \
  --cce-enable-pto-passes \             # 启用 PTO Auto Mode
  kernel.cpp -o kernel.o
```

**关键编译选项**：

| 参数 | 作用 |
|------|------|
| `-xcce` | 启用 CCEC 后端，编译为 NPU 代码 |
| `--cce-aicore-only` | 只生成 AICore 代码 |
| `--cce-aicore-arch=` | 目标架构：`dav-c220-vec`(A2), `dav-c310-vec`(A3) |
| `--cce-enable-pto-passes` | 启用 PTO Auto Mode（自动 buffer 分配和同步） |

### CCEC 后端流水线

```
1. CCE IR 生成：LLVM IR → CCE 中间表示，内存层级建模
2. 指令选择与调度：CCE intrinsics → 硬件指令编码，流水线调度
3. 寄存器分配与地址转换：AICore 寄存器、UB 地址分配
4. 代码生成：生成 .o 目标文件
```

### 查看中间结果

```bash
bisheng -xcce -E kernel.cpp -o kernel.i          # 预处理后的代码
bisheng -xcce -S -emit-llvm kernel.cpp -o kernel.ll  # LLVM IR
bisheng -xcce -S kernel.cpp -o kernel.s           # 汇编代码
bisheng -xcce --save-temps kernel.cpp -o kernel.o # 保留所有中间文件
```

---

## 手写 PTO Kernel

### 在编译栈中的位置

手写 Kernel 位于 **Layer 2**，直接使用 pto-isa 的虚拟指令 API，**绕过了 PyPTO 的 DSL、IR 和 Codegen 层**。

**优势**：完全手动控制流水线、手动调参（qkPreloadNum, FIFO depth）、最优性能。

**劣势**：需理解硬件流水线，开发效率低。

### 示例：Flash Attention

以 `kernels/manual/a2a3/flash_atten` 为例：

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│   QK    │────▶│    P    │────▶│   PV    │────▶│   GU    │
│ (Cube)  │     │ (Vec)   │     │ (Cube)  │     │ (Vec)   │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
     │               │               │               │
     ▼               ▼               ▼               ▼
qk_tile_fifo     p_tile_fifo    pv_tile_fifo     o_out
```

详细说明见 [Flash Attention 文档](kernels/manual/a2a3/flash_atten/README_zh.md)。

### JIT vs AOT 编译

| 方面 | AOT | JIT |
|------|-----|-----|
| **编译时机** | 预先编译 | 运行时编译 |
| **首次延迟** | 无 | 有 |
| **灵活性** | 固定参数 | 动态 shape |
| **集成方式** | AscendCL API | Python ctypes |
| **示例** | `kernels/manual/` | `demos/torch_jit/` |

---

## 硬件内存模型

```
┌─────────────────────────────────────────┐
│           GM (Global Memory)            │  DDR/HBM，几十 GB
└───────────────────┬─────────────────────┘
                    │ TLOAD / TSTORE
                    ▼
┌─────────────────────────────────────────┐
│              L1 (MAT)                   │  512 KB
└───────────────────┬─────────────────────┘
                    │ TMOV
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│  L0A (LEFT)   │       │  L0B (RIGHT)  │  各 64 KB
└───────┬───────┘       └───────┬───────┘
        └───────────┬───────────┘
                    │ TMATMUL (Cube 单元)
                    ▼
           ┌───────────────┐
           │   L0C (ACC)   │  128-256 KB
           └───────────────┘
```

**流水线执行**：

```
时间 ──────────────────────────────────────────────────────────────▶

PIPE_MTE2:  ═══TLOAD══════════════════════════════════════════════
                 │ SET_FLAG
PIPE_MTE1:  ─────WAIT_FLAG═══TMOV══════════════════════════════════
                                 │ SET_FLAG
PIPE_M:     ────────────────────WAIT_FLAG═══TMATMUL════════════════
                                                │ SET_FLAG
PIPE_MTE3:  ───────────────────────────────────WAIT_FLAG═══TSTORE═══
```

---

## 文档导航

### PyPTO 文档

| 模块 | 说明 |
|------|------|
| [Pass 文档](pypto/dev/passes/00-pass_manager.md) | 19 个 IR 变换 Pass 的详细说明 |
| [IR 文档](pypto/dev/ir/00-overview.md) | IR 层级结构、类型系统、操作定义 |
| [Codegen 文档](pypto/dev/codegen/00-pto_codegen.md) | PTO Codegen 和 CCE Codegen |
| [Language 文档](pypto/dev/language/00-python_syntax.md) | Python DSL 语法参考 |
| [参考文档](pypto/reference/pto-isa/00-cluster_architecture.md) | Cluster 架构、TPUSH/TPOP、Buffer 管理 |

### pto-as 文档

| 文档 | 说明 |
|------|------|
| [PTO IR 手册](pto-as/PTO_IR_manual.md) | PTO-ISA MLIR 方言定义 |
| [TPUSH/TPOP 设计](pto-as/designs/ptoas-tpush-tpop-design.md) | 跨核通信机制 |
| [无 NPU 编译指南](pto-as/no_npu_compile_only_guide_zh.md) | 仅编译模式 |

### pto-isa 文档

| 章节 | 说明 |
|------|------|
| [概述](pto-isa/01-overview_zh.md) | PTO-ISA 整体介绍 |
| [机器模型](pto-isa/02-machine-model_zh.md) | Ascend NPU 硬件架构 |
| [Tile 与 GlobalTensor](pto-isa/04-tiles-and-globaltensor_zh.md) | 内存层级与数据结构 |
| [同步机制](pto-isa/05-synchronization_zh.md) | SET_FLAG/WAIT_FLAG 同步 |
| [指令集](pto-isa/07-instructions_zh.md) | 完整指令参考 |
| [编程指南](pto-isa/08-programming_zh.md) | NPU 编程实践 |
