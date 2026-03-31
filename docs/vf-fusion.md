# PyPTO → PTO-ISA → Binary 全链路中的算子融合（VF Fusion）

> ⚠️ **重要说明**
>
> PTOAS 和 pto-isa 的源码仓库（`PTO-ISA` 组织）目前**不可公开访问**。
> 本文档只描述**可以从公开资料核实的内容**，并明确标注哪些属于推断。
> 上一版本文档中出现的 `VectorFusionPass`、`--pto-vf-fusion`、`pto.fused_vec_op`
> 等具体名称**均无法核实**，已从本文中删除。

---

## 直接回答：原本的 Pass 流程中有没有 VF Fusion？

**无法从公开资料确认。**

PTOAS 的内部 Pass 列表和实现细节**没有公开文档**。从已知的编译模型来看：

- 公开文档（`PTO_IR_manual.md` 的对外描述）只明确提到两个自动化 Pass：
  - **`PlanMemory` Pass**：Level-1 → Level-2，自动完成 `tile_buf` 分配和内存规划
  - **`InsertSync` Pass**：Level-2 → Level-3，自动分析流水线依赖并插入事件同步
- **是否存在独立的算子融合 Pass（如 VF Fusion）**，在可访问的公开资料中**没有明确说明**。

---

## 从编译模型可以推断的内容（非确认）

下面的内容是基于 PTO IR 的分层设计**合理推断**的，不代表真实实现：

### PlanMemory Pass 中可能包含 Buffer 复用

Level-1 IR 中 `pto.tile` 是 SSA 值，`PlanMemory` 将其映射为 Level-2 的 `pto.tile_buf`。
在这个映射过程中，生命周期不重叠的 tile 可以**共用同一个片上缓冲区地址**。
这是一种**内存层面的隐式优化**，效果类似于消除了中间 buffer，但这和"算子融合 Pass"是不同概念。

```
Level-1（SSA）:
  %tmp = pto.tadd %a, %b   → PlanMemory → tile_buf_slot_0
  %out = pto.tmul %tmp, %c → PlanMemory → tile_buf_slot_1

如果 %tmp 的生命周期在 tmul 执行后即结束，
PlanMemory 可以让 tile_buf_slot_0 和另一个 tile_buf 共享物理地址。
这是 buffer aliasing，不是算子融合。
```

### MLIR 基础设施支持 Fusion Pass，但 PTOAS 是否使用不明

PTOAS 基于 MLIR 构建，MLIR 本身提供了 `linalg.fuse_into_containing_op`、
`affine loop fusion` 等融合基础设施。PTOAS **可以**在其 Pass Pipeline 中加入融合 Pass，
但**是否加入、以何种形式加入**，无法从公开资料确认。

---

## 已知的 Pass 流程（仅包含公开资料明确提及的部分）

```
Level-1 PTO IR（pto.tile，SSA 值，由 PyPTO 等前端生成）
    │
    ↓  PlanMemory Pass（公开提及）
    │  · 将 pto.tile SSA 值 → pto.tile_buf 显式缓冲区
    │  · 完成片上内存地址分配
    │  · 是否包含算子融合：不明
    │
Level-2 PTO IR（pto.tile_buf，显式 DPS 风格）
    │
    ↓  InsertSync Pass（公开提及）
    │  · 分析流水线依赖（DMA / Vec / Mat 等硬件流水线之间）
    │  · 自动插入 record_event / wait_event
    │  · 是否合并/优化事件对（类融合效果）：不明
    │
Level-3 PTO IR（pto.tile_buf + 显式事件同步）
    │
    ↓  EmitC / 代码生成（公开提及）
    │  · 降层为 C++ intrinsic 调用
    │
C++ 源文件（调用 pto-isa 头文件中的 intrinsics）
    │
    ↓  BiSheng 编译器
    │  · 标准编译器优化（内联、向量化等）
    │  · 是否有专用 NPU intrinsic fusion：不明（BiSheng 内部）
    │
NPU binary
```

---

## 如何获取准确答案

如果需要确认 PTOAS 原本的 Pass 流程中是否有算子融合 Pass，建议：

1. **查阅内部文档**：`PTO_IR_manual.md`（需访问 PTO-ISA 内部仓库）
2. **运行 `ptoas --help` 或 `ptoas --print-pass-pipeline`**：查看实际注册的 Pass 列表
3. **在 PTOAS 源码中搜索关键词**：如 `fusion`、`fuse`、`VF` 等（需访问内部仓库）
4. **询问 PTOAS/CANN 团队**：这是最准确的来源

---

## 延伸阅读

- [`docs/pto-ir-levels.md`](pto-ir-levels.md) — Level-1/2/3 IR 层次与降层流程（基于公开资料）
