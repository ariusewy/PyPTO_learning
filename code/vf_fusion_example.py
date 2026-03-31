"""
VF Fusion（Vector Fusion）示例代码
演示在 PyPTO 中如何触发 / 观察 / 关闭算子融合

对应文档：docs/vf-fusion.md
参考来源：
  - https://github.com/PTO-ISA/PTOAS/blob/main/docs/PTO_IR_manual.md
  - https://github.com/PTO-ISA/pto-isa
"""

# ────────────────────────────────────────────────────────────────────────────────
# 说明：以下代码使用伪 PyPTO API（pto.*）演示融合场景。
# 真实 PyPTO 包尚未公开发布，API 以 PTO_IR_manual.md 描述为准。
# ────────────────────────────────────────────────────────────────────────────────

import pto  # type: ignore  # noqa: F401  — 真实环境需安装 PyPTO


# ─────────────────────────────────────────────
# 示例 1：不融合（两个独立操作）
# → PTOAS 生成两段独立的 Level-2 tile_buf op
# ─────────────────────────────────────────────
def unfused_example(a, b, c):
    """
    两个 tile 操作没有被放在同一 compute_region 中。
    即使 VF Fusion Pass 开启，如果 %x 有多个 use，
    或者在某些 Shape 不匹配的情况下，也不会被融合。

    生成的 Level-2 IR 片段见：code/ir_examples/before_fusion.mlir
    """
    x = pto.tadd(a, b)      # %x = pto.tadd %a, %b  → 独立 tile_buf
    y = pto.tmul(x, c)      # %y = pto.tmul %x, %c  → 独立 tile_buf
    return y


# ─────────────────────────────────────────────
# 示例 2：编译器自动识别可融合的 chain
# → %x 只有一个 use（tmul），VF Fusion Pass 自动触发
# ─────────────────────────────────────────────
def auto_fusion_example(a, b, c, dst):
    """
    x = tadd(a, b)  — x 只被 tmul 消费（single-use SSA value）
    y = tmul(x, c)  — 消费 x

    PTOAS 的 VectorFusionPass 检测到这一模式后，将两个 op 融合：
      - 中间 tile_buf (%x 对应的缓冲区) 被消除
      - 生成 pto.fused_vec_op { tadd; tmul } 内部节点
      - EmitC 阶段输出单条 C++ intrinsic 调用链

    编译命令（默认开启 VF Fusion）：
        ptoas --pto-vf-fusion input.pto -o output.ll

    融合后的 Level-2 IR 片段见：code/ir_examples/after_fusion.mlir
    """
    x = pto.tadd(a, b)          # single-use → 候选融合节点
    y = pto.tmul(x, c)          # 消费 x → 触发 VF Fusion
    pto.tstore(y, dst)          # 结果写回 GM
    return y


# ─────────────────────────────────────────────
# 示例 3：使用 compute_region 显式声明融合意图
# → 整个 region 作为一个融合单元提交给 PTOAS
# ─────────────────────────────────────────────
def explicit_fusion_example(a, b, c, dst):
    """
    compute_region 是 PyPTO 提供的融合 hint，告诉 PTOAS：
    "这些操作应该在同一个融合 kernel 中执行"。

    适用于：
    1. 多于两个算子的长链（tadd → tmul → tactivate → ...）
    2. 用户希望明确控制融合边界的场景

    PTOAS 在 Level-1 IR 中生成 pto.compute_region { ... } 块，
    ComputeRegionFusionPass 会将整块 region 作为一个 fused op 处理。
    """
    with pto.compute_region(name="add_mul_activate_fused"):
        x = pto.tadd(a, b)
        y = pto.tmul(x, c)
        z = pto.tactivate(y, mode="relu")   # tanh/sigmoid/relu 等激活函数
        pto.tstore(z, dst)
    return z


# ─────────────────────────────────────────────
# 示例 4：多个 use 阻止融合
# → x 被两个 op 消费，VF Fusion 无法消除中间 tile_buf
# ─────────────────────────────────────────────
def multi_use_no_fusion(a, b, c, d, dst1, dst2):
    """
    x = tadd(a, b) 被 tmul 和 tstore 两个 op 消费。
    由于 x 有 2 个 use，VF Fusion Pass 不会消除 %x 对应的 tile_buf。

    对应 VF Fusion 限制：
    「中间 tile_buf 有多个 use → fusion 后 use 语义改变 → 不融合」
    （见 docs/vf-fusion.md §五）
    """
    x = pto.tadd(a, b)           # x 有 2 个 use → 不可融合
    pto.tstore(x, dst1)          # use 1：写回 GM
    y = pto.tmul(x, c)           # use 2：参与计算
    pto.tstore(y, dst2)


# ─────────────────────────────────────────────
# 示例 5：关闭 VF Fusion（调试 / 性能对比）
# ─────────────────────────────────────────────
def debug_no_fusion(a, b, c, dst):
    """
    有时需要关闭 VF Fusion 来：
    1. 对比融合前后的性能差异
    2. 调试中间 IR 是否正确
    3. 验证片上内存是否被正确分配

    编译时关闭 VF Fusion：
        ptoas --no-pto-vf-fusion input.pto -o output.ll

    或在 Level-3 模式下（PlanMemory + InsertSync 均禁用）：
        ptoas --pto-level=level3 input.pto -o output.ll
        （此模式下 VF Fusion Pass 也被跳过）

    参考：docs/pto-ir-levels.md §Level-3
    """
    x = pto.tadd(a, b)
    y = pto.tmul(x, c)
    pto.tstore(y, dst)
    return y


# ─────────────────────────────────────────────
# 示例 6：GEMM + Bias + Activation 的典型融合场景
# → matmul 结果经 tadd (bias) 再经 tactivate (relu)
# ─────────────────────────────────────────────
def gemm_bias_relu_fused(A, B, bias, dst):
    """
    典型的 GEMM + Bias Add + ReLU 融合场景，在昇腾 NPU 上非常常见。

    融合链：
        tmm (矩阵乘) → tadd (加 bias) → tactivate (relu)

    注意：
    - tmm (tile matrix multiply) 使用 mat/acc 类型的 tile_buf
    - tadd/tactivate 使用 vec 类型的 tile_buf
    - 跨 loc 类型（mat/acc → vec）的融合需要一次隐式 loc 转换
    - PTOAS 的 VF Fusion Pass 在处理跨 loc 融合时会插入
      必要的 tile_buf loc 转换节点（pto.convert_tile_buf_loc）

    参考 docs/vf-fusion.md §五「VF Fusion 的限制条件」
    """
    with pto.compute_region(name="gemm_bias_relu"):
        acc = pto.tmm(A, B)                        # 矩阵乘：loc=acc
        acc_with_bias = pto.tadd(acc, bias)        # 加 bias：loc 转换 acc→vec
        result = pto.tactivate(acc_with_bias, mode="relu")  # relu：loc=vec
        pto.tstore(result, dst)
    return result
