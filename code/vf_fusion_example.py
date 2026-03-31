"""
⚠️  概念示意代码，API 均未经核实
=========================================
本文件中的 pto.* API（如 pto.tadd、pto.tmul、pto.compute_region 等）
是根据 PTO IR 设计模型**推断**的示意写法，并非真实可运行的 PyPTO API。

PTOAS / pto-isa 的源码仓库目前不可公开访问，
以下示例仅用于说明"算子融合在概念上如何工作"，
不代表 PTOAS 实际存在这些融合 Pass 或这些接口。

如需了解真实 API，请参阅内部文档或联系 CANN/PTOAS 团队。
"""

# ────────────────────────────────────────────────────────────────────────────────
# 以下为概念示意，使用伪 PyPTO API 演示融合场景
# ────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────
# 示例 1：不融合（两个独立操作）
# 概念：两个 tile 操作各自持有独立的中间 tile_buf
# ─────────────────────────────────────────────
def unfused_example(a, b, c):
    """
    两个 tile 操作各自独立，中间结果 x 是独立的片上缓冲区。

    概念上生成的 Level-2 IR（示意）：
        %tmp = pto.alloc_tile : !pto.tile_buf<loc=vec, ...>
        pto.tadd ins(%a, %b) outs(%tmp)
        pto.tmul ins(%tmp, %c) outs(%result)

    对应 code/ir_examples/before_fusion.mlir（概念示意）
    """
    x = pto.tadd(a, b)          # 概念：产生独立 tile_buf
    y = pto.tmul(x, c)          # 概念：消费 x 的 tile_buf
    return y


# ─────────────────────────────────────────────
# 示例 2：如果编译器有自动融合能力（概念示意）
# 触发条件：x 只有一个 use
# ─────────────────────────────────────────────
def auto_fusion_concept(a, b, c, dst):
    """
    如果 PTOAS 存在自动融合 Pass，且 x 只有一个 use（single-use SSA value），
    则编译器**可能**消除 x 对应的中间 tile_buf，将两个 op 合并处理。

    注意："可能"不等于"确认存在"——PTOAS 是否有此 Pass 不明。
    """
    x = pto.tadd(a, b)          # 概念：单次使用 → 候选融合对象
    y = pto.tmul(x, c)          # 概念：消费 x
    pto.tstore(y, dst)
    return y


# ─────────────────────────────────────────────
# 示例 3：多个 use 阻止融合（概念示意）
# ─────────────────────────────────────────────
def multi_use_no_fusion_concept(a, b, c, dst1, dst2):
    """
    x 被两个 op 消费（multi-use），即使编译器有融合 Pass，
    也无法安全消除 x 对应的 tile_buf——因为消除后 dst1 的写回语义会改变。
    """
    x = pto.tadd(a, b)           # 有 2 个 use → 不可融合
    pto.tstore(x, dst1)          # use 1：写回 GM
    y = pto.tmul(x, c)           # use 2：参与计算
    pto.tstore(y, dst2)

