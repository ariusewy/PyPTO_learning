// ─────────────────────────────────────────────────────────────────────────────
// before_fusion.mlir
// VF Fusion Pass 执行前的 PTO Level-2 IR 示例
//
// 场景：tadd + tmul 两个 Vector op 串联，但尚未融合
// 对应 Python 代码：code/vf_fusion_example.py auto_fusion_example()
// 对应文档：docs/vf-fusion.md §二 VF Fusion Pass
//
// 参考来源：
//   https://github.com/PTO-ISA/PTOAS/blob/main/docs/PTO_IR_manual.md
// ─────────────────────────────────────────────────────────────────────────────

module @auto_fusion_example {
  func.func @kernel(
    %pv_a : !pto.partition_tensor_view<16x16xf16>,
    %pv_b : !pto.partition_tensor_view<16x16xf16>,
    %pv_c : !pto.partition_tensor_view<16x16xf16>,
    %pv_dst : !pto.partition_tensor_view<16x16xf16>
  ) {

    // ── 显式分配三块 tile_buf（Level-2 DPS 风格）──────────────────────────
    // PlanMemory Pass 已将 Level-1 的 SSA tile 分配为具体的片上缓冲区地址

    %a = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                         v_row=16, v_col=16, blayout=row_major,
                                         slayout=none_box, fractal=512, pad=0>

    %b = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                         v_row=16, v_col=16, blayout=row_major,
                                         slayout=none_box, fractal=512, pad=0>

    // %tmp 是 tadd 的输出缓冲区，也是 tmul 的输入缓冲区。
    // 融合前：%tmp 是独立分配的 tile_buf，占用片上内存。
    // 融合后：%tmp 将被消除（见 after_fusion.mlir）。
    %tmp = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                           v_row=16, v_col=16, blayout=row_major,
                                           slayout=none_box, fractal=512, pad=0>

    %c = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                         v_row=16, v_col=16, blayout=row_major,
                                         slayout=none_box, fractal=512, pad=0>

    %result = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                              v_row=16, v_col=16, blayout=row_major,
                                              slayout=none_box, fractal=512, pad=0>

    // ── 从 GM 加载数据到片上 ───────────────────────────────────────────────
    pto.tload ins(%pv_a : !pto.partition_tensor_view<16x16xf16>)
              outs(%a   : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                        v_row=16, v_col=16, blayout=row_major,
                                        slayout=none_box, fractal=512, pad=0>)

    pto.tload ins(%pv_b : !pto.partition_tensor_view<16x16xf16>)
              outs(%b   : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                        v_row=16, v_col=16, blayout=row_major,
                                        slayout=none_box, fractal=512, pad=0>)

    pto.tload ins(%pv_c : !pto.partition_tensor_view<16x16xf16>)
              outs(%c   : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                        v_row=16, v_col=16, blayout=row_major,
                                        slayout=none_box, fractal=512, pad=0>)

    // ── Vector 计算（融合前：两个独立 op，%tmp 作为中间缓冲区）────────────
    // op 1：tadd，结果写入 %tmp
    pto.tadd ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                         v_row=16, v_col=16, blayout=row_major,
                                         slayout=none_box, fractal=512, pad=0>,
                          !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                         v_row=16, v_col=16, blayout=row_major,
                                         slayout=none_box, fractal=512, pad=0>)
              outs(%tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                         v_row=16, v_col=16, blayout=row_major,
                                         slayout=none_box, fractal=512, pad=0>)

    // op 2：tmul，从 %tmp 读入，结果写入 %result
    // %tmp 只有这一个 use → VF Fusion 可以安全消除它
    pto.tmul ins(%tmp, %c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                           v_row=16, v_col=16, blayout=row_major,
                                           slayout=none_box, fractal=512, pad=0>,
                            !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                           v_row=16, v_col=16, blayout=row_major,
                                           slayout=none_box, fractal=512, pad=0>)
              outs(%result : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                            v_row=16, v_col=16, blayout=row_major,
                                            slayout=none_box, fractal=512, pad=0>)

    // ── 结果写回 GM ───────────────────────────────────────────────────────
    pto.tstore ins(%result : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                            v_row=16, v_col=16, blayout=row_major,
                                            slayout=none_box, fractal=512, pad=0>)
               outs(%pv_dst : !pto.partition_tensor_view<16x16xf16>)

    return
  }
}
