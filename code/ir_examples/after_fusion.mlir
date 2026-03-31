// ─────────────────────────────────────────────────────────────────────────────
// after_fusion.mlir
// ⚠️  概念示意 IR，并非从真实 PTOAS 工具链产出
//
// 此文件展示：假设编译器存在算子融合能力，融合 Pass 执行后的
// PTO Level-2 IR 可能是什么样子（中间 tile_buf 被消除）。
//
// `pto.fused_vec_op` 是此处自创的示意节点名称，
// 并非 PTOAS 中已确认存在的 IR Op。
// 真实 PTOAS 的 Pass 名称、IR 语法细节以内部文档为准。
// ─────────────────────────────────────────────────────────────────────────────

module @auto_fusion_example_after_vf_fusion {
  func.func @kernel(
    %pv_a   : !pto.partition_tensor_view<16x16xf16>,
    %pv_b   : !pto.partition_tensor_view<16x16xf16>,
    %pv_c   : !pto.partition_tensor_view<16x16xf16>,
    %pv_dst : !pto.partition_tensor_view<16x16xf16>
  ) {

    // ── 显式分配缓冲区（%tmp 已被 VF Fusion 消除）────────────────────────
    // 融合前有 5 个 alloc_tile（a, b, tmp, c, result）
    // 融合后只有 4 个（a, b, c, result）— %tmp 对应的 alloc 被消除

    %a = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                         v_row=16, v_col=16, blayout=row_major,
                                         slayout=none_box, fractal=512, pad=0>

    %b = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
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

    // ── 融合后的 Vector 计算（pto.fused_vec_op）──────────────────────────
    //
    // VF Fusion Pass 将原来的两个独立 op：
    //   pto.tadd ins(%a, %b) outs(%tmp)
    //   pto.tmul ins(%tmp, %c) outs(%result)
    // 合并为一个 pto.fused_vec_op block。
    //
    // 融合效果：
    //   - %tmp 缓冲区被消除，片上内存节省 16x16xf16 = 512 Bytes
    //   - 两个 Vector 指令序列合并为一个 intrinsic 调用链
    //   - EmitC 阶段将此块 lowering 为：
    //       pto_tadd(a_ptr, b_ptr, fused_buf_ptr);
    //       pto_tmul(fused_buf_ptr, c_ptr, result_ptr);
    //     （编译器可能进一步合并为 pto_add_mul 复合 intrinsic，取决于 pto-isa 版本）
    //
    pto.fused_vec_op {
      // 内部节点 1：tadd，结果写入 fused 内部临时 buf（不再是独立 alloc_tile）
      pto.tadd ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                           v_row=16, v_col=16, blayout=row_major,
                                           slayout=none_box, fractal=512, pad=0>,
                            !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                           v_row=16, v_col=16, blayout=row_major,
                                           slayout=none_box, fractal=512, pad=0>)
               outs(%fused_internal_buf : !pto.tile_buf<loc=vec, dtype=f16,
                                           rows=16, cols=16, v_row=16, v_col=16,
                                           blayout=row_major, slayout=none_box,
                                           fractal=512, pad=0>)

      // 内部节点 2：tmul，消费 fused 内部 buf，结果写入最终 %result
      pto.tmul ins(%fused_internal_buf, %c :
                   !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                  v_row=16, v_col=16, blayout=row_major,
                                  slayout=none_box, fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                  v_row=16, v_col=16, blayout=row_major,
                                  slayout=none_box, fractal=512, pad=0>)
               outs(%result : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                             v_row=16, v_col=16, blayout=row_major,
                                             slayout=none_box, fractal=512, pad=0>)
    }  // end pto.fused_vec_op

    // ── 结果写回 GM ───────────────────────────────────────────────────────
    pto.tstore ins(%result : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                            v_row=16, v_col=16, blayout=row_major,
                                            slayout=none_box, fractal=512, pad=0>)
               outs(%pv_dst : !pto.partition_tensor_view<16x16xf16>)

    return
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 对比：融合前 vs 融合后
//
// 融合前（before_fusion.mlir）：
//   alloc_tile × 5（a, b, tmp, c, result）
//   tload × 3
//   tadd（写 %tmp）
//   tmul（读 %tmp，写 %result）
//   tstore × 1
//
// 融合后（after_fusion.mlir）：
//   alloc_tile × 4（a, b, c, result）    ← 减少 1 个 tile_buf 分配
//   tload × 3
//   fused_vec_op { tadd; tmul }          ← 2 个 op 合并为 1 个 fused block
//   tstore × 1
//
// 性能收益：
//   - 片上内存节省：1 × 16×16×f16 = 512 Bytes
//   - Vector 流水线无中间写-读气泡
//   - InsertSync 事件同步点从 2 对减少到 1 对（Load→VEC 依赖链缩短）
// ─────────────────────────────────────────────────────────────────────────────
