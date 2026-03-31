# PyPTO_learning
PyPTO All Docs In One

## 文档索引

| 文档 | 描述 |
|------|------|
| [PTO IR Level-1/2/3 区别](docs/pto-ir-levels.md) | 解释 PTO IR 三个抽象层次（SSA IR、DPS tile-buffer IR、低级调度 IR）的设计意图与差异 |
| [VF Fusion / 算子融合](docs/vf-fusion.md) | 说明 PyPTO→PTOAS→binary 全链路中的 VectorFusionPass 及算子融合机制，含融合前后 IR 对比 |

## 代码示例索引

| 文件 | 描述 |
|------|------|
| [code/vf_fusion_example.py](code/vf_fusion_example.py) | PyPTO Python 示例：自动融合、显式 compute_region 融合、多 use 阻止融合、关闭融合等场景 |
| [code/ir_examples/before_fusion.mlir](code/ir_examples/before_fusion.mlir) | VF Fusion Pass 执行前的 PTO Level-2 IR（tadd + tmul 独立，含中间 tile_buf） |
| [code/ir_examples/after_fusion.mlir](code/ir_examples/after_fusion.mlir) | VF Fusion Pass 执行后的 PTO Level-2 IR（tadd + tmul 融合为 fused_vec_op，中间 tile_buf 消除） |
