# PyPTO_learning
PyPTO All Docs In One

## 文档索引

| 文档 | 描述 |
|------|------|
| [PTO IR Level-1/2/3 区别](docs/pto-ir-levels.md) | 解释 PTO IR 三个抽象层次（SSA IR、DPS tile-buffer IR、低级调度 IR）的设计意图与差异 |
| [VF Fusion / 算子融合](docs/vf-fusion.md) | 说明 PyPTO→PTOAS→binary 全链路中是否存在算子融合 Pass（含已知事实与推断的明确区分） |

## 代码示例索引

| 文件 | 描述 |
|------|------|
| [code/vf_fusion_example.py](code/vf_fusion_example.py) | 概念示意代码（API 未经核实）：展示算子融合在 PyPTO 层面上的语义，包含 single-use 融合、multi-use 阻止融合等场景 |
| [code/ir_examples/before_fusion.mlir](code/ir_examples/before_fusion.mlir) | 概念示意 IR：假设存在融合 Pass，融合前的 PTO Level-2 IR（含独立中间 tile_buf） |
| [code/ir_examples/after_fusion.mlir](code/ir_examples/after_fusion.mlir) | 概念示意 IR：假设存在融合 Pass，融合后的 PTO Level-2 IR（中间 tile_buf 消除） |
