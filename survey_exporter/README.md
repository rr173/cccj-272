# 巡测子窗离线导出器（chunk-export）

从三轴分块整数数据中按等差子窗采样，换轴/逆序后离线交付给机具。
只支持一种公开的本地整数块格式：**NumPy `.npy` + 目录声明的单一整数 dtype**，
缺值标志为目录中的 `fill_value`。

## 数据布局与目录

逻辑数组三轴分块，末轴变化最快（C 序），边缘块可缩小。`catalog.json`：

```json
{
  "logical_shape": [100, 80, 60],
  "chunk_shape": [16, 16, 16],
  "dtype": "int32",
  "fill_value": -1,
  "max_decode_elements": 4096,
  "chunks": {
    "0,0,0": {
      "path": "chunks/c0_0_0.npy",
      "shape": [16, 16, 16],
      "file_axis_order": [2, 0, 1],
      "value_count": 4096,
      "sha256": "<64 hex>"
    }
  },
  "unsampled": [[1, 0, 0]]
}
```

- `shape`：块实际形状（逻辑轴序），必须与边缘规则一致；
- `file_axis_order`：文件轴 `j` 存放逻辑轴 `file_axis_order[j]`，逐块可不同；
- `value_count` / `sha256`：值数与文件字节摘要，开块时核验；
- `unsampled`：明确标为未采的块——产缺值且**绝不打开**；
- 既不在 `chunks` 也不在 `unsampled` 的块为**未声明块**，命中即拒绝交付。

## 请求

```json
{
  "axes": [
    {"start": 99, "step": -3, "count": 20},
    {"start": 0,  "step": 1,  "count": 8},
    {"start": 5,  "step": 2,  "count": 0}
  ],
  "target_axis_order": [2, 0, 1],
  "target_chunk_shape": [4, 8, 8]
}
```

- 逐轴 `start` / 非零有符号 `step` / 非负 `count`，严格生成等差坐标
  `start + i*step` 并校验落在 `[0, logical_shape)`；
- 任一轴 `count` 为 0 → 交付空结果，但类型与步长仍然校验（布尔、浮点、
  零步长、负点数一律拒绝）；
- `target_axis_order[t]` = 目标轴 `t` 对应的逻辑轴；
- 交付按**目标块号字典序**，块内末轴最快，边缘块仅含实际点。

## 交付布局与溯源

```
out/
  manifest.json        # 清单 + 溯源参数 + 读块核账
  chunks/t0_0_0.npy    # 目标块，字典序命名，C 序
```

清单只存逐轴 `(start, step, count)`、源块网格与未采块集合——不存每点三元组，
任何交付点的源坐标/源块都可即时推出。真实缺值与未采同值（均为
`fill_value`），但 `origin` 区分来源：`data`（实采，`is_fill=true` 即真实缺值）
或 `unsampled`（未采填值）。

```bash
# 导出
PYTHONPATH=src python3 -m chunk_export export \
    --catalog catalog.json --request req.json --out out/

# 查询交付点（块 0,1,0 内偏移 37，末轴最快）的源坐标/源块/来源
PYTHONPATH=src python3 -m chunk_export query \
    --export out --chunk 0,1,0 --offset 37
# => {"coord": [..], "chunk": [..], "origin": "data", "value": ..., "is_fill": false}
```

Python API：`Exporter(...).run()` 返回清单；`ResultIndex(out).locate(chunk, offset)`
返回 `SourceRef(coord, chunk, origin, value, is_fill)`。

## 多包逻辑合并（merge）

巡测人员拿到多份上述局部交付后，可按**逻辑坐标**合并成覆盖完整
`logical_shape` 的数据卷。输入是**有序的多个导出目录，顺序靠后的包优先**。

兼容前提（不一致即 `incompatible_packages` / 退出码 2，拒绝发布）：

- `logical_shape` 一致；
- 整数 `dtype` 一致；
- `fill_value` 一致；
- 源块网格一致（`chunk_shape` 及由其推出的 `grid_shape`）。

各包的采样范围（逐轴 start/step/count）、`target_axis_order`、
`target_chunk_shape` 允许不同——合并时统一先映回逻辑坐标再对齐。

覆盖规则：

- **只有 `origin == "data"`（实采）的点参与覆盖**；
- 实采点的值即使等于 `fill_value`，也仍是**真实缺测**，后包的真实缺测可以
  覆盖前包的实采值；
- `origin == "unsampled"` 的点一律不覆盖已有实采；
- 所有包都没有实采到的位置保持 `fill_value`，并由独立的获胜掩码判为
  `origin="unobserved"`，与真实缺测（`origin="data", is_fill=true`）区分。

结果覆盖完整逻辑形状，默认按逻辑轴序 `[0,1,2]`、用 `--target-chunk-shape`
指定的目标块形交付（也可用 `--target-axis-order` 换轴）。每个结果块
`chunks/m_a_b_c.npy` 旁有获胜输入掩码 `provenance/p_a_b_c.npy`（int32，
值为获胜输入序号，`-1` 表示未观测）。合并清单格式为 `chunk-merge/1`，
`inputs` 按顺序记录每个输入包。

```bash
# 合并（位置参数 / --inputs / 重复 --input 均可；--inputs 也支持 a,b,c）
PYTHONPATH=src python3 -m chunk_export merge out_a out_b out_c \
    --out merged --target-chunk-shape 8,8,8

# 也可用合并参数 JSON：{"inputs": [...], "out": "...", "target_chunk_shape": [8,8,8]}
PYTHONPATH=src python3 -m chunk_export merge --request merge_req.json

# 查询合并结果点：获胜输入序号/路径、源坐标、是否真实缺测
PYTHONPATH=src python3 -m chunk_export query \
    --export merged --chunk 0,0,0 --offset 37
# 或显式子命令（别名 merge_query / query-merge / mquery）
PYTHONPATH=src python3 -m chunk_export merge-query \
    --merge merged --chunk 0,0,0 --offset 37
# => {"coord":[...],"origin":"data","value":...,"is_fill":false,
#     "input_index":2,"input_path":"out_c","winner":2,
#     "source_coord":[...],"source_chunk":[...], ...}
```

Python API：

```python
from chunk_export import Merger, MergeIndex, ResultIndex

manifest = Merger(["out_a", "out_b", "out_c"], "merged",
                  target_chunk_shape=(8, 8, 8)).run()
idx = MergeIndex("merged")          # 或 ResultIndex("merged")（自动委托）
ref = idx.locate((0, 0, 0), 37)
ref.input_index      # 获胜输入序号（靠后的包优先）；未观测为 None
ref.input_path       # 获胜输入路径
ref.source_coord     # = ref.coord，逻辑源坐标
ref.origin           # 'data'（含真实缺测）| 'unobserved'
ref.is_fill          # 值是否等于 fill_value
```

保证：

- **原子发布**：先写 `merged.merge-tmp-<pid>/` 与外部暂存，全部成功后才
  `rename` 成目标目录；格式/网格不兼容、命中包损坏（不可读 / sha256 不符 /
  无法解码 / 形状类型不符）或合并失败时删除临时目录，目标路径不存在可被误用
  的半成品；CLI 不兼容退出码 2、命中包损坏退出码 3；
- **只在命中时开块**：按优先级逆序落子，已被更高优先级实采全包覆盖的输入块
  不再打开/核验，因此这种“未命中”块的损坏不会阻止发布；真正需要的块损坏
  仍按 `reason="corrupt"` 拒绝；
- 合并包目录也可再次作为 `merge` 输入（其获胜掩码 `>=0` 的点按实采处理）。

## 保证与限制

- **单次开块**：每个命中源块整个作业只打开/解码一次（远端逐开块计费），
  即使被多个目标块复用——复用走外部暂存（raw memmap）；
- **不开闲块**：没采到任何点的块绝不打开，包围盒穿过坏块也不开；
- **拒绝交付**：命中坏块（摘要/形状/类型/解码失败）、未声明块或超过
  `max_decode_elements` 的块时，报错并报**逻辑块号**，临时输出整体删除；
- **驻留上限**：同一时刻至多一个解码后的源块（受 `max_decode_elements`
  约束）+ 一个有界输出缓冲（单个目标块）；暂存与交付文件在外部存储；
- **核账**：`manifest["accounting"]` 按打开顺序记录每块的字节数与元素数，
  可审计 `open_count == 去重块数`。

实现为两阶段：① 逐源块（字典序）开一次、解码、换轴，把采样子块散射到各
目标块的暂存文件；② 按目标块号字典序从暂存落盘 `.npy` 并写清单。

## 测试

```bash
python3 -m pytest tests -q
```

夹具覆盖：逆序跳过坏块（含命中坏块拒付）、六种文件轴序、换轴与边缘块、
跨目标块复用的单次开块与字节核账、真实缺值 vs 未采的来源区分、空结果
仍验类型与步长、未声明块拒付、超解码上限拒付、CLI 导出/查询/拒付退出码。
