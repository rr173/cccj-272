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
