"""交付结果的源坐标/源块溯源查询。

清单只存逐轴 (start, step, count)、源块网格与未采块集合，不存每点三元组；
任何交付点的源坐标/源块都可由这些参数即时推出。真实缺值与未采可同值
（都等于 fill_value），但 origin 字段区分来源：'data' 为实采数据
（is_fill=True 时是真实缺值），'unsampled' 为未采块填值。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .core import MANIFEST_FORMAT


@dataclass(frozen=True)
class SourceRef:
    coord: tuple   # 逻辑源坐标 (a, b, c)
    chunk: tuple   # 逻辑源块号 (i, j, k)
    origin: str    # 'data' | 'unsampled'
    value: int     # 交付值
    is_fill: bool  # value == fill_value


class ResultIndex:
    """导出目录的可查询索引；至多驻留 2 个交付块。"""

    def __init__(self, export_dir):
        self.dir = Path(export_dir)
        m = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        if m.get("format") != MANIFEST_FORMAT:
            raise ValueError(f"不是 {MANIFEST_FORMAT} 清单")
        self.manifest = m
        self._axes = m["request"]["axes"]
        self._order = m["request"]["target_axis_order"]
        self._tcs = m["request"]["target_chunk_shape"]
        self._chunk_shape = m["source"]["chunk_shape"]
        self._fill = m["source"]["fill_value"]
        self._unsampled = {tuple(c) for c in m["source"]["unsampled_chunks"]}
        self._chunks = {tuple(c["chunk"]): c for c in m["chunks"]}
        self._cache = {}

    @property
    def chunks(self):
        """交付顺序（目标块号字典序）的块号列表。"""
        return [tuple(c["chunk"]) for c in self.manifest["chunks"]]

    def _load(self, tc):
        if tc not in self._cache:
            if len(self._cache) >= 2:
                self._cache.clear()
            self._cache[tc] = np.load(self.dir / self._chunks[tc]["file"])
        return self._cache[tc]

    def locate(self, chunk, offset):
        """交付点 -> SourceRef。chunk 为目标块号，offset 为块内线性偏移（末轴最快）。"""
        tc = tuple(chunk)
        if tc not in self._chunks:
            raise KeyError(f"目标块 {tc} 不在交付清单中")
        meta = self._chunks[tc]
        shape = tuple(meta["shape"])
        total = int(np.prod(shape))
        if not 0 <= offset < total:
            raise IndexError(f"偏移 {offset} 超出目标块 {tc} 的 {total} 个元素")
        local = tuple(int(v) for v in np.unravel_index(offset, shape))  # C 序展开

        # 目标网格坐标 -> 逻辑采样下标 -> 逻辑源坐标 -> 源块号
        u = [tc[t] * self._tcs[t] + local[t] for t in range(3)]
        i = [0, 0, 0]
        for t in range(3):
            i[self._order[t]] = u[t]
        coord = tuple(self._axes[a]["start"] + i[a] * self._axes[a]["step"]
                      for a in range(3))
        src_chunk = tuple(coord[a] // self._chunk_shape[a] for a in range(3))

        value = int(self._load_value(tc, local))
        origin = "unsampled" if src_chunk in self._unsampled else "data"
        return SourceRef(coord=coord, chunk=src_chunk, origin=origin,
                         value=value, is_fill=(value == self._fill))

    def _load_value(self, tc, local):
        return self._load(tc)[local]
