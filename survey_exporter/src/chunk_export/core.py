"""核心导出流程。

不变量（对应工单约束）：

- 每个命中源块单次作业只打开/解码一次，即使被多个目标块复用；
- 没采到任何点的源块绝不打开（包围盒穿过坏块也不开）；
- 同一时刻至多驻留一个解码后的源块（尺寸受 max_decode_elements 约束）；
- 输出缓冲为一个目标块；跨目标块的复用走外部暂存（raw memmap）；
- 任一阶段失败即拒绝交付：临时输出目录整体删除，不留半成品。
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np

from .catalog import Catalog
from .errors import DeliveryRefused, ExportError
from .request import Request

MANIFEST_FORMAT = "chunk-export/1"


def _ceil_div(a, b):
    return -((-a) // b)


def axis_chunk_hits(start, step, count, chunk_size, axis_len):
    """返回 [(块号, i_lo, i_hi)]：该轴采样点落入各块的下标区间（含端点）。

    坐标 coord(i) = start + i*step 关于 i 单调，故每块命中的下标是连续区间；
    未命中的块（包括坏块）不出现在结果里，随后也绝不会被打开。
    """
    if count == 0:
        return []
    first = start
    last = start + (count - 1) * step
    lo_c = min(first, last) // chunk_size
    hi_c = max(first, last) // chunk_size
    hits = []
    for c in range(lo_c, hi_c + 1):
        r_lo = c * chunk_size
        r_hi = min((c + 1) * chunk_size, axis_len) - 1  # 含端点的坐标区间
        a = r_lo - start
        b = r_hi - start
        # 求 i 使 a <= i*step <= b
        if step > 0:
            i0, i1 = _ceil_div(a, step), b // step
        else:
            i0, i1 = _ceil_div(b, step), a // step
        i0 = max(i0, 0)
        i1 = min(i1, count - 1)
        if i0 <= i1:
            hits.append((c, i0, i1))
    return hits


@dataclass
class ReadLog:
    """读块核账：按打开顺序记录，供审计与单次打开断言。"""

    entries: list = field(default_factory=list)

    def record(self, chunk, nbytes, elements):
        self.entries.append({
            "chunk": [int(c) for c in chunk],
            "bytes": int(nbytes),
            "elements": int(elements),
            "order": len(self.entries),
        })

    @property
    def total_bytes(self):
        return sum(e["bytes"] for e in self.entries)

    @property
    def opened_chunks(self):
        return [tuple(e["chunk"]) for e in self.entries]


class Exporter:
    """单次作业导出器。"""

    def __init__(self, catalog: Catalog, request: Request, out_dir, staging_dir=None):
        self.catalog = catalog
        self.request = request
        self.out_dir = Path(out_dir)
        self.staging_dir = staging_dir
        self.read_log = ReadLog()

    # ------------------------------------------------------------------ API

    def run(self):
        """执行导出，返回交付清单（dict）；拒绝交付时抛 DeliveryRefused。"""
        cat, req = self.catalog, self.request
        req.validate_range(cat.logical_shape)
        if self.out_dir.exists():
            raise ExportError(f"输出目录已存在: {self.out_dir}")

        tmp_out = self.out_dir.parent / (self.out_dir.name + f".tmp-{os.getpid()}")
        if tmp_out.exists():
            shutil.rmtree(tmp_out)
        (tmp_out / "chunks").mkdir(parents=True)

        if self.staging_dir is not None:
            staging_root = Path(self.staging_dir)
            staging_root.mkdir(parents=True, exist_ok=True)
            staging = staging_root / f"job-{os.getpid()}"
        else:
            staging = Path(tempfile.mkdtemp(prefix="chunk-export-"))
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            manifest = self._export(tmp_out, staging)
            os.rename(tmp_out, self.out_dir)  # 全部成功才落位，否则拒绝交付
            return manifest
        finally:
            shutil.rmtree(tmp_out, ignore_errors=True)
            shutil.rmtree(staging, ignore_errors=True)

    # -------------------------------------------------------------- 内部

    def _export(self, tmp_out, staging):
        cat, req = self.catalog, self.request
        q = req.target_axis_order
        tcs = req.target_chunk_shape
        counts = tuple(ax.count for ax in req.axes)
        target_grid = tuple(counts[q[t]] for t in range(3))
        target_dims = tuple(-(-g // c) for g, c in zip(target_grid, tcs))

        mmaps = {}
        if 0 not in counts:
            # 各轴命中的源块及采样下标区间；命中集合是各轴命中集合的笛卡尔积
            hits_per_axis = [
                axis_chunk_hits(ax.start, ax.step, ax.count,
                                cat.chunk_shape[a], cat.logical_shape[a])
                for a, ax in enumerate(req.axes)
            ]
            range_of = [{c: (i0, i1) for c, i0, i1 in hits} for hits in hits_per_axis]
            needed = sorted(itertools.product(*[list(r) for r in range_of]))

            # 未声明块：打开任何文件之前先拒绝，报逻辑块号
            for idx in needed:
                if idx not in cat.chunks and idx not in cat.unsampled:
                    raise DeliveryRefused(
                        f"命中未声明块 {idx}，拒绝交付", chunk=idx, reason="undeclared")

            try:
                for idx in needed:
                    ranges = [range_of[a][idx[a]] for a in range(3)]
                    if idx in cat.unsampled:
                        values = None  # 未采：不写文件，直接填缺值
                    else:
                        block = self._open_chunk(idx)  # 每块全程仅这一次打开
                        values = self._extract(block, idx, ranges)
                    # 目标轴 t 对应逻辑轴 q[t]，目标坐标即该轴采样下标
                    self._scatter(mmaps, staging,
                                  [ranges[q[t]] for t in range(3)], values)
            finally:
                for mm in mmaps.values():
                    mm.flush()

        # 按目标块号字典序交付（itertools.product 末轴变化最快）
        chunks_meta = []
        for tc in itertools.product(*[range(d) for d in target_dims]):
            mm = mmaps.get(tc)
            if mm is None:  # 网格全覆盖，逻辑上不可达
                raise ExportError(f"内部错误: 目标块 {tc} 未填充")
            shape = tuple(min(tcs[t], target_grid[t] - tc[t] * tcs[t]) for t in range(3))
            arr = np.array(mm, copy=True)  # 输出缓冲：一次一个目标块
            rel = f"chunks/t{tc[0]}_{tc[1]}_{tc[2]}.npy"
            np.save(tmp_out / rel, arr)  # C 序，块内末轴最快；边缘块仅含实际点
            chunks_meta.append({
                "chunk": list(tc),
                "shape": list(shape),
                "file": rel,
                "elements": int(np.prod(shape)),
            })

        manifest = {
            "format": MANIFEST_FORMAT,
            "source": {
                "logical_shape": list(cat.logical_shape),
                "chunk_shape": list(cat.chunk_shape),
                "dtype": str(cat.dtype),
                "fill_value": int(cat.fill_value),
                "unsampled_chunks": sorted(list(c) for c in cat.unsampled),
            },
            "request": {
                "axes": [{"start": a.start, "step": a.step, "count": a.count}
                         for a in req.axes],
                "target_axis_order": list(req.target_axis_order),
                "target_chunk_shape": list(req.target_chunk_shape),
            },
            "target_grid_shape": list(target_grid),
            "chunks": chunks_meta,
            "accounting": {
                "opened": self.read_log.entries,
                "open_count": len(self.read_log.entries),
                "bytes_read": self.read_log.total_bytes,
            },
        }
        (tmp_out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    def _open_chunk(self, idx):
        """打开并解码一个源块；整个作业中每个命中块只进这里一次。"""
        cat = self.catalog
        entry = cat.chunks[idx]
        if entry.value_count > cat.max_decode_elements:
            raise DeliveryRefused(
                f"块 {idx} 值数 {entry.value_count} 超过最大解码尺寸 "
                f"{cat.max_decode_elements}，拒绝交付",
                chunk=idx, reason="oversize")
        path = cat.chunk_path(entry)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise DeliveryRefused(f"块 {idx} 不可读: {exc}，拒绝交付",
                                  chunk=idx, reason="corrupt") from exc
        if hashlib.sha256(raw).hexdigest() != entry.sha256:
            raise DeliveryRefused(f"块 {idx} 摘要校验失败，拒绝交付",
                                  chunk=idx, reason="corrupt")
        try:
            arr = np.load(BytesIO(raw), allow_pickle=False)
        except Exception as exc:
            raise DeliveryRefused(f"块 {idx} 解码失败: {exc}，拒绝交付",
                                  chunk=idx, reason="corrupt") from exc
        expected_file_shape = tuple(entry.shape[p] for p in entry.file_axis_order)
        if (arr.dtype != cat.dtype or arr.shape != expected_file_shape
                or arr.size != entry.value_count):
            raise DeliveryRefused(
                f"块 {idx} 形状/类型与目录不符，拒绝交付", chunk=idx, reason="corrupt")
        self.read_log.record(idx, len(raw), arr.size)
        # 文件轴 j 存放逻辑轴 p[j]；转回逻辑轴序
        inv = [0, 0, 0]
        for j, p in enumerate(entry.file_axis_order):
            inv[p] = j
        return np.transpose(arr, axes=inv)

    def _extract(self, block, idx, ranges):
        """从逻辑轴序块中取出采样下标区间对应的跨步子块，并转到目标轴序。"""
        cat, req = self.catalog, self.request
        slices = []
        for a in range(3):
            i0, i1 = ranges[a]
            ax = req.axes[a]
            base = idx[a] * cat.chunk_shape[a]
            d = ax.step
            first = ax.start + i0 * d - base
            last = ax.start + i1 * d - base
            if d > 0:
                slices.append(slice(first, last + 1, d))
            elif last > 0:
                slices.append(slice(first, last - 1, d))
            else:
                slices.append(slice(first, None, d))  # stop=-1 会漏掉末元素
        view = block[tuple(slices)]  # 轴 a 上第 k 元对应采样下标 i0+k
        return np.transpose(view, axes=req.target_axis_order)

    def _scatter(self, mmaps, staging, box, values):
        """把目标轴序子块按目标块边界切分，写入各目标块的外部暂存。

        box: 目标网格坐标区间 [(lo, hi), ...]（含端点）；
        values: 与 box 同形的目标轴序数组，None 表示整块未采填缺值。
        """
        tcs = self.request.target_chunk_shape
        per_axis = [range(lo // tcs[t], hi // tcs[t] + 1)
                    for t, (lo, hi) in enumerate(box)]
        for tc in itertools.product(*per_axis):
            dst, src = [], []
            for t, (lo, hi) in enumerate(box):
                c_lo = tc[t] * tcs[t]
                s = max(lo, c_lo)
                e = min(hi, c_lo + tcs[t] - 1)
                dst.append(slice(s - c_lo, e - c_lo + 1))
                src.append(slice(s - lo, e - lo + 1))
            mm = self._mmap(mmaps, staging, tc)
            if values is None:
                mm[tuple(dst)] = self.catalog.fill_value
            else:
                mm[tuple(dst)] = values[tuple(src)]

    def _mmap(self, mmaps, staging, tc):
        mm = mmaps.get(tc)
        if mm is None:
            req = self.request
            counts = tuple(a.count for a in req.axes)
            target_grid = tuple(counts[req.target_axis_order[t]] for t in range(3))
            shape = tuple(
                min(req.target_chunk_shape[t],
                    target_grid[t] - tc[t] * req.target_chunk_shape[t])
                for t in range(3))
            path = staging / ("t%d_%d_%d.raw" % tc)
            mm = np.memmap(path, dtype=self.catalog.dtype, mode="w+", shape=shape)
            mmaps[tc] = mm
        return mm
