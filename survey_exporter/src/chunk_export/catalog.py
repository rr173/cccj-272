"""目录（catalog）加载与校验。

目录描述一个三轴逻辑数组的分块布局：

- ``logical_shape``       逻辑轴长（三元正整数）
- ``chunk_shape``         标称块形（三元正整数，边缘块可缩小）
- ``dtype``               唯一定整数 dtype（如 ``int32``）
- ``fill_value``          缺值标志（落在 dtype 范围内的整数）
- ``max_decode_elements`` 单块最大解码尺寸（元素数）
- ``chunks``              已采块表，键 ``"i,j,k"`` 为逻辑块号，值含
                          ``path`` / ``shape``（逻辑轴序实际形状）/
                          ``file_axis_order``（文件轴 j 对应逻辑轴 p[j]）/
                          ``value_count`` / ``sha256``
- ``unsampled``           明确标为未采的块号列表

文件内末轴变化最快（C 序）。未列于 ``chunks`` 且未列入 ``unsampled``
的块为未声明块，命中即拒绝交付。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import CatalogInvalid

_CHUNK_KEY = re.compile(r"^(\d+),(\d+),(\d+)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _int_triplet(value, what):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise CatalogInvalid(f"{what} 必须是三元整数列表")
    out = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, int):
            raise CatalogInvalid(f"{what} 含非整数: {v!r}")
        out.append(int(v))
    return tuple(out)


@dataclass(frozen=True)
class ChunkEntry:
    index: tuple            # 逻辑块号 (i, j, k)
    relpath: str            # 相对目录根的文件路径
    shape: tuple            # 逻辑轴序下的实际形状（边缘块已缩小）
    file_axis_order: tuple  # 文件轴 j 存放逻辑轴 file_axis_order[j]
    value_count: int        # 值数（== prod(shape)）
    sha256: str             # 文件字节摘要


@dataclass(frozen=True)
class Catalog:
    root: Path
    logical_shape: tuple
    chunk_shape: tuple
    dtype: np.dtype
    fill_value: int
    max_decode_elements: int
    chunks: dict            # (i,j,k) -> ChunkEntry
    unsampled: frozenset    # {(i,j,k), ...}
    grid_shape: tuple       # 每轴块数

    def chunk_region(self, index):
        """块 index 覆盖的逻辑坐标区间 ((lo, hi), ...)（hi 开区间）。"""
        return tuple(
            (c * cs, min((c + 1) * cs, ls))
            for c, cs, ls in zip(index, self.chunk_shape, self.logical_shape)
        )

    def chunk_path(self, entry):
        return self.root / entry.relpath


def load_catalog(path):
    """加载并校验目录；不一致即抛 CatalogInvalid。"""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CatalogInvalid(f"目录文件不可读: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CatalogInvalid(f"目录不是合法 JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise CatalogInvalid("目录顶层必须是对象")

    logical_shape = _int_triplet(raw.get("logical_shape"), "logical_shape")
    chunk_shape = _int_triplet(raw.get("chunk_shape"), "chunk_shape")
    if any(v <= 0 for v in logical_shape):
        raise CatalogInvalid("logical_shape 必须为正整数")
    if any(v <= 0 for v in chunk_shape):
        raise CatalogInvalid("chunk_shape 必须为正整数")

    dtype_name = raw.get("dtype")
    if not isinstance(dtype_name, str):
        raise CatalogInvalid("缺少 dtype 字段")
    try:
        dtype = np.dtype(dtype_name)
    except TypeError as exc:
        raise CatalogInvalid(f"不支持的 dtype: {dtype_name!r}") from exc
    if not np.issubdtype(dtype, np.integer):
        raise CatalogInvalid(f"仅支持一种公开整数块格式（npy + 整数 dtype），得到 {dtype}")

    fill_value = raw.get("fill_value")
    if isinstance(fill_value, bool) or not isinstance(fill_value, int):
        raise CatalogInvalid("fill_value 必须是整数")
    info = np.iinfo(dtype)
    if not (info.min <= fill_value <= info.max):
        raise CatalogInvalid(f"fill_value {fill_value} 超出 {dtype} 范围")

    max_decode = raw.get("max_decode_elements")
    if isinstance(max_decode, bool) or not isinstance(max_decode, int) or max_decode <= 0:
        raise CatalogInvalid("max_decode_elements 必须是正整数")

    grid_shape = tuple(-(-ls // cs) for ls, cs in zip(logical_shape, chunk_shape))

    chunks = {}
    raw_chunks = raw.get("chunks", {})
    if not isinstance(raw_chunks, dict):
        raise CatalogInvalid("chunks 必须是对象")
    for key, spec in raw_chunks.items():
        m = _CHUNK_KEY.match(key)
        if not m:
            raise CatalogInvalid(f"非法块键: {key!r}")
        index = tuple(int(g) for g in m.groups())
        if any(i >= g for i, g in zip(index, grid_shape)):
            raise CatalogInvalid(f"块 {key} 超出块网格 {grid_shape}")
        if not isinstance(spec, dict):
            raise CatalogInvalid(f"块 {key} 描述必须是对象")

        shape = _int_triplet(spec.get("shape"), f"块 {key} 的 shape")
        expected = tuple(
            min((i + 1) * cs, ls) - i * cs
            for i, cs, ls in zip(index, chunk_shape, logical_shape)
        )
        if shape != expected:
            raise CatalogInvalid(
                f"块 {key} 声明形状 {shape} 与边缘规则推出的 {expected} 不符")

        order = _int_triplet(spec.get("file_axis_order"), f"块 {key} 的 file_axis_order")
        if sorted(order) != [0, 1, 2]:
            raise CatalogInvalid(f"块 {key} 的 file_axis_order 不是轴排列: {order}")

        value_count = spec.get("value_count")
        if isinstance(value_count, bool) or not isinstance(value_count, int):
            raise CatalogInvalid(f"块 {key} 的 value_count 必须是整数")
        if value_count != int(np.prod(shape)):
            raise CatalogInvalid(
                f"块 {key} 的 value_count {value_count} != prod(shape) {int(np.prod(shape))}")

        sha256 = spec.get("sha256")
        if not isinstance(sha256, str) or not _SHA256.match(sha256):
            raise CatalogInvalid(f"块 {key} 的 sha256 非法")

        relpath = spec.get("path")
        if not isinstance(relpath, str) or not relpath:
            raise CatalogInvalid(f"块 {key} 缺少 path")
        p = Path(relpath)
        if p.is_absolute() or ".." in p.parts:
            raise CatalogInvalid(f"块 {key} 路径非法: {relpath!r}")

        chunks[index] = ChunkEntry(
            index=index, relpath=relpath, shape=shape,
            file_axis_order=order, value_count=value_count, sha256=sha256)

    unsampled = set()
    raw_unsampled = raw.get("unsampled", [])
    if not isinstance(raw_unsampled, list):
        raise CatalogInvalid("unsampled 必须是列表")
    for item in raw_unsampled:
        idx = _int_triplet(item, "unsampled 项")
        if any(i < 0 or i >= g for i, g in zip(idx, grid_shape)):
            raise CatalogInvalid(f"未采块 {idx} 超出块网格 {grid_shape}")
        if idx in chunks:
            raise CatalogInvalid(f"块 {idx} 既已采又标未采")
        unsampled.add(idx)

    return Catalog(
        root=path.parent,
        logical_shape=logical_shape,
        chunk_shape=chunk_shape,
        dtype=dtype,
        fill_value=int(fill_value),
        max_decode_elements=max_decode,
        chunks=chunks,
        unsampled=frozenset(unsampled),
        grid_shape=grid_shape,
    )
