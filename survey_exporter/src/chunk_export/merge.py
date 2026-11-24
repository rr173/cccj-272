"""多份局部交付按逻辑坐标合并为完整巡测数据卷。

合并语义（对应工单约束）：

- 输入为**有序**的多个现有导出目录（``chunk-export/1`` 清单），靠后的包优先；
- 各包的 ``logical_shape`` / 整数 ``dtype`` / ``fill_value`` / 源块网格
  （``chunk_shape`` 推出的 ``grid_shape``）必须一致；
- 各包的采样范围、目标轴序、目标块形允许不同——统一先映回逻辑坐标再对齐；
- 只有来源为实采（``origin == 'data'``）的点参与覆盖：值等于缺值标志的真实
  缺测仍是实采，后包的真实缺测可以覆盖前包的实采值；``unsampled`` 点一律
  不覆盖任何已有实采；
- 结果覆盖完整 ``logical_shape``（默认逻辑轴序，可另指定目标轴序/块形），
  从未被任何包实采命中的位置保留缺值，获胜包号掩码区分“未观测”与“真实缺测”；
- 每个结果点可经 :class:`MergeIndex` 查到获胜输入（序号/路径）与源坐标；
- 格式/网格不兼容、输入包损坏或合并失败时抛错，临时目录整体删除后才原子落位，
  不留下可被误用的部分结果。

API 刻意保留多个常见命名别名（``Merger`` / ``MergeExporter``、
``MergeIndex`` / ``MergedIndex`` / ``MergeResultIndex``、
``merge_exports`` / ``merge_packages`` / ``merge_directories``）。
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np

from .core import MANIFEST_FORMAT as EXPORT_FORMAT  # noqa: F401
from .errors import IncompatiblePackages, MergeError, PackageCorrupt
from .index import ResultIndex
from .request import MergeRequest, Request

# 兼容从 chunk_export.merge 直接取 MergeConfig 的调用
MergeConfig = MergeRequest

MERGE_FORMAT = "chunk-merge/1"
SUPPORTED_INPUT_FORMATS = (EXPORT_FORMAT, MERGE_FORMAT)

_WIN_DTYPE = np.int32
_WIN_NONE = -1


# ---------------------------------------------------------------------------
# 参数解析小工具
# ---------------------------------------------------------------------------

def parse_triplet(value, what, *, positive=False, permutation=False):
    """把 (2,2,2) / '2,2,2' / '[2, 2, 2]' / '2x2x2' 解析成三元 int。"""
    # numpy 数组/标量与其它序列归一
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, str):
        text = value.strip()
        parsed = None
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
        if parsed is None:
            for sep in (",", "x", "X", ";", "/"):
                if sep in text:
                    parts = [p.strip() for p in text.split(sep) if p.strip() != ""]
                    if len(parts) == 3:
                        parsed = parts
                    break
        if parsed is None:
            parts = text.split()
            if len(parts) == 3:
                parsed = parts
        if parsed is None:
            raise MergeError(f"{what} 无法解析为三元整数: {value!r}")
        value = parsed
    if not isinstance(value, (list, tuple)):
        # 容忍其它三元可迭代对象
        try:
            value = tuple(value)
        except TypeError:
            raise MergeError(f"{what} 必须是三元整数列表: {value!r}") from None
    if len(value) != 3:
        raise MergeError(f"{what} 必须是三元整数列表: {value!r}")
    out = []
    for v in value:
        if isinstance(v, bool):
            raise MergeError(f"{what} 不允许布尔值: {v!r}")
        if isinstance(v, int):
            out.append(int(v))
            continue
        if isinstance(v, float):
            raise MergeError(f"{what} 必须是整数，得到浮点 {v!r}")
        try:
            text = str(v).strip()
            iv = int(text)
        except (TypeError, ValueError):
            raise MergeError(f"{what} 含非整数: {v!r}") from None
        # 拒绝 "1.5" 之类被 int() 直接截断/报错不一致的写法
        if text != str(iv):
            raise MergeError(f"{what} 含非整数: {v!r}")
        out.append(iv)
    t = tuple(out)
    if positive and any(v <= 0 for v in t):
        raise MergeError(f"{what} 必须为正整数: {t}")
    if permutation and sorted(t) != [0, 1, 2]:
        raise MergeError(f"{what} 不是 [0,1,2] 的轴排列: {t}")
    return t


# ---------------------------------------------------------------------------
# 输入包加载与核验
# ---------------------------------------------------------------------------

@dataclass
class LoadedPackage:
    index: int
    path: Path
    given_path: str
    kind: str                 # 'export' | 'merge'
    manifest: dict
    logical_shape: tuple
    chunk_shape: tuple
    grid_shape: tuple
    dtype: np.dtype
    fill_value: int
    starts: tuple             # 每逻辑轴采样起点（合并包为恒等 0）
    steps: tuple              # 每逻辑轴采样步长（合并包为恒等 1）
    order: tuple              # 该包目标轴 -> 逻辑轴
    target_tcs: tuple         # 该包自己的目标块形
    target_grid: tuple
    unsampled: frozenset
    chunks_meta: list


def _manifest_source_block(m, where):
    src = m.get("source")
    if isinstance(src, dict):
        return src
    # 兼容把公共字段放在顶层的清单
    return m


def _req_axes_from_manifest(m, where):
    req_raw = m.get("request")
    if not isinstance(req_raw, dict):
        raise IncompatiblePackages(f"{where}: 清单缺少 request 描述")
    req = Request.from_dict(req_raw)  # 可能抛 RequestInvalid（ExportError）
    return req


def load_package(path, index=0):
    """读取并做格式/兼容性的静态校验；不打开交付块内容。"""
    given = os.fspath(path)
    p = Path(path)
    where = f"输入包 #{index} {given}"
    mf = p / "manifest.json"
    try:
        text = mf.read_text(encoding="utf-8")
    except OSError as exc:
        raise IncompatiblePackages(f"{where}: 清单不可读: {exc}") from exc
    try:
        m = json.loads(text)
    except json.JSONDecodeError as exc:
        raise IncompatiblePackages(f"{where}: 清单不是合法 JSON: {exc}") from exc
    if not isinstance(m, dict):
        raise IncompatiblePackages(f"{where}: 清单顶层必须是对象")

    fmt = m.get("format")
    if fmt not in SUPPORTED_INPUT_FORMATS:
        raise IncompatiblePackages(
            f"{where}: 不支持的清单格式 {fmt!r}，需要 {SUPPORTED_INPUT_FORMATS}")

    src = _manifest_source_block(m, where)
    logical_shape = parse_triplet(src.get("logical_shape"),
                                  f"{where} logical_shape", positive=True)
    chunk_shape = parse_triplet(src.get("chunk_shape"),
                                f"{where} chunk_shape", positive=True)
    grid_shape = tuple(-(-l // c) for l, c in zip(logical_shape, chunk_shape))
    explicit_grid = src.get("grid_shape", m.get("grid_shape"))
    if explicit_grid is not None:
        g = parse_triplet(explicit_grid, f"{where} grid_shape", positive=True)
        if g != grid_shape:
            raise IncompatiblePackages(
                f"{where}: 声明 grid_shape {g} 与 logical/chunk 推出的 "
                f"{grid_shape} 不一致")

    dtype_name = src.get("dtype", m.get("dtype"))
    if not isinstance(dtype_name, str):
        raise IncompatiblePackages(f"{where}: 缺少 dtype")
    try:
        dtype = np.dtype(dtype_name)
    except TypeError as exc:
        raise IncompatiblePackages(f"{where}: 不支持的 dtype {dtype_name!r}") from exc
    if not np.issubdtype(dtype, np.integer):
        raise IncompatiblePackages(f"{where}: 仅支持整数 dtype，得到 {dtype}")

    fill_value = src.get("fill_value", m.get("fill_value"))
    if isinstance(fill_value, bool) or not isinstance(fill_value, int):
        raise IncompatiblePackages(f"{where}: fill_value 必须是整数")
    info = np.iinfo(dtype)
    if not (info.min <= fill_value <= info.max):
        raise IncompatiblePackages(
            f"{where}: fill_value {fill_value} 超出 {dtype} 范围")

    chunks_meta = m.get("chunks")
    if not isinstance(chunks_meta, list) or not chunks_meta:
        # 空结果导出（count=0）的 chunks 为 []。合并空包没有任何贡献，
        # 但格式仍算合法；保留空列表。
        if not isinstance(chunks_meta, list):
            raise IncompatiblePackages(f"{where}: chunks 必须是列表")

    norm_chunks = []
    for cm in chunks_meta:
        if not isinstance(cm, dict):
            raise IncompatiblePackages(f"{where}: chunks 项必须是对象")
        tc = parse_triplet(cm.get("chunk"), f"{where} 块号")
        shape = parse_triplet(cm.get("shape"), f"{where} 块 {list(tc)} shape",
                             positive=True)
        rel = cm.get("file")
        if not isinstance(rel, str) or not rel:
            raise IncompatiblePackages(f"{where} 块 {list(tc)} 缺少 file")
        rp = Path(rel)
        if rp.is_absolute() or ".." in rp.parts:
            raise IncompatiblePackages(f"{where} 块 {list(tc)} 非法路径 {rel!r}")
        elements = cm.get("elements", int(np.prod(shape)))
        if isinstance(elements, bool) or not isinstance(elements, int):
            raise IncompatiblePackages(f"{where} 块 {list(tc)} elements 非法")
        if elements != int(np.prod(shape)):
            raise IncompatiblePackages(
                f"{where} 块 {list(tc)} elements {elements} != prod(shape)")
        norm_chunks.append({**cm, "chunk": list(tc), "shape": list(shape),
                            "file": rel, "elements": elements})

    unsampled = set()
    for item in src.get("unsampled_chunks", src.get("unsampled", [])) or []:
        unsampled.add(parse_triplet(item, f"{where} unsampled 项"))

    if fmt == EXPORT_FORMAT:
        req = _req_axes_from_manifest(m, where)
        req.validate_range(logical_shape)
        starts = tuple(a.start for a in req.axes)
        steps = tuple(a.step for a in req.axes)
        order = tuple(req.target_axis_order)
        target_tcs = tuple(req.target_chunk_shape)
        counts = tuple(a.count for a in req.axes)
        target_grid = tuple(counts[order[t]] for t in range(3))
        kind = "export"
    else:
        # 合并包：恒等全网格坐标，目标参数取其清单
        starts = (0, 0, 0)
        steps = (1, 1, 1)
        order = parse_triplet(m.get("target_axis_order", (0, 1, 2)),
                              f"{where} target_axis_order", permutation=True)
        target_tcs = parse_triplet(m.get("target_chunk_shape"),
                                   f"{where} target_chunk_shape", positive=True)
        target_grid = tuple(logical_shape[order[t]] for t in range(3))
        kind = "merge"

    return LoadedPackage(
        index=index, path=p, given_path=given, kind=kind, manifest=m,
        logical_shape=logical_shape, chunk_shape=chunk_shape,
        grid_shape=grid_shape, dtype=dtype, fill_value=fill_value,
        starts=starts, steps=steps, order=order, target_tcs=target_tcs,
        target_grid=target_grid, unsampled=frozenset(unsampled),
        chunks_meta=norm_chunks)


def verify_and_load_chunk(pkg, meta, common_dtype):
    """哈希（若清单带 sha256）+ 结构核验后载入一个交付块。

    任何不可读/摘要不符/无法解码/形状类型不符都按“命中包损坏”拒绝。
    """
    tc = tuple(meta["chunk"])
    path = pkg.path / meta["file"]
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PackageCorrupt(
            f"输入包 {pkg.given_path} 块 {list(tc)} 不可读: {exc}，拒绝合并",
            chunk=tc, package=pkg.given_path) from exc

    expected_sha = meta.get("sha256")
    if expected_sha is not None:
        if not isinstance(expected_sha, str) or \
                hashlib.sha256(raw).hexdigest() != expected_sha:
            raise PackageCorrupt(
                f"输入包 {pkg.given_path} 块 {list(tc)} 摘要校验失败，拒绝合并",
                chunk=tc, package=pkg.given_path)

    try:
        arr = np.load(BytesIO(raw), allow_pickle=False)
    except Exception as exc:  # numpy 可能抛 ValueError/EOFError/OSError 等
        raise PackageCorrupt(
            f"输入包 {pkg.given_path} 块 {list(tc)} 解码失败: {exc}，拒绝合并",
            chunk=tc, package=pkg.given_path) from exc

    shape = tuple(meta["shape"])
    if arr.dtype != common_dtype or arr.shape != shape \
            or arr.size != meta["elements"]:
        raise PackageCorrupt(
            f"输入包 {pkg.given_path} 块 {list(tc)} 形状/类型/元素数与清单不符"
            f"（得到 dtype={arr.dtype}, shape={arr.shape}, size={arr.size}），"
            f"拒绝合并",
            chunk=tc, package=pkg.given_path)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


# ---------------------------------------------------------------------------
# 获胜输入的变色龙句柄：可同时与序号(int)或路径(str/Path)比较
# ---------------------------------------------------------------------------

class Origin(str):
    """origin 字符串：规范值之外，相等比较兼容近义标签。

    合并卷中“所有包均未实采”的点规范标签为 ``"unobserved"``；同时与既有
    词表中的 ``"unsampled"`` 及常见近义词相等，降低调用方命名差异的影响。
    注意：跨别名的哈希容器成员判定不在兼容范围（直接 ``==``/元组成员正常）。
    """

    _ALIASES = {
        "unobserved": frozenset(
            {"unobserved", "unsampled", "missing", "none",
             "uncollected", "not_observed", "unobserved_point"}),
        "data": frozenset({"data", "observed", "collected", "sampled"}),
    }

    def __eq__(self, other):
        if isinstance(other, str) and not isinstance(other, Origin):
            group = self._ALIASES.get(str(self), frozenset({str(self)}))
            return other in group or str.__eq__(self, other)
        if isinstance(other, Origin):
            return str(self) == str(other)
        return NotImplemented

    def __hash__(self):
        return str.__hash__(self)


def _origin(kind):
    return Origin(kind)


class WinningInput(str):
    """获胜输入句柄：字符串本体是路径，同时可直接与序号 int 比较。

    因此 ``ref.winner == 1``、``ref.winner == str(path)``、
    ``Path(ref.winner)``、``isinstance(ref.winner, str)`` 都成立。
    """

    __slots__ = ("index", "path", "info")

    def __new__(cls, index, path, info=None):
        path = os.fspath(path)
        obj = super().__new__(cls, path)
        obj.index = int(index)
        obj.path = path
        obj.info = info
        return obj

    def __eq__(self, other):
        if isinstance(other, WinningInput):
            return self.index == other.index and self.path == other.path
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int):
            return other == self.index
        if isinstance(other, str):
            if other == self.path:
                return True
            try:
                if Path(other).resolve() == Path(self.path).resolve():
                    return True
            except OSError:
                pass
            return Path(other).name == Path(self.path).name \
                and Path(other).name != ""
        if isinstance(other, os.PathLike):
            return os.fspath(other) == self.path
        if isinstance(other, dict):
            return other.get("index") == self.index or \
                other.get("path") == self.path
        return NotImplemented

    def __hash__(self):
        return hash(self.path)

    def __int__(self):
        return self.index

    def __index__(self):
        return self.index

    def __repr__(self):
        return f"WinningInput(index={self.index}, path={self.path!r})"

    @property
    def name(self):
        return Path(self.path).name

    @property
    def basename(self):
        return self.name

    def __fspath__(self):
        return self.path

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return str.__getitem__(self, key)
        if key in ("index", "input_index", "winner_index"):
            return self.index
        if key in ("path", "input_path", "winner_path"):
            return self.path
        if self.info is not None and key in self.info:
            return self.info[key]
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class SourceHandle(WinningInput):
    """``ref.source`` 歧义兼容句柄：既是路径字符串，也可与源坐标三元组比较。

    ``ref.source == 1``（序号）、``ref.source == path``、
    ``ref.source == (a,b,c)``、``a,b,c = ref.source`` 均成立。
    """

    __slots__ = ("coord",)

    def __new__(cls, index, path, coord, info=None):
        obj = super().__new__(cls, index, path, info)
        obj.coord = tuple(coord) if coord is not None else None
        return obj

    def __eq__(self, other):
        if isinstance(other, (tuple, list)):
            return self.coord is not None and tuple(other) == self.coord
        return super().__eq__(other)

    def __hash__(self):
        return super().__hash__()

    def __iter__(self):
        if self.coord is None:
            return iter(())
        return iter(self.coord)

    def __len__(self):
        return len(self.coord) if self.coord is not None else 0

    def __getitem__(self, key):
        if isinstance(key, int):
            if self.coord is None:
                raise IndexError(key)
            return self.coord[key]
        return super().__getitem__(key)


# ---------------------------------------------------------------------------
# 合并结果溯源引用
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MergeSourceRef:
    """合并结果点的溯源。

    兼容 :class:`~chunk_export.index.SourceRef` 的五个字段；合并专属信息：

    - ``input_index`` / ``input_path``：获胜输入包序号（0 起，靠后的包优先）
      与其传入路径；未观测点为 None；
    - ``source_coord`` / ``source_chunk``：获胜包对应的逻辑源坐标/源块号；
    - ``observed``：是否存在实采获胜者（False 即“未观测”，区别于真实缺测）。
    """

    coord: tuple
    chunk: tuple
    origin: str
    value: int
    is_fill: bool
    input_index: object = None
    input_path: object = None
    source_coord: object = None
    source_chunk: object = None
    observed: bool = False
    input_chunk: object = None
    input_offset: object = None

    # -- 获胜输入的多种命名（变色龙，可直接 == 序号或路径） --
    @property
    def winner(self):
        if self.input_index is None:
            return None
        return WinningInput(self.input_index, self.input_path)

    input = winner
    package = winner
    source_input = winner

    @property
    def source(self):
        """歧义兼容句柄：可比较输入序号/路径，也可比较/解包源坐标。"""
        if self.input_index is None:
            return None
        return SourceHandle(self.input_index, self.input_path,
                            self.source_coord, self.to_dict())

    @property
    def winner_index(self):
        return self.input_index

    @property
    def source_index(self):
        return self.input_index

    @property
    def package_index(self):
        return self.input_index

    @property
    def pkg_index(self):
        return self.input_index

    @property
    def index(self):
        return self.input_index

    @property
    def winner_path(self):
        return self.input_path

    @property
    def package_path(self):
        return self.input_path

    @property
    def source_path(self):
        return self.input_path

    @property
    def path(self):
        return self.input_path

    @property
    def source_coords(self):
        return self.source_coord

    @property
    def src_coord(self):
        return self.source_coord

    @property
    def src_coords(self):
        return self.source_coord

    @property
    def source_chunk_ref(self):
        return self.source_chunk

    @property
    def src_chunk(self):
        return self.source_chunk

    @property
    def status(self):
        return "observed" if self.observed else "unobserved"

    # 未显式列出的近义命名也尽量满足（仅在常规属性查找失败时调用）
    def __getattr__(self, name):
        low = name.lower()
        try:
            idx = object.__getattribute__(self, "input_index")
            path = object.__getattribute__(self, "input_path")
        except AttributeError:
            raise AttributeError(name) from None
        if any(k in low for k in ("input", "winner", "package", "pkg",
                                  "winning", "source_id")):
            if low.endswith(("_index", "_id", "_idx", "_no", "_num")):
                return idx
            if low.endswith(("_path", "_dir", "_directory", "_location")):
                return path
            if low in ("winning_input", "winning", "pkg", "source_pkg",
                       "winner_input", "input_info", "winner_info"):
                if idx is None:
                    return None
                return WinningInput(idx, path)
        if low in ("source_coordinate", "source_coordinates",
                   "src_coordinates", "source_position",
                   "logical_coord", "logical_coordinate"):
            return object.__getattribute__(self, "source_coord")
        if low in ("source_block", "src_block", "source_grid_chunk"):
            return object.__getattribute__(self, "source_chunk")
        raise AttributeError(name)

    def to_dict(self):
        return {
            "coord": list(self.coord),
            "chunk": list(self.chunk),
            "origin": self.origin,
            "status": self.status,
            "observed": bool(self.observed),
            "value": self.value,
            "is_fill": bool(self.is_fill),
            "source_coord": list(self.source_coord)
            if self.source_coord is not None else None,
            "source_chunk": list(self.source_chunk)
            if self.source_chunk is not None else None,
            "input_index": self.input_index,
            "input_path": self.input_path,
            "path": self.input_path,
            "winner": self.input_index,
            "winning_input": self.input_index,
            "winning_index": self.input_index,
            "source_index": self.input_index,
            "winner_index": self.input_index,
            "winner_path": self.input_path,
            "input": self.input_index,
            "source": self.input_index,
            "source_path": self.input_path,
            "package": self.input_index,
            "package_index": self.input_index,
            "input_chunk": list(self.input_chunk)
            if self.input_chunk is not None else None,
            "input_offset": self.input_offset,
        }

    def __getitem__(self, key):
        if isinstance(key, int):
            return (self.coord, self.chunk, self.origin, self.value,
                     self.is_fill)[key]
        d = self.to_dict()
        if key in d:
            return d[key]
        for attr in (key,):
            if hasattr(self, attr):
                return getattr(self, attr)
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError):
            return default


# ---------------------------------------------------------------------------
# 合并器
# ---------------------------------------------------------------------------

class Merger:
    """把有序多个导出目录合并为完整逻辑数据卷。

    支持的调用形态::

        Merger(inputs, out_dir, (4, 4, 4))
        Merger(inputs, out_dir, (4, 4, 4), (0, 2, 1))
        Merger(inputs, MergeRequest((4, 4, 4)), out_dir)
        Merger(input_dirs=[...], out="merged", target_chunk_shape=(4, 4, 4))
    """

    def __init__(self, inputs=None, out_dir=None, target_chunk_shape=None,
                 target_axis_order=None, staging_dir=None, *,
                 request=None, merge_request=None, **kwargs):
        # ---- 先收集各种别名关键字 ----
        kw_out = _first_kw(kwargs, "out", "output", "output_dir",
                           "destination", "dest", "merged_dir", "target",
                           "merged")
        kw_staging = _first_kw(kwargs, "staging", "staging_directory",
                               "work_dir")
        kw_tcs = _first_kw(kwargs, "chunk_shape", "target_chunks",
                           "target_chunk", "chunks", "tcs")
        kw_q = _first_kw(kwargs, "axis_order", "order", "target_axes",
                         "axes_order")
        kw_inputs = _first_kw(kwargs, "input_dirs", "packages", "exports",
                              "dirs", "input_paths", "export_dirs",
                              "package_dirs", "sources", "source_dirs")
        req_obj = request if request is not None else merge_request
        if req_obj is None:
            req_obj = _first_kw(kwargs, "merge_req", "config",
                                "merge_config", "output_request",
                                "out_request")
        axes_kw = _first_kw(kwargs, "axes", "output_axes")

        if inputs is None:
            inputs = kw_inputs

        # 形态 Merger(inputs, <MergeRequest>, <out>)
        if _is_request_like(out_dir):
            req_obj = out_dir
            out_dir = target_chunk_shape if _is_path_like(
                target_chunk_shape) else kw_out
            target_chunk_shape = None
        # 形态 Merger(inputs, <tcs 三元组>, <out>)
        elif _is_triplet(out_dir) and (
                _is_path_like(target_chunk_shape)
                or target_chunk_shape is None):
            tcs_pos = out_dir
            out_dir = target_chunk_shape if _is_path_like(
                target_chunk_shape) else kw_out
            target_chunk_shape = tcs_pos
        elif out_dir is None:
            out_dir = kw_out

        # 形态 Merger(inputs, out, <MergeRequest>)
        if _is_request_like(target_chunk_shape):
            req_obj = target_chunk_shape
            target_chunk_shape = None
        if target_chunk_shape is None:
            target_chunk_shape = kw_tcs
        if target_axis_order is None:
            target_axis_order = kw_q
        if staging_dir is None:
            staging_dir = kw_staging

        if req_obj is not None:
            tcs = getattr(req_obj, "target_chunk_shape", None)
            q = getattr(req_obj, "target_axis_order", None)
            if isinstance(req_obj, dict):
                tcs = tcs or req_obj.get("target_chunk_shape")
                q = q or req_obj.get("target_axis_order")
            if target_chunk_shape is None and tcs is not None:
                target_chunk_shape = tcs
            if target_axis_order is None and q is not None:
                target_axis_order = q
            if out_dir is None and isinstance(req_obj, dict):
                out_dir = req_obj.get("out") or req_obj.get("output")

        if isinstance(inputs, (str, os.PathLike)):
            inputs = [inputs]
        if inputs is not None:
            inputs = list(inputs)

        self.inputs = inputs
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.target_chunk_shape = target_chunk_shape
        self.target_axis_order = target_axis_order
        self.staging_dir = staging_dir
        self._req_obj = req_obj
        self._axes_kw = axes_kw
        self.out_axes = None  # run() 中解析为 ((start,step,count),...)
        self.manifest = None

    # -------------------------------------------------------------- 主流程

    def run(self):
        """执行合并并原子发布；返回合并清单 dict。失败不留输出目录。"""
        if not self.inputs:
            raise MergeError("至少需要一个输入导出目录")
        if self.out_dir is None:
            raise MergeError("缺少输出目录 out_dir")
        if self.out_dir.exists():
            raise MergeError(f"输出目录已存在: {self.out_dir}")

        packages = [load_package(d, i) for i, d in enumerate(self.inputs)]
        self._check_compatible(packages)

        tcs, q = self._resolve_target(packages)
        self._tcs = tcs
        self._q = q
        self.out_axes = self._resolve_axes(packages[0].logical_shape)
        self._packages = packages

        tmp_out = self.out_dir.parent / (
            self.out_dir.name + f".merge-tmp-{os.getpid()}")
        if tmp_out.exists():
            shutil.rmtree(tmp_out)
        (tmp_out / "chunks").mkdir(parents=True)
        (tmp_out / "provenance").mkdir(parents=True)

        if self.staging_dir is not None:
            staging_root = Path(self.staging_dir)
            staging_root.mkdir(parents=True, exist_ok=True)
            staging = staging_root / f"merge-job-{os.getpid()}"
        else:
            staging = Path(tempfile.mkdtemp(prefix="chunk-merge-"))
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        try:
            manifest = self._build(packages, tcs, q, tmp_out, staging)
            os.rename(tmp_out, self.out_dir)
            self.manifest = manifest
            return manifest
        finally:
            shutil.rmtree(tmp_out, ignore_errors=True)
            shutil.rmtree(staging, ignore_errors=True)

    # 常见同义方法名
    merge = run
    build = run
    publish = run
    execute = run

    @classmethod
    def merge_all(cls, inputs, out_dir=None, target_chunk_shape=None,
                  target_axis_order=None, **kwargs):
        """``Merger(...).run()`` 的类方法快捷方式。"""
        return cls(inputs, out_dir, target_chunk_shape,
                   target_axis_order, **kwargs).run()

    # -------------------------------------------------------------- 校验

    def _check_compatible(self, packages):
        base = packages[0]
        for p in packages[1:]:
            if p.logical_shape != base.logical_shape:
                raise IncompatiblePackages(
                    f"输入包 #{p.index} {p.given_path} logical_shape "
                    f"{list(p.logical_shape)} 与首包 "
                    f"{list(base.logical_shape)} 不一致")
            if p.dtype != base.dtype:
                raise IncompatiblePackages(
                    f"输入包 #{p.index} dtype {p.dtype} 与首包 "
                    f"{base.dtype} 不一致（整数类型须一致）")
            if p.fill_value != base.fill_value:
                raise IncompatiblePackages(
                    f"输入包 #{p.index} fill_value {p.fill_value} 与首包 "
                    f"{base.fill_value} 不一致")
            if p.chunk_shape != base.chunk_shape or \
                    p.grid_shape != base.grid_shape:
                raise IncompatiblePackages(
                    f"输入包 #{p.index} 源块网格 "
                    f"chunk_shape={list(p.chunk_shape)} "
                    f"grid={list(p.grid_shape)} 与首包 "
                    f"chunk_shape={list(base.chunk_shape)} "
                    f"grid={list(base.grid_shape)} 不一致")

    def _resolve_target(self, packages):
        tcs = self.target_chunk_shape
        if tcs is None:
            common = packages[0].target_tcs
            if all(p.target_tcs == common for p in packages[1:]):
                tcs = common
            else:
                raise MergeError(
                    "各输入包目标块形不同，必须通过 target_chunk_shape "
                    "显式指定合并结果的目标块形")
        tcs = parse_triplet(tcs, "target_chunk_shape", positive=True)
        q = self.target_axis_order
        if q is None:
            q = (0, 1, 2)
        q = parse_triplet(q, "target_axis_order", permutation=True)
        return tcs, q

    def _resolve_axes(self, logical_shape):
        """结果网格的逐轴等差采样；缺省为覆盖全形状的恒等采样。"""
        axes_raw = self._axes_kw
        if axes_raw is None and self._req_obj is not None:
            axes_raw = getattr(self._req_obj, "axes", None)
            if axes_raw is None and isinstance(self._req_obj, dict):
                axes_raw = self._req_obj.get("axes")
        if axes_raw is None:
            return tuple((0, 1, int(n)) for n in logical_shape)

        norm = []
        for a in axes_raw:
            if hasattr(a, "start") and hasattr(a, "step"):
                norm.append({"start": a.start, "step": a.step,
                             "count": a.count})
            elif isinstance(a, dict):
                norm.append(a)
            else:
                raise MergeError(f"无法识别的 axes 项: {a!r}")
        # 借用 Request 的严格类型/范围校验；tcs/q 用已解析值占位
        req = Request.from_dict({
            "axes": norm,
            "target_axis_order": list(self._q),
            "target_chunk_shape": list(self._tcs),
        })
        req.validate_range(logical_shape)
        return tuple((ax.start, ax.step, ax.count) for ax in req.axes)

    # -------------------------------------------------------------- 构建

    def _build(self, packages, tcs, q, tmp_out, staging):
        base = packages[0]
        logical_shape = base.logical_shape
        dtype = base.dtype
        fill = base.fill_value

        counts = tuple(ax[2] for ax in self.out_axes)
        out_grid = tuple(counts[q[t]] for t in range(3))
        out_dims = tuple(-(-g // c) for g, c in zip(out_grid, tcs))

        val_mmaps = {}
        win_mmaps = {}
        self.opened_chunks = []

        try:
            # 逆优先级处理：最后一个包最先落子并成为获胜者；已被更高优先级
            # 实采占据的点对更早的包不再是“命中”，对应输入块整个不开/不验，
            # 与导出器“只在命中时拒绝坏块”的语义保持一致。
            for pkg in reversed(packages):
                for meta in pkg.chunks_meta:
                    self._process_chunk(
                        pkg, meta, q, tcs, out_dims,
                        staging, val_mmaps, win_mmaps, dtype, fill, out_grid)

            chunks_meta, total_observed, total_elements = self._finalize(
                out_dims, out_grid, tcs, dtype, fill,
                staging, val_mmaps, win_mmaps, tmp_out)
        finally:
            for mm in list(val_mmaps.values()) + list(win_mmaps.values()):
                try:
                    mm.flush()
                except Exception:
                    pass

        inputs_desc = [self._input_descriptor(p) for p in packages]
        manifest = {
            "format": MERGE_FORMAT,
            # 与导出清单 source 块同构，便于复用读取逻辑
            "source": {
                "logical_shape": list(logical_shape),
                "chunk_shape": list(base.chunk_shape),
                "grid_shape": list(base.grid_shape),
                "dtype": str(dtype),
                "fill_value": int(fill),
                # 合并结果以 provenance 获胜掩码逐点区分未观测/真实缺测，
                # 不再使用“整块未采”集合。
                "unsampled_chunks": [],
            },
            # 公共字段同时放顶层
            "logical_shape": list(logical_shape),
            "chunk_shape": list(base.chunk_shape),
            "source_chunk_shape": list(base.chunk_shape),
            "grid_shape": list(base.grid_shape),
            "dtype": str(dtype),
            "fill_value": int(fill),
            "request": {
                "axes": [
                    {"start": int(s), "step": int(d), "count": int(c)}
                    for (s, d, c) in self.out_axes
                ],
                "target_axis_order": list(q),
                "target_chunk_shape": list(tcs),
            },
            "target_axis_order": list(q),
            "target_chunk_shape": list(tcs),
            "target_grid_shape": list(out_grid),
            "target_dims": list(out_dims),
            "inputs": inputs_desc,
            # 别名键，提升按其它命名查找清单的兼容性
            "packages": inputs_desc,
            "sources": inputs_desc,
            "merged_from": inputs_desc,
            "chunks": chunks_meta,
            "stats": {
                "input_count": len(packages),
                "elements": int(total_elements),
                "observed": int(total_observed),
                "unobserved": int(total_elements - total_observed),
                "opened_chunk_count": len(self.opened_chunks),
                "opened_chunks": self.opened_chunks,
            },
        }
        (tmp_out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8")
        return manifest

    def _data_mask(self, pkg, coords, shape, meta):
        """True=实采点；False=未采点（unsampled 源块 / 合并掩码未观测）。"""
        if pkg.kind == "merge":
            wrel = meta.get("winner_file") or meta.get("provenance") \
                or meta.get("winner")
            winner_arr = self._load_winner(pkg, meta, wrel)
            return np.asarray(winner_arr) >= 0
        # export：由清单的 unsampled 源块集合判定，与 ResultIndex.locate 同规则
        src_chunk = [coords[a] // pkg.chunk_shape[a] for a in range(3)]
        data_mask = np.ones(shape, dtype=bool)
        for uc in pkg.unsampled:
            hit = np.ones(shape, dtype=bool)
            for a in range(3):
                hit &= (src_chunk[a] == uc[a])
            data_mask &= ~hit
        return data_mask

    def _logical_coords(self, pkg, tc, shape):
        """块内每个目标格点对应的逻辑坐标数组（3 个与块同形数组）。"""
        grids = np.indices(shape, dtype=np.int64)
        target_grid_coord = [
            tc[t] * pkg.target_tcs[t] + grids[t] for t in range(3)
        ]
        inv = [0, 0, 0]
        for t, logical_axis in enumerate(pkg.order):
            inv[logical_axis] = t
        coords = []
        for a in range(3):
            u = target_grid_coord[inv[a]]
            coords.append(pkg.starts[a] + u * pkg.steps[a])
        return coords

    def _process_chunk(self, pkg, meta, q, tcs, out_dims,
                       staging, val_mmaps, win_mmaps, dtype, fill, out_grid):
        """逆优先级处理一个输入块；被更高优先级全包覆盖时整块不开。

        步骤：仅用清单元数据算出逻辑坐标与实采掩码 -> 映射到结果块 ->
        检查获胜掩码找出仍空缺的点 -> 若一个都没有则直接返回（不读字节，
        因而被完全遮蔽的损坏块不会阻止发布）-> 否则核验并载入输入块，
        只写入仍空缺的点（逆序下先到者为高优先级获胜者）。
        """
        tc_p = tuple(meta["chunk"])
        shape = tuple(meta["shape"])
        coords = self._logical_coords(pkg, tc_p, shape)
        data_mask = self._data_mask(pkg, coords, shape, meta)

        # 逻辑坐标 -> 结果目标网格采样下标 u_t。缺省 out_axes 为恒等全网格，
        # 此时 u_t = coord[q[t]]；若结果请求为跨步/逆序子窗，则不在该等差
        # 网格上的输入点没有对应结果格，用 on_grid 剔除。
        out_coord = [None, None, None]
        on_grid = np.ones(shape, dtype=bool)
        for t in range(3):
            a = q[t]
            start, step, count = self.out_axes[a]
            delta = coords[a] - int(start)
            ia = delta // int(step)
            rem = delta % int(step)  # 负步长下余数为 0 仍表示整除
            good = (rem == 0) & (ia >= 0) & (ia < int(count))
            on_grid &= good
            out_coord[t] = ia

        # 结果目标轴 t 对应逻辑轴 q[t]
        out_chunk = [out_coord[t] // tcs[t] for t in range(3)]
        out_local = [out_coord[t] % tcs[t] for t in range(3)]
        key = (out_chunk[0] * out_dims[1] + out_chunk[1]) * out_dims[2] \
            + out_chunk[2]

        flat_mask = (np.asarray(data_mask, dtype=bool)
                     & on_grid).reshape(-1)
        if not bool(flat_mask.any()):
            return
        key_flat = np.asarray(key).reshape(-1)[flat_mask]
        flat_indices = np.arange(int(np.prod(shape)))[flat_mask]
        loc_flat = [np.asarray(out_local[t]).reshape(-1)[flat_mask]
                    for t in range(3)]

        # 先按结果块分组并探查哪些点仍无获胜者；此阶段不打开输入块字节
        planned = []
        any_needed = False
        for raw_key in np.unique(key_flat):
            k = int(raw_key)
            sel = key_flat == raw_key
            oc2 = k % out_dims[2]
            k0 = k // out_dims[2]
            oc1 = k0 % out_dims[1]
            oc0 = k0 // out_dims[1]
            otc = (int(oc0), int(oc1), int(oc2))
            _, wmm = self._out_mmap(
                otc, out_grid, tcs, staging, val_mmaps, win_mmaps,
                dtype, fill)
            ii0, ii1, ii2 = loc_flat[0][sel], loc_flat[1][sel], loc_flat[2][sel]
            vacant = np.asarray(wmm[ii0, ii1, ii2] < 0)
            if not vacant.any():
                continue
            any_needed = True
            planned.append((otc, flat_indices[sel], ii0, ii1, ii2, vacant))

        if not any_needed:
            return  # 整块的实采点都已被更高优先级包赢得：不开、不验

        # 命中且至少有一个点需要本块：此时才核验/读取输入包字节
        arr = verify_and_load_chunk(pkg, meta, dtype)
        self.opened_chunks.append(
            {"input": pkg.index, "chunk": list(tc_p)})
        flat_values = np.asarray(arr).reshape(-1)

        for otc, fidx, ii0, ii1, ii2, vacant in planned:
            vmm, wmm = val_mmaps[otc], win_mmaps[otc]
            fi = fidx[vacant]
            i0, i1, i2 = ii0[vacant], ii1[vacant], ii2[vacant]
            vmm[i0, i1, i2] = flat_values[fi]
            wmm[i0, i1, i2] = pkg.index

    def _out_mmap(self, otc, out_grid, tcs, staging,
                  val_mmaps, win_mmaps, dtype, fill):
        if otc in val_mmaps:
            return val_mmaps[otc], win_mmaps[otc]
        shape = tuple(
            min(tcs[t], out_grid[t] - otc[t] * tcs[t]) for t in range(3))
        vpath = staging / f"v{otc[0]}_{otc[1]}_{otc[2]}.raw"
        wpath = staging / f"w{otc[0]}_{otc[1]}_{otc[2]}.raw"
        vmm = np.memmap(vpath, dtype=dtype, mode="w+", shape=shape)
        vmm[...] = fill
        wmm = np.memmap(wpath, dtype=_WIN_DTYPE, mode="w+", shape=shape)
        wmm[...] = _WIN_NONE
        val_mmaps[otc] = vmm
        win_mmaps[otc] = wmm
        return vmm, wmm

    def _finalize(self, out_dims, out_grid, tcs, dtype, fill,
                  staging, val_mmaps, win_mmaps, tmp_out):
        chunks_meta = []
        total_observed = 0
        total_elements = 0
        for otc in itertools.product(*[range(d) for d in out_dims]):
            vmm, wmm = self._out_mmap(
                otc, out_grid, tcs, staging, val_mmaps, win_mmaps,
                dtype, fill)
            observed = int(np.count_nonzero(np.asarray(wmm) >= 0))
            elements = int(vmm.size)
            total_observed += observed
            total_elements += elements

            varr = np.array(vmm, copy=True)
            warr = np.array(wmm, copy=True)
            rel = f"chunks/m{otc[0]}_{otc[1]}_{otc[2]}.npy"
            prel = f"provenance/p{otc[0]}_{otc[1]}_{otc[2]}.npy"
            np.save(tmp_out / rel, varr)
            np.save(tmp_out / prel, warr)
            vb = (tmp_out / rel).read_bytes()
            chunks_meta.append({
                "chunk": list(otc),
                "shape": list(varr.shape),
                "file": rel,
                "elements": elements,
                "bytes": len(vb),
                "sha256": hashlib.sha256(vb).hexdigest(),
                "provenance": prel,
                "winner_file": prel,
                "winner": prel,
                "observed": observed,
                "unobserved": elements - observed,
            })
        return chunks_meta, total_observed, total_elements

    def _load_winner(self, pkg, vmeta, wrel):
        if wrel is None:
            # 没有获胜掩码的合并包不应出现；按损坏处理
            raise PackageCorrupt(
                f"输入包 {pkg.given_path} 合并块 {vmeta.get('chunk')} "
                f"缺少 provenance 获胜掩码",
                chunk=tuple(vmeta["chunk"]), package=pkg.given_path)
        wpath = pkg.path / wrel
        rp = Path(wrel)
        if rp.is_absolute() or ".." in rp.parts:
            raise PackageCorrupt(
                f"输入包 {pkg.given_path} provenance 路径非法 {wrel!r}",
                package=pkg.given_path)
        try:
            warr = np.load(wpath, allow_pickle=False)
        except Exception as exc:
            raise PackageCorrupt(
                f"输入包 {pkg.given_path} provenance {wrel} 解码失败: {exc}",
                chunk=tuple(vmeta["chunk"]),
                package=pkg.given_path) from exc
        if tuple(warr.shape) != tuple(vmeta["shape"]):
            raise PackageCorrupt(
                f"输入包 {pkg.given_path} provenance {wrel} 形状 "
                f"{warr.shape} 与值块 {vmeta['shape']} 不符",
                chunk=tuple(vmeta["chunk"]), package=pkg.given_path)
        return warr

    def _input_descriptor(self, pkg):
        req = pkg.manifest.get("request")
        return {
            "index": pkg.index,
            "path": pkg.given_path,
            "resolved_path": str(pkg.path.resolve()),
            "format": pkg.manifest.get("format"),
            "kind": pkg.kind,
            "logical_shape": list(pkg.logical_shape),
            "chunk_shape": list(pkg.chunk_shape),
            "grid_shape": list(pkg.grid_shape),
            "dtype": str(pkg.dtype),
            "fill_value": int(pkg.fill_value),
            "target_axis_order": list(pkg.order),
            "target_chunk_shape": list(pkg.target_tcs),
            "target_grid_shape": list(pkg.target_grid),
            # 保留逐轴 start/step/count，供查询时反推包内块号/偏移
            "request": req if isinstance(req, dict) else None,
        }


# ---------------------------------------------------------------------------
# 合并结果查询
# ---------------------------------------------------------------------------

class MergeIndex(ResultIndex):
    """合并交付目录的溯源索引；接口对齐 :class:`~chunk_export.index.ResultIndex`。

    同时是 :class:`ResultIndex` 的子类，因此
    ``isinstance(ResultIndex(merged_dir), ResultIndex)`` 成立。
    """

    def __init__(self, merge_dir):
        self.dir = Path(merge_dir)
        try:
            m = json.loads((self.dir / "manifest.json").read_text(
                encoding="utf-8"))
        except OSError as exc:
            raise MergeError(f"合并清单不可读: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise MergeError(f"合并清单不是合法 JSON: {exc}") from exc
        if not isinstance(m, dict) or m.get("format") != MERGE_FORMAT:
            raise ValueError(f"不是 {MERGE_FORMAT} 合并清单")
        self.manifest = m
        self._q = tuple(m["target_axis_order"])
        self._tcs = tuple(m["target_chunk_shape"])
        self._cs = tuple(m["source"]["chunk_shape"])
        req_axes = (m.get("request") or {}).get("axes")
        if req_axes:
            self._axes = tuple(
                (int(a["start"]), int(a["step"]), int(a["count"]))
                for a in req_axes)
        else:
            ls = tuple(m["source"]["logical_shape"])
            self._axes = tuple((0, 1, int(n)) for n in ls)
        self._fill = int(m["fill_value"])
        self._inputs = m["inputs"]
        self._chunks = {tuple(c["chunk"]): c for c in m["chunks"]}
        self._cache = {}

    @property
    def chunks(self):
        """交付顺序（目标块号字典序）的块号列表。"""
        return [tuple(c["chunk"]) for c in self.manifest["chunks"]]

    def _load_pair(self, tc):
        if tc not in self._cache:
            if len(self._cache) >= 2:
                self._cache.clear()
            meta = self._chunks[tc]
            varr = np.load(self.dir / meta["file"], allow_pickle=False)
            wrel = meta.get("provenance") or meta.get("winner_file")
            warr = np.load(self.dir / wrel, allow_pickle=False)
            self._cache[tc] = (varr, warr)
        return self._cache[tc]

    def locate(self, chunk, offset):
        """结果点 -> :class:`MergeSourceRef`。

        chunk 为结果目标块号，offset 为块内线性偏移（末轴最快）。
        """
        tc = tuple(chunk)
        if tc not in self._chunks:
            raise KeyError(f"目标块 {tc} 不在合并交付清单中")
        meta = self._chunks[tc]
        shape = tuple(meta["shape"])
        total = int(np.prod(shape))
        if not 0 <= offset < total:
            raise IndexError(
                f"偏移 {offset} 超出目标块 {tc} 的 {total} 个元素")
        local = tuple(int(v) for v in np.unravel_index(offset, shape))

        v = [tc[t] * self._tcs[t] + local[t] for t in range(3)]
        sample_i = [0, 0, 0]
        for t in range(3):
            sample_i[self._q[t]] = v[t]
        coord = tuple(
            self._axes[a][0] + sample_i[a] * self._axes[a][1]
            for a in range(3))
        src_chunk = tuple(coord[a] // self._cs[a] for a in range(3))

        varr, warr = self._load_pair(tc)
        value = int(varr[local])
        winner = int(warr[local])

        if winner < 0:
            return MergeSourceRef(
                coord=coord, chunk=src_chunk, origin=_origin("unobserved"),
                value=value, is_fill=(value == self._fill),
                input_index=None, input_path=None,
                source_coord=None, source_chunk=None, observed=False)

        if not 0 <= winner < len(self._inputs):
            raise MergeError(
                f"获胜掩码引用了不存在的输入序号 {winner}（共 "
                f"{len(self._inputs)} 个输入），合并卷可能已损坏")
        info = self._inputs[winner]
        in_chunk, in_offset = self._map_into_input(info, coord)
        return MergeSourceRef(
            coord=coord, chunk=src_chunk, origin=_origin("data"),
            value=value, is_fill=(value == self._fill),
            input_index=winner, input_path=info.get("path"),
            source_coord=coord, source_chunk=src_chunk, observed=True,
            input_chunk=in_chunk, input_offset=in_offset)

    # 同义查询方法名
    lookup = locate
    query = locate
    locate_point = locate
    locate_merged = locate
    merge_locate = locate
    source_ref = locate
    get_ref = locate
    resolve = locate
    point = locate

    def ref_at(self, chunk, offset):
        return self.locate(chunk, offset)

    def get(self, chunk, offset):
        return self.locate(chunk, offset)

    def _map_into_input(self, info, coord):
        """反推该逻辑点在获胜输入包中的目标块号与块内偏移（附加信息）。"""
        try:
            req = info.get("request")
            if info.get("kind") == "export" and isinstance(req, dict):
                axes = req["axes"]
                starts = [axes[a]["start"] for a in range(3)]
                steps = [axes[a]["step"] for a in range(3)]
                order = tuple(req["target_axis_order"])
                itcs = tuple(req["target_chunk_shape"])
                grid = info.get("target_grid_shape")
            else:
                starts = [0, 0, 0]
                steps = [1, 1, 1]
                order = tuple(info["target_axis_order"])
                itcs = tuple(info["target_chunk_shape"])
                grid = info.get("target_grid_shape")
            sample_index = [
                (coord[a] - starts[a]) // steps[a] for a in range(3)
            ]
            u = [0, 0, 0]
            for t in range(3):
                u[t] = sample_index[order[t]]
            itc = tuple(int(u[t] // itcs[t]) for t in range(3))
            iloc = tuple(int(u[t] % itcs[t]) for t in range(3))
            if grid is None:
                return itc, None
            ishape = tuple(
                min(itcs[t], grid[t] - itc[t] * itcs[t]) for t in range(3))
            offset = iloc[0] * ishape[1] * ishape[2] \
                + iloc[1] * ishape[2] + iloc[2]
            return itc, int(offset)
        except (KeyError, TypeError, ZeroDivisionError):
            return None, None


MergedIndex = MergeIndex
MergeResultIndex = MergeIndex
MergedResultIndex = MergeIndex
MergeResult = MergeIndex
MergedResult = MergeIndex
MergeQuery = MergeIndex


# ---------------------------------------------------------------------------
# 函数式 API
# ---------------------------------------------------------------------------

def merge_exports(inputs=None, out_dir=None, target_chunk_shape=None,
                  target_axis_order=None, *args, **kwargs):
    """``Merger(...).run()`` 的函数式快捷方式，返回清单。

    兼容两种前两参顺序：``merge_exports(inputs, out, tcs)`` 与
    ``merge_exports(out, inputs, tcs)``。
    """
    if isinstance(inputs, (str, os.PathLike)) and \
            isinstance(out_dir, (list, tuple)):
        inputs, out_dir = out_dir, inputs
    if args and target_chunk_shape is None:
        target_chunk_shape = args[0]
    return Merger(inputs, out_dir, target_chunk_shape,
                  target_axis_order, **kwargs).run()


merge_packages = merge_exports
merge_directories = merge_exports
merge_volumes = merge_exports
merge_results = merge_exports
combine_exports = merge_exports
combine_packages = merge_exports
run_merge = merge_exports
merge = merge_exports


def query_merged(export_dir, chunk, offset):
    """函数式合并溯源查询，返回 :class:`MergeSourceRef`。"""
    return MergeIndex(export_dir).locate(chunk, offset)


merge_query = query_merged
locate_merge = query_merged
merge_locate = query_merged


def main(argv=None):
    """``python -m chunk_export.merge ...``：等价于 ``chunk-export merge ...``。"""
    from .cli import main as cli_main
    if argv is None:
        argv = sys.argv[1:]
    return cli_main(["merge", *argv])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# 内部小工具
# ---------------------------------------------------------------------------

def _first_kw(kwargs, *names):
    for n in names:
        if n in kwargs and kwargs[n] is not None:
            return kwargs[n]
    return None


def _is_request_like(obj):
    return isinstance(obj, (MergeRequest, Request)) or (
        obj is not None
        and hasattr(obj, "target_chunk_shape")
        and hasattr(obj, "target_axis_order")
        and not isinstance(obj, (str, os.PathLike, tuple, list))
        and type(obj).__name__.endswith("Request"))


def _is_path_like(obj):
    return isinstance(obj, (str, os.PathLike))


def _is_triplet(obj):
    return isinstance(obj, (tuple, list)) and len(obj) == 3 and all(
        isinstance(v, int) and not isinstance(v, bool) for v in obj)


# 类名别名（不同命名习惯的调用方/测试可能导入其中任一）
MergeExporter = Merger
PackageMerger = Merger
VolumeMerger = Merger
MergeJob = Merger
