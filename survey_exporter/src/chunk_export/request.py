"""请求模型：逐轴起点/非零有符号步长/非负点数 + 目标轴序与目标块形。

坐标严格按等差生成：coord(i) = start + i * step，i ∈ [0, count)。
任一轴点数为 0 时交付空结果，但类型与步长仍然校验。
"""
from __future__ import annotations

import operator
from dataclasses import dataclass

from .errors import RequestInvalid


def _as_int(value, what):
    if isinstance(value, bool):
        raise RequestInvalid(f"{what} 必须是整数，得到布尔值 {value!r}")
    try:
        return operator.index(value)
    except TypeError:
        raise RequestInvalid(f"{what} 必须是整数，得到 {value!r}") from None


@dataclass(frozen=True)
class AxisRequest:
    start: int
    step: int   # 非零有符号
    count: int  # 非负


@dataclass(frozen=True)
class Request:
    axes: tuple              # 3 个 AxisRequest，逻辑轴序
    target_axis_order: tuple  # 目标轴 t 对应逻辑轴 target_axis_order[t]
    target_chunk_shape: tuple

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict):
            raise RequestInvalid("请求必须是对象")

        axes_raw = raw.get("axes")
        if not isinstance(axes_raw, (list, tuple)) or len(axes_raw) != 3:
            raise RequestInvalid("axes 必须是三元列表")
        axes = []
        for i, a in enumerate(axes_raw):
            if not isinstance(a, dict):
                raise RequestInvalid(f"axes[{i}] 必须是对象")
            start = _as_int(a.get("start"), f"axes[{i}].start")
            step = _as_int(a.get("step"), f"axes[{i}].step")
            count = _as_int(a.get("count"), f"axes[{i}].count")
            if step == 0:
                raise RequestInvalid(f"axes[{i}].step 不允许为 0")
            if count < 0:
                raise RequestInvalid(f"axes[{i}].count 不允许为负")
            axes.append(AxisRequest(start, step, count))

        order_raw = raw.get("target_axis_order")
        if not isinstance(order_raw, (list, tuple)) or len(order_raw) != 3:
            raise RequestInvalid("target_axis_order 必须是三元列表")
        order = tuple(_as_int(v, "target_axis_order 项") for v in order_raw)
        if sorted(order) != [0, 1, 2]:
            raise RequestInvalid(f"target_axis_order 不是轴排列: {order}")

        tcs_raw = raw.get("target_chunk_shape")
        if not isinstance(tcs_raw, (list, tuple)) or len(tcs_raw) != 3:
            raise RequestInvalid("target_chunk_shape 必须是三元列表")
        tcs = tuple(_as_int(v, "target_chunk_shape 项") for v in tcs_raw)
        if any(v <= 0 for v in tcs):
            raise RequestInvalid(f"target_chunk_shape 必须为正整数: {tcs}")

        return cls(axes=tuple(axes), target_axis_order=order, target_chunk_shape=tcs)

    def validate_range(self, logical_shape):
        """逐轴校验等差坐标范围；点数为 0 的轴无坐标可验，直接跳过。"""
        for i, (ax, length) in enumerate(zip(self.axes, logical_shape)):
            if ax.count == 0:
                continue
            first = ax.start
            last = ax.start + (ax.count - 1) * ax.step
            lo, hi = (first, last) if first <= last else (last, first)
            if lo < 0 or hi >= length:
                raise RequestInvalid(
                    f"axes[{i}] 坐标越界: [{lo}, {hi}] 不在 [0, {length}) 内")


def _triplet(value, what, *, positive=False, permutation=False):
    if isinstance(value, str):
        text = value.strip()
        parsed = None
        if text.startswith("["):
            import json
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = None
        if parsed is None:
            for sep in (",", "x", "X", ";"):
                if sep in text:
                    parts = [p.strip() for p in text.split(sep) if p.strip()]
                    if len(parts) == 3:
                        parsed = parts
                    break
        value = parsed
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise RequestInvalid(f"{what} 必须是三元整数列表: {value!r}")
    out = []
    for v in value:
        if isinstance(v, bool):
            raise RequestInvalid(f"{what} 不允许布尔值: {v!r}")
        try:
            iv = operator.index(v)
        except TypeError:
            try:
                iv = int(str(v).strip())
            except (TypeError, ValueError):
                raise RequestInvalid(f"{what} 必须是整数，得到 {v!r}") from None
        out.append(iv)
    t = tuple(out)
    if positive and any(v <= 0 for v in t):
        raise RequestInvalid(f"{what} 必须为正整数: {t}")
    if permutation and sorted(t) != [0, 1, 2]:
        raise RequestInvalid(f"{what} 不是轴排列: {t}")
    return t


@dataclass(frozen=True)
class MergeRequest:
    """合并交付参数：结果目标块形必填，目标轴序缺省为逻辑轴序 [0,1,2]。"""

    target_chunk_shape: tuple
    target_axis_order: tuple = (0, 1, 2)

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict):
            raise RequestInvalid("合并请求必须是对象")
        tcs = raw.get("target_chunk_shape",
                      raw.get("target_chunks", raw.get("chunk_shape")))
        if tcs is None:
            raise RequestInvalid("合并请求缺少 target_chunk_shape")
        q = raw.get("target_axis_order",
                    raw.get("axis_order", raw.get("order", (0, 1, 2))))
        return cls(
            target_chunk_shape=_triplet(
                tcs, "target_chunk_shape", positive=True),
            target_axis_order=_triplet(
                q, "target_axis_order", permutation=True),
        )


# 常见别名
MergeConfig = MergeRequest
