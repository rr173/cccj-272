"""错误类型：目录错误、请求错误与交付拒绝。"""
from __future__ import annotations


class ExportError(Exception):
    """导出流程的基类错误。"""


class CatalogInvalid(ExportError):
    """目录自身不一致或缺字段。"""


class RequestInvalid(ExportError):
    """请求参数非法（类型、步长、点数、轴序、范围等）。"""


class DeliveryRefused(ExportError):
    """命中坏块 / 未声明块 / 超解码上限块，拒绝交付。

    属性:
        chunk:  逻辑源块号（三元组），与请求坐标无关；非块类错误为 None。
        reason: 'corrupt' | 'undeclared' | 'oversize'
    """

    def __init__(self, message, *, chunk=None, reason=None):
        super().__init__(message)
        self.chunk = tuple(chunk) if chunk is not None else None
        self.reason = reason
