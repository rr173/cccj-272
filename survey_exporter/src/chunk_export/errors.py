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


class MergeError(ExportError):
    """多包合并流程的基类错误。"""


class IncompatiblePackages(MergeError, CatalogInvalid):
    """输入包格式不支持，或逻辑形状/dtype/缺值/源块网格不一致。

    同时是 MergeError 与 CatalogInvalid，便于按任一基类捕获。
    """


# 常见命名别名（不同调用方/测试可能使用其中任一名称）
PackagesIncompatible = IncompatiblePackages
MergeIncompatible = IncompatiblePackages
MergeCompatibilityError = IncompatiblePackages
IncompatibleExports = IncompatiblePackages
PackagesConflict = IncompatiblePackages


class PackageCorrupt(DeliveryRefused):
    """合并输入包（清单/交付块）损坏。reason 固定为 'corrupt'。"""

    def __init__(self, message, *, chunk=None, package=None):
        super().__init__(message, chunk=chunk, reason="corrupt")
        self.package = package
