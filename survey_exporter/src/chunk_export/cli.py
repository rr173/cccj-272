"""命令行：export 导出子窗；query 查询交付点的源坐标/源块。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .catalog import load_catalog
from .core import Exporter
from .errors import CatalogInvalid, DeliveryRefused, ExportError, RequestInvalid
from .index import ResultIndex
from .request import Request


def _emit(obj, *, err=False):
    print(json.dumps(obj, ensure_ascii=False),
          file=sys.stderr if err else sys.stdout)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="chunk-export", description="三轴分块整数数据的离线子窗导出")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("export", help="按请求导出子窗")
    p_exp.add_argument("--catalog", required=True, help="目录 JSON 路径")
    p_exp.add_argument("--request", required=True, help="请求 JSON 路径")
    p_exp.add_argument("--out", required=True, help="输出目录（须不存在）")
    p_exp.add_argument("--staging", default=None, help="外部暂存目录（默认系统临时目录）")

    p_q = sub.add_parser("query", help="查询交付点的源坐标/源块")
    p_q.add_argument("--export", required=True, help="导出目录")
    p_q.add_argument("--chunk", required=True, help="目标块号，如 0,1,2")
    p_q.add_argument("--offset", required=True, type=int, help="块内线性偏移（末轴最快）")

    args = parser.parse_args(argv)

    if args.cmd == "export":
        try:
            catalog = load_catalog(args.catalog)
            try:
                raw_req = json.loads(Path(args.request).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RequestInvalid(f"请求文件不可读: {exc}") from exc
            request = Request.from_dict(raw_req)
            manifest = Exporter(catalog, request, args.out,
                                staging_dir=args.staging).run()
        except RequestInvalid as exc:
            _emit({"error": "invalid_request", "detail": str(exc)}, err=True)
            return 2
        except CatalogInvalid as exc:
            _emit({"error": "invalid_catalog", "detail": str(exc)}, err=True)
            return 2
        except DeliveryRefused as exc:
            _emit({"error": "delivery_refused", "reason": exc.reason,
                   "chunk": list(exc.chunk) if exc.chunk is not None else None,
                   "detail": str(exc)}, err=True)
            return 3
        except ExportError as exc:
            _emit({"error": "export_failed", "detail": str(exc)}, err=True)
            return 1
        _emit({"out": args.out,
               "chunks": len(manifest["chunks"]),
               "source_chunks_opened": manifest["accounting"]["open_count"],
               "bytes_read": manifest["accounting"]["bytes_read"]})
        return 0

    # query
    idx = ResultIndex(args.export)
    tc = tuple(int(x) for x in args.chunk.split(","))
    ref = idx.locate(tc, args.offset)
    _emit({"coord": list(ref.coord), "chunk": list(ref.chunk),
           "origin": ref.origin, "value": ref.value, "is_fill": ref.is_fill})
    return 0


if __name__ == "__main__":
    sys.exit(main())
