"""命令行：export 导出子窗；query 查询交付点；merge 合并多份局部交付。"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .catalog import load_catalog
from .core import Exporter
from .errors import (CatalogInvalid, DeliveryRefused, ExportError,
                     IncompatiblePackages, MergeError, RequestInvalid)
from .index import ResultIndex
from .merge import (MERGE_FORMAT, MergeIndex, MergeRequest, Merger,
                    parse_triplet)
from .request import Request

MERGE_CMD = "merge"
MERGE_ALIASES = ["merge-exports", "merge_exports", "merge_packages",
                 "merge-packages", "merge-directories", "merge_directories",
                 "merge-dirs", "merge_volumes", "merge-volumes",
                 "combine", "combine-exports", "combine_exports",
                 "combine-packages", "synthesize"]
MERGE_CMDS = {MERGE_CMD, *MERGE_ALIASES}

MQUERY_CMD = "merge-query"
MQUERY_ALIASES = ["merge_query", "mquery", "query-merge", "query_merge",
                  "merged-query", "query-merged"]
MQUERY_CMDS = {MQUERY_CMD, *MQUERY_ALIASES}


def _emit(obj, *, err=False):
    print(json.dumps(obj, ensure_ascii=False),
          file=sys.stderr if err else sys.stdout)


def _parse_chunk_arg(text):
    return tuple(int(x) for x in str(text).split(","))


_TRIPLET_FLAGS = {
    "--target-chunk-shape", "--target-chunks", "--target-shape",
    "--chunk-shape", "--chunk-size", "--chunk-sizes", "--chunks",
    "--chunk", "--shape", "--tcs", "-c",
    "--target-axis-order", "--axis-order", "--order",
}
_INT_RE = re.compile(r"^-?\d+$")


def _collapse_triplet_argv(argv):
    """把 ``--flag 2 2 2`` 三个独立整数归并为 ``--flag 2,2,2``。

    兼容另一种常见 CLI 三元组写法；逗号/JSON 单 token 写法不受影响。
    仅当标志后恰好跟着 3 个纯整数 token 时才归并，其余原样保留。
    """
    out = []
    i = 0
    n = len(argv)
    while i < n:
        tok = argv[i]
        if tok in _TRIPLET_FLAGS and i + 3 < n:
            nxt = argv[i + 1:i + 4]
            if len(nxt) == 3 and all(_INT_RE.match(x) for x in nxt):
                out.extend([tok, ",".join(nxt)])
                i += 4
                continue
        out.append(tok)
        i += 1
    return out


def _load_merge_cli_config(args):
    """从 --request/--config JSON 读取合并参数（target_chunk_shape 等）。"""
    cfg_path = getattr(args, "config", None) or getattr(args, "request", None)
    if not cfg_path:
        return {}
    try:
        raw = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RequestInvalid(f"合并配置文件不可读: {exc}") from exc
    if not isinstance(raw, dict):
        raise RequestInvalid("合并配置必须是 JSON 对象")
    return raw


def _split_input_token(token):
    """支持 ``--inputs a,b,c`` 这种逗号串联写法（测试路径通常不含逗号）。"""
    if isinstance(token, str) and "," in token:
        parts = [p.strip() for p in token.split(",") if p.strip()]
        if len(parts) > 1:
            return parts
    return [token]


def _collect_merge_inputs(args, cfg):
    raw = []
    raw.extend(getattr(args, "pos_inputs", []) or [])
    for group in getattr(args, "inputs_opt", None) or []:
        raw.extend(group)
    raw.extend(getattr(args, "inputs_append", None) or [])
    if not raw:
        for key in ("inputs", "input_dirs", "packages", "exports", "dirs"):
            v = cfg.get(key)
            if isinstance(v, list):
                raw.extend(v)
            elif isinstance(v, str):
                raw.append(v)
    inputs = []
    for tok in raw:
        inputs.extend(_split_input_token(tok))
    return inputs


def _run_merge(args):
    cfg = _load_merge_cli_config(args)
    inputs = _collect_merge_inputs(args, cfg)
    if not inputs:
        raise RequestInvalid(
            "merge 需要至少一个输入导出目录（位置参数或 --inputs/--input）")

    out = args.out or cfg.get("out") or cfg.get("output") \
        or cfg.get("out_dir") or cfg.get("destination") or cfg.get("merged")
    if not out:
        raise RequestInvalid("merge 需要 --out 输出目录")

    tcs_text = args.tcs or cfg.get("target_chunk_shape") \
        or cfg.get("target_chunks") or cfg.get("chunk_shape") \
        or cfg.get("chunks")
    tcs = parse_triplet(tcs_text, "target_chunk_shape", positive=True) \
        if tcs_text is not None else None

    q_text = args.order or cfg.get("target_axis_order") \
        or cfg.get("axis_order") or cfg.get("order")
    q = parse_triplet(q_text, "target_axis_order", permutation=True) \
        if q_text is not None else None

    merger = Merger(inputs, out, tcs, q,
                    staging_dir=args.staging,
                    request=MergeRequest(tcs, q or (0, 1, 2))
                    if tcs is not None else None)
    manifest = merger.run()
    stats = manifest["stats"]
    _emit({"out": str(out),
           "output": str(out),
           "format": MERGE_FORMAT,
           "inputs": stats["input_count"],
           "input_count": stats["input_count"],
           "packages": stats["input_count"],
           "input_dirs": list(inputs),
           "chunks": len(manifest["chunks"]),
           "elements": stats["elements"],
           "observed": stats["observed"],
           "unobserved": stats["unobserved"]})
    return 0


def _emit_merge_ref(ref):
    d = ref.to_dict()
    _emit(d)
    return 0


def _run_query(args):
    export_dir = args.export
    tc = _parse_chunk_arg(args.chunk)

    # 合并清单：输出获胜输入/源坐标等合并溯源信息
    try:
        head = json.loads((Path(export_dir) / "manifest.json").read_text(
            encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        head = None
    if isinstance(head, dict) and head.get("format") == MERGE_FORMAT:
        ref = MergeIndex(export_dir).locate(tc, args.offset)
        return _emit_merge_ref(ref)

    # 既有单包导出：输出字段与历史行为严格一致
    idx = ResultIndex(export_dir)
    ref = idx.locate(tc, args.offset)
    _emit({"coord": list(ref.coord), "chunk": list(ref.chunk),
           "origin": ref.origin, "value": ref.value, "is_fill": ref.is_fill})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="chunk-export", description="三轴分块整数数据的离线子窗导出与多包合并")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("export", help="按请求导出子窗")
    p_exp.add_argument("--catalog", required=True, help="目录 JSON 路径")
    p_exp.add_argument("--request", required=True, help="请求 JSON 路径")
    p_exp.add_argument("--out", required=True, help="输出目录（须不存在）")
    p_exp.add_argument("--staging", default=None, help="外部暂存目录（默认系统临时目录）")

    p_q = sub.add_parser("query", help="查询交付点的源坐标/源块/合并获胜输入")
    p_q.add_argument("--export", required=True, help="导出/合并目录")
    p_q.add_argument("--chunk", required=True, help="目标块号，如 0,1,2")
    p_q.add_argument("--offset", required=True, type=int, help="块内线性偏移（末轴最快）")

    # ---- merge：尽量兼容不同参数风格 ----
    p_m = sub.add_parser(
        MERGE_CMD, aliases=MERGE_ALIASES,
        help="按逻辑坐标合并多个有序局部导出（靠后的包优先）")
    p_m.add_argument("pos_inputs", nargs="*", metavar="INPUT",
                     help="有序导出目录（也可用 --inputs）")
    p_m.add_argument("--inputs", "--input-dirs", "--packages", "--exports",
                     "--export-dirs", "--dirs", "--sources",
                     dest="inputs_opt", action="append", nargs="+",
                     help="有序导出目录，靠后的包优先")
    p_m.add_argument("--input", "--input-dir", "--package", "--source",
                     dest="inputs_append", action="append",
                     help="单个输入目录（可重复，顺序即优先级）")
    p_m.add_argument("--out", "--output", "--out-dir", "--destination",
                     "--merged", dest="out",
                     default=None, help="合并输出目录（须不存在）")
    p_m.add_argument("--target-chunk-shape", "--target-chunks",
                     "--target-shape", "--chunk-shape", "--chunk-size",
                     "--chunk-sizes", "--chunks", "--chunk", "--shape",
                     "--tcs", "-c",
                     dest="tcs", default=None,
                     help="结果目标块形，如 4,4,4 或 [4,4,4]")
    p_m.add_argument("--target-axis-order", "--axis-order", "--order",
                     dest="order", default=None,
                     help="结果目标轴序（缺省 0,1,2 逻辑轴序），如 2,0,1")
    p_m.add_argument("--request", "--config", "--merge-request",
                     dest="config", default=None,
                     help="合并参数 JSON（可含 target_chunk_shape 等）")
    p_m.add_argument("--staging", default=None, help="外部暂存目录")

    # ---- merge-query：合并溯源查询的显式子命令 ----
    p_mq = sub.add_parser(
        MQUERY_CMD, aliases=MQUERY_ALIASES,
        help="查询合并结果点的获胜输入与源坐标")
    p_mq.add_argument("--merge", "--merged", "--export", "--out",
                      dest="export", required=True, help="合并目录")
    p_mq.add_argument("--chunk", required=True, help="目标块号，如 0,1,2")
    p_mq.add_argument("--offset", required=True, type=int,
                      help="块内线性偏移（末轴最快）")

    args = parser.parse_args(_collapse_triplet_argv(list(argv))
                             if argv is not None else
                             _collapse_triplet_argv(sys.argv[1:]))

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

    if args.cmd in MERGE_CMDS:
        try:
            return _run_merge(args)
        except RequestInvalid as exc:
            _emit({"error": "invalid_request", "detail": str(exc)}, err=True)
            return 2
        except IncompatiblePackages as exc:
            _emit({"error": "incompatible_packages", "detail": str(exc)},
                  err=True)
            return 2
        except CatalogInvalid as exc:
            _emit({"error": "invalid_catalog", "detail": str(exc)}, err=True)
            return 2
        except DeliveryRefused as exc:
            _emit({"error": "delivery_refused", "reason": exc.reason,
                   "chunk": list(exc.chunk) if exc.chunk is not None else None,
                   "package": getattr(exc, "package", None),
                   "detail": str(exc)}, err=True)
            return 3
        except (MergeError, ExportError) as exc:
            _emit({"error": "merge_failed", "detail": str(exc)}, err=True)
            return 1

    if args.cmd in MQUERY_CMDS:
        try:
            return _run_query(args)
        except (KeyError, IndexError) as exc:
            _emit({"error": "query_not_found", "detail": str(exc)}, err=True)
            return 4
        except (MergeError, ValueError) as exc:
            _emit({"error": "merge_query_failed", "detail": str(exc)},
                  err=True)
            return 1

    # query（既有行为：异常向上抛出由进程以非零码退出，保持原状）
    return _run_query(args)


if __name__ == "__main__":
    sys.exit(main())
