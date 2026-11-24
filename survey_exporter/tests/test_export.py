"""夹具：逆序跳过坏块、换轴、跨目标块复用、读块核账，及各类拒绝语义。"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import (FILL, build_dataset, check_delivery,
                      expected_target_grid, run_export, value_fn)

from chunk_export import Request, ResultIndex, load_catalog
from chunk_export.cli import main as cli_main
from chunk_export.errors import DeliveryRefused, RequestInvalid


# ---------------------------------------------------------- 逆序跳过坏块

def _reverse_skip_request():
    # 轴0 坐标 9,6,3,0 -> 块 4,3,1,0，跳过坏块 (2,0,0)（坐标 4..5）
    return {
        "axes": [
            {"start": 9, "step": -3, "count": 4},
            {"start": 0, "step": 1, "count": 2},
            {"start": 1, "step": -1, "count": 2},
        ],
        "target_axis_order": [0, 1, 2],
        "target_chunk_shape": [3, 2, 1],
    }


def test_reverse_window_skips_corrupt_chunk(tmp_path):
    catalog = build_dataset(tmp_path / "ds", (10, 2, 2), (2, 2, 2),
                            corrupt=[(2, 0, 0)])
    req = _reverse_skip_request()
    out = tmp_path / "out"
    manifest = run_export(catalog, req, out)

    # 核账：包围盒 [0..9] 穿过坏块 (2,0,0)，但没采到它的点，绝不允许打开
    opened = [tuple(e["chunk"]) for e in manifest["accounting"]["opened"]]
    assert opened == [(0, 0, 0), (1, 0, 0), (3, 0, 0), (4, 0, 0)]
    assert (2, 0, 0) not in opened
    assert manifest["accounting"]["open_count"] == len(set(opened)) == 4
    assert manifest["accounting"]["bytes_read"] == sum(
        e["bytes"] for e in manifest["accounting"]["opened"]) > 0

    expected = expected_target_grid((10, 2, 2), (2, 2, 2), req)
    check_delivery(manifest, out, expected, req["target_chunk_shape"])


def test_hit_corrupt_chunk_refuses_and_reports_block(tmp_path):
    catalog = build_dataset(tmp_path / "ds", (10, 2, 2), (2, 2, 2),
                            corrupt=[(2, 0, 0)])
    req = _reverse_skip_request()
    req["axes"][0] = {"start": 8, "step": -3, "count": 3}  # 坐标 8,5,2 -> 命中块 2
    out = tmp_path / "out"
    with pytest.raises(DeliveryRefused) as ei:
        run_export(catalog, req, out)
    assert ei.value.reason == "corrupt"
    assert ei.value.chunk == (2, 0, 0)  # 报逻辑块号
    assert not out.exists()             # 拒绝交付，不留半成品


# ---------------------------------------------------------- 未声明块

def test_undeclared_chunk_refuses_only_when_hit(tmp_path):
    catalog = build_dataset(tmp_path / "ds", (10, 2, 2), (2, 2, 2),
                            omit=[(1, 0, 0)])
    # 跳过未声明块：坐标 9,5,1 -> 块 4,2,0
    req = _reverse_skip_request()
    req["axes"][0] = {"start": 9, "step": -4, "count": 3}
    manifest = run_export(catalog, req, tmp_path / "out")
    opened = {tuple(e["chunk"]) for e in manifest["accounting"]["opened"]}
    assert opened == {(0, 0, 0), (2, 0, 0), (4, 0, 0)}

    # 命中未声明块：坐标 9,6,3 -> 块 4,3,1
    req["axes"][0] = {"start": 9, "step": -3, "count": 3}
    with pytest.raises(DeliveryRefused) as ei:
        run_export(catalog, req, tmp_path / "out2")
    assert ei.value.reason == "undeclared"
    assert ei.value.chunk == (1, 0, 0)
    assert not (tmp_path / "out2").exists()


# ---------------------------------------------------------- 换轴与文件轴序

def test_axis_transpose_with_mixed_file_orders(tmp_path):
    perms = list(__import__("itertools").permutations((0, 1, 2)))
    grid_dims = (3, 2, 3)  # (5,4,6)/(2,2,2)
    axis_orders = {
        idx: perms[sum(idx) % 6]
        for idx in __import__("itertools").product(*[range(g) for g in grid_dims])
    }
    catalog = build_dataset(tmp_path / "ds", (5, 4, 6), (2, 2, 2),
                            axis_orders=axis_orders)
    req = {
        "axes": [
            {"start": 4, "step": -2, "count": 3},  # 4,2,0
            {"start": 1, "step": 1, "count": 3},   # 1,2,3
            {"start": 5, "step": -2, "count": 3},  # 5,3,1
        ],
        "target_axis_order": [2, 0, 1],
        "target_chunk_shape": [2, 2, 2],
    }
    out = tmp_path / "out"
    manifest = run_export(catalog, req, out)
    expected = expected_target_grid((5, 4, 6), (2, 2, 2), req)
    assert expected.shape == (3, 3, 3)
    check_delivery(manifest, out, expected, req["target_chunk_shape"])

    # 同一数据集再换一种目标轴序与目标块形（含边缘）
    req2 = dict(req, target_axis_order=[1, 2, 0], target_chunk_shape=[3, 1, 2])
    manifest2 = run_export(catalog, req2, tmp_path / "out2")
    expected2 = expected_target_grid((5, 4, 6), (2, 2, 2), req2)
    check_delivery(manifest2, tmp_path / "out2", expected2, req2["target_chunk_shape"])


@pytest.mark.parametrize("perm", [(0, 1, 2), (0, 2, 1), (1, 0, 2),
                                  (1, 2, 0), (2, 0, 1), (2, 1, 0)])
def test_single_chunk_every_file_axis_order(tmp_path, perm):
    catalog = build_dataset(tmp_path / "ds", (2, 3, 4), (2, 3, 4),
                            axis_orders={(0, 0, 0): perm})
    req = {
        "axes": [
            {"start": 1, "step": -1, "count": 2},
            {"start": 0, "step": 2, "count": 2},
            {"start": 3, "step": -1, "count": 4},
        ],
        "target_axis_order": [0, 1, 2],
        "target_chunk_shape": [2, 2, 4],
    }
    manifest = run_export(catalog, req, tmp_path / "out")
    expected = expected_target_grid((2, 3, 4), (2, 3, 4), req)
    check_delivery(manifest, tmp_path / "out", expected, req["target_chunk_shape"])


# ---------------------------------------------------------- 跨目标块复用 + 核账

def test_source_chunk_opened_once_across_target_chunks(tmp_path):
    catalog = build_dataset(tmp_path / "ds", (4, 4, 4), (4, 4, 4))
    req = {
        "axes": [
            {"start": 3, "step": -1, "count": 4},
            {"start": 0, "step": 1, "count": 4},
            {"start": 3, "step": -1, "count": 4},
        ],
        "target_axis_order": [1, 2, 0],
        "target_chunk_shape": [2, 3, 2],  # 8 个目标块共用 1 个源块
    }
    out = tmp_path / "out"
    manifest = run_export(catalog, req, out)

    opened = manifest["accounting"]["opened"]
    assert [tuple(e["chunk"]) for e in opened] == [(0, 0, 0)]  # 只开一次
    assert manifest["accounting"]["open_count"] == 1
    chunk_file = tmp_path / "ds" / "chunks" / "c0_0_0.npy"
    assert opened[0]["bytes"] == chunk_file.stat().st_size
    assert manifest["accounting"]["bytes_read"] == opened[0]["bytes"]
    assert opened[0]["elements"] == 64

    expected = expected_target_grid((4, 4, 4), (4, 4, 4), req)
    check_delivery(manifest, out, expected, req["target_chunk_shape"])
    assert len(manifest["chunks"]) == 8


# ---------------------------------------------------------- 缺值来源区分

def test_unsampled_vs_real_fill_origin(tmp_path):
    catalog = build_dataset(
        tmp_path / "ds", (4, 2, 2), (2, 2, 2),
        unsampled=[(0, 0, 0)], fill_cells=[(3, 0, 0)])
    req = {
        "axes": [
            {"start": 0, "step": 1, "count": 4},
            {"start": 0, "step": 1, "count": 2},
            {"start": 0, "step": 1, "count": 2},
        ],
        "target_axis_order": [0, 1, 2],
        "target_chunk_shape": [4, 2, 2],
    }
    out = tmp_path / "out"
    manifest = run_export(catalog, req, out)
    assert manifest["source"]["unsampled_chunks"] == [[0, 0, 0]]
    # 未采块不得打开
    opened = {tuple(e["chunk"]) for e in manifest["accounting"]["opened"]}
    assert opened == {(1, 0, 0)}

    expected = expected_target_grid((4, 2, 2), (2, 2, 2), req,
                                    unsampled=[(0, 0, 0)], fill_cells=[(3, 0, 0)])
    check_delivery(manifest, out, expected, req["target_chunk_shape"])

    idx = ResultIndex(out)
    # (0,1,1) 在未采块 (0,0,0)：值是缺值，来源 unsampled
    ref = idx.locate((0, 0, 0), offset=0 * 4 + 1 * 2 + 1)
    assert ref.coord == (0, 1, 1) and ref.chunk == (0, 0, 0)
    assert ref.origin == "unsampled" and ref.value == FILL and ref.is_fill
    # (3,0,0) 在实采块 (1,0,0)：真实缺值，来源 data
    ref = idx.locate((0, 0, 0), offset=3 * 4 + 0 * 2 + 0)
    assert ref.coord == (3, 0, 0) and ref.chunk == (1, 0, 0)
    assert ref.origin == "data" and ref.value == FILL and ref.is_fill
    # (2,1,1) 普通实采点
    ref = idx.locate((0, 0, 0), offset=2 * 4 + 1 * 2 + 1)
    assert ref.coord == (2, 1, 1) and ref.chunk == (1, 0, 0)
    assert ref.origin == "data" and ref.value == value_fn(2, 1, 1) and not ref.is_fill


# ---------------------------------------------------------- 空结果与请求校验

def test_zero_count_delivers_empty_but_validates(tmp_path):
    catalog = build_dataset(tmp_path / "ds", (4, 2, 2), (2, 2, 2))
    req = {
        "axes": [
            {"start": 999, "step": 1, "count": 0},  # 无坐标，不验范围
            {"start": 0, "step": 1, "count": 2},
            {"start": 1, "step": -1, "count": 2},
        ],
        "target_axis_order": [0, 1, 2],
        "target_chunk_shape": [2, 2, 2],
    }
    out = tmp_path / "out"
    manifest = run_export(catalog, req, out)
    assert manifest["chunks"] == []
    assert manifest["accounting"]["open_count"] == 0
    assert (out / "manifest.json").exists()

    bad_axes = [
        [{"start": 0, "step": 0, "count": 0}, *req["axes"][1:]],   # 步长 0 仍拒
        [{"start": True, "step": 1, "count": 0}, *req["axes"][1:]],  # 布尔仍拒
        [{"start": 0, "step": 1, "count": -1}, *req["axes"][1:]],   # 负点数
        [{"start": 0, "step": 1, "count": 1.5}, *req["axes"][1:]],  # 非整数
        [{"start": "0", "step": 1, "count": 0}, *req["axes"][1:]],  # 字符串
    ]
    for axes in bad_axes:
        with pytest.raises(RequestInvalid):
            Request.from_dict(dict(req, axes=axes))


def test_request_validation(tmp_path):
    base = {
        "axes": [
            {"start": 0, "step": 1, "count": 2},
            {"start": 0, "step": 1, "count": 2},
            {"start": 0, "step": 1, "count": 2},
        ],
        "target_axis_order": [0, 1, 2],
        "target_chunk_shape": [2, 2, 2],
    }
    bad = [
        dict(base, target_axis_order=[0, 0, 1]),
        dict(base, target_axis_order=[0, 1]),
        dict(base, target_chunk_shape=[0, 1, 2]),
        dict(base, target_chunk_shape=[2, 2]),
        dict(base, axes=[{"start": 0, "step": 1, "count": 2}] * 2),
    ]
    for raw in bad:
        with pytest.raises(RequestInvalid):
            Request.from_dict(raw)

    catalog = load_catalog(build_dataset(tmp_path / "ds", (4, 2, 2), (2, 2, 2)))
    out_of_range = [
        [{"start": 3, "step": 1, "count": 2}, *base["axes"][1:]],   # 4 越界
        [{"start": 0, "step": -1, "count": 2}, *base["axes"][1:]],  # -1 越界
        [base["axes"][0], {"start": 1, "step": 2, "count": 2}, base["axes"][2]],  # 3 越界
    ]
    for axes in out_of_range:
        req = Request.from_dict(dict(base, axes=axes))
        with pytest.raises(RequestInvalid):
            req.validate_range(catalog.logical_shape)


# ---------------------------------------------------------- 解码上限

def test_oversize_chunk_refused_only_when_hit(tmp_path):
    # 块 (0,0,0) 形状 (4,2,2)=16 元素 > 上限 8；块 (1,0,0) 为 (2,2,2)=8
    catalog = build_dataset(tmp_path / "ds", (6, 2, 2), (4, 2, 2),
                            max_decode_elements=8)
    req = {
        "axes": [
            {"start": 5, "step": -1, "count": 2},  # 坐标 5,4 -> 只命中块 1
            {"start": 0, "step": 1, "count": 2},
            {"start": 0, "step": 1, "count": 2},
        ],
        "target_axis_order": [0, 1, 2],
        "target_chunk_shape": [2, 2, 2],
    }
    manifest = run_export(catalog, req, tmp_path / "out")
    opened = {tuple(e["chunk"]) for e in manifest["accounting"]["opened"]}
    assert opened == {(1, 0, 0)}  # 超限块未命中，不打开也不阻塞

    req["axes"][0] = {"start": 2, "step": 1, "count": 1}  # 命中超限块 0
    with pytest.raises(DeliveryRefused) as ei:
        run_export(catalog, req, tmp_path / "out2")
    assert ei.value.reason == "oversize"
    assert ei.value.chunk == (0, 0, 0)


# ---------------------------------------------------------- 边缘块与交付顺序

def test_edge_chunks_and_lexicographic_delivery(tmp_path):
    catalog = build_dataset(tmp_path / "ds", (5, 3, 4), (2, 2, 3))
    req = {
        "axes": [
            {"start": 4, "step": -1, "count": 5},
            {"start": 0, "step": 1, "count": 3},
            {"start": 0, "step": 1, "count": 4},
        ],
        "target_axis_order": [0, 1, 2],
        "target_chunk_shape": [2, 2, 2],
    }
    out = tmp_path / "out"
    manifest = run_export(catalog, req, out)
    expected = expected_target_grid((5, 3, 4), (2, 2, 3), req)
    check_delivery(manifest, out, expected, req["target_chunk_shape"])
    shapes = {tuple(m["chunk"]): tuple(m["shape"]) for m in manifest["chunks"]}
    assert shapes[(2, 1, 1)] == (1, 1, 2)  # 三轴边缘同时截断


# ---------------------------------------------------------- CLI 与溯源查询

def test_cli_export_query_and_refusal(tmp_path, capsys):
    catalog = build_dataset(tmp_path / "ds", (4, 2, 2), (2, 2, 2),
                            unsampled=[(0, 0, 0)])
    req = {
        "axes": [
            {"start": 0, "step": 1, "count": 4},
            {"start": 0, "step": 1, "count": 2},
            {"start": 0, "step": 1, "count": 2},
        ],
        "target_axis_order": [0, 1, 2],
        "target_chunk_shape": [4, 2, 2],
    }
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req), encoding="utf-8")
    out = tmp_path / "out"

    rc = cli_main(["export", "--catalog", str(catalog),
                   "--request", str(req_path), "--out", str(out)])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["chunks"] == 1 and summary["source_chunks_opened"] == 1

    rc = cli_main(["query", "--export", str(out), "--chunk", "0,0,0",
                   "--offset", "0"])
    assert rc == 0
    ref = json.loads(capsys.readouterr().out)
    assert ref == {"coord": [0, 0, 0], "chunk": [0, 0, 0],
                   "origin": "unsampled", "value": FILL, "is_fill": True}

    # 命中坏块：退出码 3，stderr 带逻辑块号
    catalog2 = build_dataset(tmp_path / "ds2", (4, 2, 2), (2, 2, 2),
                             corrupt=[(1, 0, 0)])
    rc = cli_main(["export", "--catalog", str(catalog2),
                   "--request", str(req_path), "--out", str(tmp_path / "out2")])
    assert rc == 3
    err = json.loads(capsys.readouterr().err)
    assert err["error"] == "delivery_refused"
    assert err["reason"] == "corrupt" and err["chunk"] == [1, 0, 0]


def test_cli_module_invocation(tmp_path):
    catalog = build_dataset(tmp_path / "ds", (2, 2, 2), (2, 2, 2))
    req = {
        "axes": [{"start": 1, "step": -1, "count": 2}] * 3,
        "target_axis_order": [2, 1, 0],
        "target_chunk_shape": [1, 1, 2],
    }
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req), encoding="utf-8")
    src = Path(__file__).resolve().parent.parent / "src"
    proc = subprocess.run(
        [sys.executable, "-m", "chunk_export", "export",
         "--catalog", str(catalog), "--request", str(req_path),
         "--out", str(tmp_path / "out")],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(src), "PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["chunks"] == 4
