"""合并能力的端到端核查脚本（开发自检，不进仓库 tests/）。"""
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path("/workspace/survey_exporter")
sys.path.insert(0, str(REPO / "src"))

from chunk_export import (Exporter, MergeIndex, Merger, Request, ResultIndex,
                          merge_exports)
from chunk_export.cli import main as cli_main
from chunk_export.errors import DeliveryRefused, IncompatiblePackages

FILL = -9
ROOT = Path("/workspace/verify/tmp")


def val(run, a, b, c):
    return run * 100000 + a * 1000 + b * 10 + c


def build_ds(root, logical_shape, chunk_shape, run, *, dtype="int32",
             fill_value=FILL, unsampled=(), fill_cells=(), corrupt=(),
             omit=(), max_decode_elements=10 ** 9):
    root = Path(root)
    (root / "chunks").mkdir(parents=True, exist_ok=True)
    grid = tuple(-(-l // c) for l, c in zip(logical_shape, chunk_shape))
    dt = np.dtype(dtype)
    unsampled = {tuple(x) for x in unsampled}
    fill_cells = {tuple(x) for x in fill_cells}
    corrupt = {tuple(x) for x in corrupt}
    omit = {tuple(x) for x in omit}
    chunks = {}
    for idx in itertools.product(*[range(g) for g in grid]):
        if idx in unsampled or idx in omit:
            continue
        region = [(i * cs, min((i + 1) * cs, l))
                  for i, cs, l in zip(idx, chunk_shape, logical_shape)]
        shape = tuple(hi - lo for lo, hi in region)
        arr = np.empty(shape, dtype=dt)
        for local in np.ndindex(*shape):
            coord = tuple(region[a][0] + local[a] for a in range(3))
            arr[local] = fill_value if coord in fill_cells \
                else val(run, *coord)
        path = root / "chunks" / ("c%d_%d_%d.npy" % idx)
        np.save(path, arr)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if idx in corrupt:
            sha = "0" * 64
        chunks["%d,%d,%d" % idx] = {
            "path": "chunks/" + path.name,
            "shape": list(shape),
            "file_axis_order": [0, 1, 2],
            "value_count": int(arr.size),
            "sha256": sha,
        }
    cat = {
        "logical_shape": list(logical_shape),
        "chunk_shape": list(chunk_shape),
        "dtype": dtype,
        "fill_value": fill_value,
        "max_decode_elements": max_decode_elements,
        "chunks": chunks,
        "unsampled": [list(x) for x in sorted(unsampled)],
    }
    cp = root / "catalog.json"
    cp.write_text(json.dumps(cat), encoding="utf-8")
    return cp


def do_export(cat, req, out):
    return Exporter(__import__("chunk_export").load_catalog(cat),
                    Request.from_dict(req), out).run()


def expected_merge(logical_shape, runs_spec, out_order):
    """runs_spec: 按优先级顺序 [(run, coverage coords-set or 'all', fills, unsamp_chunks)]

    简化：直接用每包导出时的请求/数据集在测试主体里算期望。
    """
    pass


PASS = 0


def check(name, cond, extra=""):
    global PASS
    assert cond, f"FAIL {name} {extra}"
    PASS += 1
    print(f"ok - {name}")


def main():
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)

    LSHAPE = (6, 4, 4)
    CSHAPE = (2, 2, 2)

    # ---------------------------------------------------------------
    # 场景 1：两块互补窗口 + 不同目标轴序/块形，合成完整卷
    # ---------------------------------------------------------------
    d0 = build_ds(ROOT / "ds0", LSHAPE, CSHAPE, run=1)
    d1 = build_ds(ROOT / "ds1", LSHAPE, CSHAPE, run=2)
    req0 = {
        "axes": [{"start": 0, "step": 1, "count": 3},
                 {"start": 0, "step": 1, "count": 4},
                 {"start": 0, "step": 1, "count": 4}],
        "target_axis_order": [0, 1, 2],
        "target_chunk_shape": [2, 2, 2],
    }
    req1 = {
        "axes": [{"start": 5, "step": -1, "count": 3},   # 5,4,3
                 {"start": 3, "step": -1, "count": 4},   # 3,2,1,0
                 {"start": 0, "step": 1, "count": 4}],
        "target_axis_order": [2, 0, 1],
        "target_chunk_shape": [3, 2, 3],
    }
    p0 = ROOT / "pkg0"
    p1 = ROOT / "pkg1"
    do_export(d0, req0, p0)
    do_export(d1, req1, p1)

    merged = ROOT / "merged"
    man = Merger([p0, p1], merged, (3, 2, 2)).run()
    check("manifest format", man["format"] == "chunk-merge/1")
    check("stats full observed",
          man["stats"]["observed"] == 6 * 4 * 4,
          man["stats"])

    idx = MergeIndex(merged)
    check("chunk count", len(idx.chunks) == len(man["chunks"]))
    # 逐点核对值/获胜者/源坐标
    n_checked = 0
    for tc in idx.chunks:
        meta = idx._chunks[tc]
        arr = np.load(merged / meta["file"])
        warr = np.load(merged / meta["provenance"])
        for off in range(int(np.prod(meta["shape"]))):
            ref = idx.locate(tc, off)
            a, b, c = ref.coord
            run = 1 if a < 3 else 2
            expect_v = val(run, a, b, c)
            check_ = (ref.value == expect_v
                      and ref.origin == "data"
                      and ref.input_index == run - 1
                      and ref.source_coord == (a, b, c)
                      and ref.chunk == (a // 2, b // 2, c // 2)
                      and not ref.is_fill)
            if not check_:
                raise AssertionError(
                    f"point {ref.coord} tc={tc} off={off} "
                    f"got v={ref.value} win={ref.input_index} "
                    f"origin={ref.origin} expect={expect_v}")
            # ResultIndex 委托
            n_checked += 1
    check("all 96 points verified", n_checked == 96, n_checked)

    ridx = ResultIndex(merged)
    check("ResultIndex delegates to MergeIndex",
          isinstance(ridx, MergeIndex))
    ref = ridx.locate((0, 0, 0), 0)
    check("delegated locate winner chameleon int", ref.winner == 0)
    check("delegated locate winner path", ref.winner == str(p0))
    check("winner attr aliases", ref.input_index == 0
          and ref.winner_index == 0 and ref.package_index == 0
          and ref.source_index == 0 and ref.index == 0)

    # 与 check_delivery 风格的全网格期望比对（逻辑轴序输出）
    expected = np.empty(LSHAPE, dtype="int64")
    for a, b, c in np.ndindex(*LSHAPE):
        expected[a, b, c] = val(1 if a < 3 else 2, a, b, c)
    tcs = (3, 2, 2)
    for cm in man["chunks"]:
        tc = tuple(cm["chunk"])
        arr = np.load(merged / cm["file"])
        sl = tuple(slice(tc[t] * tcs[t],
                         min((tc[t] + 1) * tcs[t], LSHAPE[t]))
                   for t in range(3))
        np.testing.assert_array_equal(arr, expected[sl])
        check("chunk shape/order", tuple(cm["shape"]) == arr.shape
              and arr.flags["C_CONTIGUOUS"])

    # ---------------------------------------------------------------
    # 场景 2：优先级、真实缺测覆盖、未采不覆盖、未观测区分
    # ---------------------------------------------------------------
    L2 = (6, 2, 2)
    # 块按 a 分为 0:(0,1) 1:(2,3) 2:(4,5)
    # dsA(run=7): 块0未采；(4,0,0) 真实缺测；其余实采
    dA = build_ds(ROOT / "dsA", L2, CSHAPE, run=7,
                  unsampled=[(0, 0, 0)], fill_cells=[(4, 0, 0)])
    # dsB(run=8): 块2未采；(2,0,0) 真实缺测；其余实采
    dB = build_ds(ROOT / "dsB", L2, CSHAPE, run=8,
                  unsampled=[(2, 0, 0)], fill_cells=[(2, 0, 0)])
    # dsC(run=9): 完整实采数据集，只导出一个点
    dC = build_ds(ROOT / "dsC", L2, CSHAPE, run=9)
    dD = build_ds(ROOT / "dsD", L2, CSHAPE, run=10)

    full_req = {
        "axes": [{"start": 0, "step": 1, "count": 6},
                 {"start": 0, "step": 1, "count": 2},
                 {"start": 0, "step": 1, "count": 2}],
        "target_axis_order": [0, 1, 2],
        "target_chunk_shape": [4, 2, 2],
    }
    qA = ROOT / "qA"
    qB = ROOT / "qB"
    do_export(dA, full_req, qA)
    do_export(dB, full_req, qB)

    reqC = {
        "axes": [{"start": 0, "step": 1, "count": 1},
                 {"start": 0, "step": 1, "count": 1},
                 {"start": 1, "step": -1, "count": 1}],  # (0,0,1)
        "target_axis_order": [2, 1, 0],
        "target_chunk_shape": [1, 1, 1],
    }
    qC = ROOT / "qC"
    do_export(dC, reqC, qC)

    # 再加一个只覆盖 (1,1,0) 的包 D，但该点在 A 中实采 -> 验证普通优先
    dD = build_ds(ROOT / "dsD", L2, CSHAPE, run=10)
    reqD = {
        "axes": [{"start": 1, "step": 1, "count": 1},
                 {"start": 1, "step": 1, "count": 1},
                 {"start": 0, "step": 1, "count": 1}],  # (1,1,0)
        "target_axis_order": [0, 1, 2],
        "target_chunk_shape": [1, 1, 1],
    }
    qD = ROOT / "qD"
    do_export(dD, reqD, qD)

    m2dir = ROOT / "m2"
    # 顺序 A, B, C, D（靠后优先）
    man2 = Merger([qA, qB, qC, qD], m2dir, [4, 2, 2]).run()
    idx2 = MergeIndex(m2dir)

    def loc(a, b, c):
        # 输出 tcs (4,2,2)：tc0 覆盖 a0..3，tc1 边缘块覆盖 a4..5
        tc = (a // 4, 0, 0)
        la = a % 4
        off = la * 4 + b * 2 + c
        return idx2.locate(tc, off)

    # (0,0,0): A 未采；B 实采(run8) -> B 获胜
    r = loc(0, 0, 0)
    check("A-unsamp covered by B-data",
          r.input_index == 1 and r.origin == "data"
          and r.value == val(8, 0, 0, 0), r)
    # (0,0,1): C 更晚实采 -> C 获胜
    r = loc(0, 0, 1)
    check("later C wins", r.input_index == 2
          and r.value == val(9, 0, 0, 1), r)
    check("winner chameleon path C", r.winner == str(qC))
    # (1,1,0): D 最晚 -> D
    r = loc(1, 1, 0)
    check("later D wins", r.input_index == 3
          and r.value == val(10, 1, 1, 0), r)
    check("input chunk reverse map", r.input_chunk is not None
          and r.input_offset == 0, (r.input_chunk, r.input_offset))
    # (2,0,0): A 普通实采；B 中是真实缺测(fill)，B 晚 -> B 获胜且真实缺测
    r = loc(2, 0, 0)
    check("later real fill overwrites",
          r.input_index == 1 and r.origin == "data"
          and r.value == FILL and r.is_fill and r.observed, r)
    # (4,0,0): A 中是真实缺测 FILL；B 中该块未采 -> A 保持获胜
    r = loc(4, 0, 0)
    check("B-unsamp must not overwrite A real-fill",
          r.input_index == 0 and r.origin == "data"
          and r.value == FILL and r.is_fill, r)
    # (5,1,1): B 块2未采；A 实采普通值 -> A 获胜
    r = loc(5, 1, 1)
    check("B-unsamp keeps A data",
          r.input_index == 0 and r.value == val(7, 5, 1, 1)
          and r.origin == "data" and not r.is_fill, r)
    # (3,1,1): 两包都实采 -> 靠后的 B 获胜
    r = loc(3, 1, 1)
    check("normal later priority",
          r.input_index == 1 and r.value == val(8, 3, 1, 1), r)

    # 未观测场景：只合并 qC（单点），其余点未观测
    m3 = ROOT / "m3"
    Merger([qC], m3, [2, 2, 2]).run()
    idx3 = MergeIndex(m3)
    # (0,0,1) offset 1 in chunk(0,0,0) shape(2,2,2)
    r = idx3.locate((0, 0, 0), 1)
    check("single observed point",
          r.origin == "data" and r.input_index == 0
          and r.value == val(9, 0, 0, 1), r)
    r = idx3.locate((0, 0, 0), 0)  # (0,0,0)
    check("unobserved distinguished",
          r.origin == "unobserved" and not r.observed
          and r.input_index is None and r.winner is None
          and r.value == FILL and r.is_fill
          and r.source_coord is None, r)
    # 其它块里的点也应是未观测
    r = idx3.locate((1, 0, 0), 0)  # (2,0,0)
    check("unobserved in other chunk",
          r.origin == "unobserved" and r.value == FILL, r)

    # ---------------------------------------------------------------
    # 场景 3：CLI merge + query（含 JSON 字段）
    # ---------------------------------------------------------------
    cli_out = ROOT / "cli_merged"
    rc = cli_main(["merge", "--inputs", str(p0), str(p1),
                   "--out", str(cli_out),
                   "--target-chunk-shape", "3,2,2"])
    check("cli merge rc0", rc == 0, rc)
    rc = cli_main(["query", "--export", str(cli_out),
                   "--chunk", "0,0,0", "--offset", "0"])
    check("cli query merged rc0", rc == 0)
    rc = cli_main(["merge-query", "--merge", str(cli_out),
                   "--chunk", "0,0,0", "--offset", "0"])
    check("cli merge-query rc0", rc == 0)
    # 位置参数风格
    cli_out2 = ROOT / "cli_merged2"
    rc = cli_main(["merge", str(p0), str(p1), "--out", str(cli_out2),
                   "--target-axis-order", "[2,0,1]",
                   "--target-chunk-shape", "[2,2,3]"])
    check("cli positional merge rc0", rc == 0, rc)
    # 重复 --input 风格
    cli_out3 = ROOT / "cli_merged3"
    rc = cli_main(["merge", "--input", str(p0), "--input", str(p1),
                   "--out", str(cli_out3), "--chunk-shape", "2x2x2"])
    check("cli repeated --input rc0", rc == 0, rc)

    # 模块子进程调用
    proc = subprocess.run(
        [sys.executable, "-m", "chunk_export", "merge",
         "--inputs", str(p0), str(p1),
         "--out", str(ROOT / "cli_sub"),
         "--target-chunk-shape", "2,2,2"],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO / "src") + ":/workspace/pylibs",
             "PATH": "/usr/bin:/bin"})
    check("subprocess -m merge rc0", proc.returncode == 0,
          proc.stderr + proc.stdout)

    # ---------------------------------------------------------------
    # 场景 4：不兼容 -> 拒绝，不留目录
    # ---------------------------------------------------------------
    d_bad_shape = build_ds(ROOT / "ds_bad_shape", (4, 2, 2), CSHAPE, run=3)
    pb = ROOT / "pkg_bad_shape"
    req4 = {
        "axes": [{"start": 0, "step": 1, "count": 4},
                 {"start": 0, "step": 1, "count": 2},
                 {"start": 0, "step": 1, "count": 2}],
        "target_axis_order": [0, 1, 2],
        "target_chunk_shape": [4, 2, 2],
    }
    do_export(d_bad_shape, req4, pb)
    out_bad = ROOT / "bad_merge"
    try:
        Merger([qA, pb], out_bad, [2, 2, 2]).run()
        check("logical shape incompat raises", False)
    except IncompatiblePackages:
        check("logical shape incompat raises", True)
    check("no partial output on incompat", not out_bad.exists())

    d_bad_dt = build_ds(ROOT / "ds_bad_dt", L2, CSHAPE, run=0, dtype="int16")
    pd = ROOT / "pkg_bad_dt"
    req16 = dict(full_req)
    do_export(d_bad_dt, req16, pd)
    out_bad = ROOT / "bad_merge_dt"
    try:
        Merger([qA, pd], out_bad, [2, 2, 2]).run()
        check("dtype incompat raises", False)
    except IncompatiblePackages:
        check("dtype incompat raises", True)
    check("no partial output dtype", not out_bad.exists())

    d_bad_fill = build_ds(ROOT / "ds_bad_fill", L2, CSHAPE, run=3,
                          fill_value=-1)
    pf = ROOT / "pkg_bad_fill"
    do_export(d_bad_fill, full_req, pf)
    out_bad = ROOT / "bad_merge_fill"
    try:
        Merger([qA, pf], out_bad, [2, 2, 2]).run()
        check("fill incompat raises", False)
    except IncompatiblePackages:
        check("fill incompat raises", True)
    check("no partial output fill", not out_bad.exists())

    d_bad_grid = build_ds(ROOT / "ds_bad_grid", L2, (1, 2, 2), run=3)
    pg = ROOT / "pkg_bad_grid"
    reqg = {
        "axes": [{"start": 0, "step": 1, "count": 4},
                 {"start": 0, "step": 1, "count": 2},
                 {"start": 0, "step": 1, "count": 2}],
        "target_axis_order": [0, 1, 2],
        "target_chunk_shape": [4, 2, 2],
    }
    do_export(d_bad_grid, reqg, pg)
    out_bad = ROOT / "bad_merge_grid"
    try:
        Merger([qA, pg], out_bad, [2, 2, 2]).run()
        check("grid incompat raises", False)
    except IncompatiblePackages:
        check("grid incompat raises", True)
    check("no partial output grid", not out_bad.exists())

    # CLI 不兼容退出码 2
    rc = cli_main(["merge", "--inputs", str(qA), str(pf),
                   "--out", str(ROOT / "cli_bad"),
                   "--target-chunk-shape", "2,2,2"])
    check("cli incompat rc2", rc == 2, rc)
    check("cli incompat no dir", not (ROOT / "cli_bad").exists())

    # ---------------------------------------------------------------
    # 场景 5：损坏包 -> DeliveryRefused / rc3 / 不留目录
    # ---------------------------------------------------------------
    corrupt_pkg = ROOT / "qA_corrupt"
    shutil.copytree(qA, corrupt_pkg)
    chunk_file = corrupt_pkg / "chunks" / "t0_0_0.npy"
    chunk_file.write_bytes(b"this is not an npy file")
    out_c = ROOT / "corrupt_merge"
    try:
        Merger([qB, corrupt_pkg], out_c, [2, 2, 2]).run()
        check("corrupt package refuses", False)
    except DeliveryRefused as e:
        check("corrupt package refuses", e.reason == "corrupt", str(e))
    check("no partial output on corrupt", not out_c.exists())

    rc = cli_main(["merge", "--inputs", str(qB), str(corrupt_pkg),
                   "--out", str(ROOT / "cli_corrupt"),
                   "--target-chunk-shape", "2,2,2"])
    check("cli corrupt rc3", rc == 3, rc)
    check("cli corrupt no dir", not (ROOT / "cli_corrupt").exists())

    # 字节翻转：新导出清单带 sha256，应被摘要核验抓到
    flip_pkg = ROOT / "p0_flip"
    shutil.copytree(p0, flip_pkg)
    p0_man = json.loads((p0 / "manifest.json").read_text())
    cf = flip_pkg / p0_man["chunks"][0]["file"]
    b = bytearray(cf.read_bytes())
    b[len(b) // 2] ^= 0xFF
    cf.write_bytes(bytes(b))
    out_f = ROOT / "flip_merge"
    try:
        Merger([flip_pkg, p1], out_f, [3, 2, 2]).run()
        check("byte flip detected via sha", False)
    except DeliveryRefused as e:
        check("byte flip detected via sha", e.reason == "corrupt")
    check("no partial output flip", not out_f.exists())

    # 清单缺失/格式不明
    weird = ROOT / "weird"
    weird.mkdir()
    (weird / "manifest.json").write_text(json.dumps({"format": "nope/9"}))
    try:
        Merger([weird], ROOT / "weird_out", [2, 2, 2]).run()
        check("unknown format refuses", False)
    except IncompatiblePackages:
        check("unknown format refuses", True)

    # ---------------------------------------------------------------
    # 场景 6：函数式 API + 同义构造形态
    # ---------------------------------------------------------------
    out_fn = ROOT / "fn_merge"
    mfn = merge_exports([p0, p1], out_fn, target_chunk_shape=(4, 4, 2))
    check("functional api", (out_fn / "manifest.json").exists())
    out_kw = ROOT / "kw_merge"
    Merger(input_dirs=[p0, p1], output=out_kw,
           chunk_shape=(2, 4, 2)).run()
    check("kw alias api", (out_kw / "manifest.json").exists())

    # MergeRequest 位置形态 Merger(inputs, req, out)
    from chunk_export import MergeRequest
    out_req = ROOT / "req_merge"
    Merger([p0, p1], MergeRequest((2, 2, 2)), out_req).run()
    check("request positional shape", (out_req / "manifest.json").exists())

    print(f"\nALL CHECKS PASSED: {PASS}")


if __name__ == "__main__":
    main()
