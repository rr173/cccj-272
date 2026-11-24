"""第三轮：用 ResultIndex 构造暴力预言机，逐点核对合并值与获胜掩码。"""
import itertools
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path("/workspace/survey_exporter")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from chunk_export import MergeIndex, Merger, ResultIndex
from conftest import build_dataset, run_export

ROOT = Path("/workspace/verify/tmp3")
if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True)

L = (8, 6, 6)
CS = (3, 3, 3)
FILL = -7

# 三个采集现实：不同未采块 / 真实缺测
configs = [
    dict(run=1, unsampled=[(0, 0, 0), (2, 1, 1)],
         fill_cells=[(4, 4, 4), (1, 1, 1)]),
    dict(run=2, unsampled=[(1, 0, 1)],
         fill_cells=[(0, 0, 0), (7, 5, 5)]),
    dict(run=3, unsampled=[],
         fill_cells=[(3, 3, 3), (2, 0, 0)]),
]
cats = []
for i, cfg in enumerate(configs):
    # conftest 的 value_fn 固定编码坐标，无法区分 run；改用后处理：
    cp = build_dataset(ROOT / f"ds{i}", L, CS, fill_value=FILL,
                       unsampled=cfg["unsampled"],
                       fill_cells=cfg["fill_cells"])
    cats.append(cp)

# 为区分包，直接改写各数据集 chunk npy 的非 fill 值加包偏移，并重算摘要
import hashlib, json
for i, cfg in enumerate(configs):
    ds = ROOT / f"ds{i}"
    cpath = ds / "catalog.json"
    cat = json.loads(cpath.read_text())
    for key, spec in cat["chunks"].items():
        fp = ds / spec["path"]
        arr = np.load(fp)
        mask = arr != FILL
        arr2 = arr.astype(np.int64)
        arr2[mask] = arr2[mask] + cfg["run"] * 10000
        arr2 = arr2.astype(np.dtype(cat["dtype"]))
        np.save(fp, arr2)
        raw = fp.read_bytes()
        spec["sha256"] = hashlib.sha256(raw).hexdigest()
    cpath.write_text(json.dumps(cat))

requests = [
    {  # 包0：子窗 + 逆序 + 换轴
        "axes": [
            {"start": 7, "step": -2, "count": 4},   # 7,5,3,1
            {"start": 0, "step": 2, "count": 3},    # 0,2,4
            {"start": 5, "step": -2, "count": 3},   # 5,3,1
        ],
        "target_axis_order": [2, 0, 1],
        "target_chunk_shape": [2, 2, 2],
    },
    {  # 包1：几乎全量，另一种轴序/块形
        "axes": [
            {"start": 0, "step": 1, "count": 8},
            {"start": 5, "step": -1, "count": 6},
            {"start": 0, "step": 1, "count": 6},
        ],
        "target_axis_order": [1, 2, 0],
        "target_chunk_shape": [4, 3, 5],
    },
    {  # 包2：局部
        "axes": [
            {"start": 2, "step": 1, "count": 3},    # 2,3,4
            {"start": 0, "step": 1, "count": 6},
            {"start": 0, "step": 1, "count": 2},
        ],
        "target_axis_order": [0, 1, 2],
        "target_chunk_shape": [2, 6, 2],
    },
]

pkg_dirs = []
for i in range(3):
    d = ROOT / f"pkg{i}"
    run_export(cats[i], requests[i], d)
    pkg_dirs.append(d)

# ---------------- 暴力预言机 ----------------
exp_val = np.full(L, FILL, dtype=np.int64)
exp_win = np.full(L, -1, dtype=np.int64)
for pi, d in enumerate(pkg_dirs):
    ri = ResultIndex(d)
    man = json.loads((d / "manifest.json").read_text())
    for cm in man["chunks"]:
        tc = tuple(cm["chunk"])
        for off in range(int(cm["elements"])):
            ref = ri.locate(tc, off)
            if ref.origin == "data":
                exp_val[ref.coord] = ref.value
                exp_win[ref.coord] = pi

# ---------------- 合并（输出用非平凡轴序/块形） ----------------
Q = (2, 0, 1)
TCS = (3, 4, 2)
out = ROOT / "merged"
man = Merger(pkg_dirs, out, TCS, Q).run()
idx = MergeIndex(out)

out_grid = tuple(L[Q[t]] for t in range(3))
checked = 0
for coord in np.ndindex(*L):
    # 逻辑坐标 -> 输出目标坐标
    ov = tuple(coord[Q[t]] for t in range(3))
    tc = tuple(ov[t] // TCS[t] for t in range(3))
    loc = tuple(ov[t] % TCS[t] for t in range(3))
    meta = idx._chunks[tc]
    arr = np.load(out / meta["file"])
    warr = np.load(out / meta["provenance"])
    got_v = int(arr[loc])
    got_w = int(warr[loc])
    assert got_v == exp_val[coord], (coord, got_v, exp_val[coord])
    assert got_w == exp_win[coord], (coord, got_w, exp_win[coord])
    # MergeIndex 一致性
    shape = tuple(meta["shape"])
    off = loc[0] * shape[1] * shape[2] + loc[1] * shape[2] + loc[2]
    r = idx.locate(tc, off)
    assert r.value == exp_val[coord]
    assert r.input_index == (None if exp_win[coord] < 0 else int(exp_win[coord]))
    if exp_win[coord] < 0:
        assert r.origin == "unobserved"
    else:
        assert r.origin == "data"
        assert r.source_coord == coord
        assert r.winner == int(exp_win[coord])
        assert r.source == int(exp_win[coord])
        assert tuple(r.source) == coord
    checked += 1

print(f"oracle cross-check passed for all {checked} logical points")
assert man["stats"]["observed"] == int((exp_win >= 0).sum())
assert man["stats"]["unobserved"] == int((exp_win < 0).sum())

# ---------------- 构造形态 Merger(inputs, tcs, out) ----------------
out2 = ROOT / "merged_pos"
Merger(pkg_dirs, (4, 4, 4), out2).run()
m2 = json.loads((out2 / "manifest.json").read_text())
assert tuple(m2["target_chunk_shape"]) == (4, 4, 4)

# source 句柄与路径/序号比较
r = idx.locate((0, 0, 0), 0)
if r.input_index is not None:
    assert r.source == r.input_index
    assert r.source == r.input_path
    a, b, c = r.source
    assert (a, b, c) == r.coord
    assert r.source[0] == r.coord[0]

print("VERIFY3 PASSED")
