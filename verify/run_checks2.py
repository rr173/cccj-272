"""第二轮核查：换轴输出、包内反推、空包、链式合并、命名空间、旧查询不变。"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path("/workspace/survey_exporter")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from chunk_export import (MergeIndex, Merger, Request, ResultIndex,
                          load_catalog)
from chunk_export.cli import main as cli_main
from chunk_export.merge import MergeRequest
from conftest import build_dataset, run_export

ROOT = Path("/workspace/verify/tmp2")
PASS = 0


def check(name, cond, extra=""):
    global PASS
    assert cond, f"FAIL {name} {extra}"
    PASS += 1
    print("ok -", name)


if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True)

# 用 conftest 的标准数据集（value_fn 编码坐标）
LSHAPE = (5, 4, 6)
CSHAPE = (2, 2, 2)
cat = build_dataset(ROOT / "ds", LSHAPE, CSHAPE)

reqA = {
    "axes": [
        {"start": 4, "step": -2, "count": 3},   # 4,2,0
        {"start": 0, "step": 1, "count": 4},    # all b
        {"start": 5, "step": -2, "count": 3},   # 5,3,1
    ],
    "target_axis_order": [2, 0, 1],
    "target_chunk_shape": [2, 2, 2],
}
reqB = {
    "axes": [
        {"start": 0, "step": 1, "count": 5},
        {"start": 3, "step": -1, "count": 2},   # 3,2
        {"start": 0, "step": 1, "count": 6},
    ],
    "target_axis_order": [1, 2, 0],
    "target_chunk_shape": [3, 3, 3],
}
pA = ROOT / "A"
pB = ROOT / "B"
run_export(cat, reqA, pA)
run_export(cat, reqB, pB)

# 输出采用第三种轴序
out = ROOT / "merged_q"
Q = (1, 2, 0)
TCS = (2, 3, 2)
man = Merger([pA, pB], out, TCS, Q).run()
idx = MergeIndex(out)

# 独立计算每个逻辑坐标的期望获胜者：B 后，若 B 采样到则 B，否则 A，否则未观测
def sampled(req, coord):
    for a in range(3):
        ax = req["axes"][a]
        d = coord[a] - ax["start"]
        if ax["step"] == 0:
            return False
        if d % ax["step"] != 0:
            return False
        i = d // ax["step"]
        if not (0 <= i < ax["count"]):
            return False
    return True


n_obs = 0
for tc in idx.chunks:
    meta = idx._chunks[tc]
    shape = tuple(meta["shape"])
    for off in range(int(np.prod(shape))):
        r = idx.locate(tc, off)
        a, b, c = r.coord
        # 结果块/偏移与输出轴序一致
        ov = [tc[t] * TCS[t] + np.unravel_index(off, shape)[t]
              for t in range(3)]
        coord2 = [0, 0, 0]
        for t in range(3):
            coord2[Q[t]] = ov[t]
        assert tuple(coord2) == r.coord
        inA = sampled(reqA, (a, b, c))
        inB = sampled(reqB, (a, b, c))
        if inB:
            assert r.input_index == 1, (r, "B should win")
            assert r.origin == "data"
            n_obs += 1
            # 反推 B 包内块/偏移，并用 B 的 ResultIndex 交叉验证坐标/值
            bri = ResultIndex(pB)
            bref = bri.locate(tuple(r.input_chunk), r.input_offset)
            assert bref.coord == (a, b, c), (bref, r)
            assert bref.value == r.value
        elif inA:
            assert r.input_index == 0, (r, "A should win")
            ari = ResultIndex(pA)
            aref = ari.locate(tuple(r.input_chunk), r.input_offset)
            assert aref.coord == (a, b, c)
            assert aref.value == r.value == r.value
            n_obs += 1
        else:
            assert r.origin == "unobserved" and r.input_index is None, r
            assert r.value == -9 and r.is_fill
check("transposed output all points + reverse map cross-check", True,
      f"obs={n_obs}")
check("observed count matches union", man["stats"]["observed"] == n_obs,
      man["stats"])

# 输出网格形状 = 逻辑形状按 Q 置换
check("target grid permuted",
      tuple(man["target_grid_shape"]) == tuple(LSHAPE[Q[t]] for t in range(3)))

# ---------------------------------------------------------------
# 空结果包（count=0）不影响合并
# ---------------------------------------------------------------
req_empty = {
    "axes": [
        {"start": 999, "step": 1, "count": 0},
        {"start": 0, "step": 1, "count": 2},
        {"start": 0, "step": 1, "count": 2}],
    "target_axis_order": [0, 1, 2],
    "target_chunk_shape": [2, 2, 2],
}
pE = ROOT / "E"
run_export(cat, req_empty, pE)
out_e = ROOT / "merged_empty"
man_e = Merger([pE, pA], out_e, (2, 2, 2)).run()
check("empty package ok, A contributes",
      man_e["stats"]["observed"] > 0
      and man_e["stats"]["input_count"] == 2)

# 只有空包 -> 全未观测
out_e0 = ROOT / "merged_empty0"
man_e0 = Merger([pE], out_e0, (2, 2, 2)).run()
check("only empty package => all unobserved",
      man_e0["stats"]["observed"] == 0
      and man_e0["stats"]["unobserved"] == np.prod(LSHAPE))
idx0 = MergeIndex(out_e0)
r = idx0.locate((0, 0, 0), 0)
check("all-unobserved query", r.origin == "unobserved" and r.winner is None)

# ---------------------------------------------------------------
# 链式合并：把合并卷再次作为输入
# ---------------------------------------------------------------
out_chain = ROOT / "chain"
man_c = Merger([out], out_chain, (3, 3, 3), (0, 2, 1)).run()
idxc = MergeIndex(out_chain)
# 抽查若干点：链包中 observed 的点在新合并中获胜者序号为 0
for coord in [(0, 0, 0), (4, 3, 5), (2, 1, 3)]:
    a, b, c = coord
    q2 = (0, 2, 1)
    tcs2 = (3, 3, 3)
    v = [a, c, b]  # 输出目标坐标 v_t=coord[q2[t]]=(a,c,b)
    tc = tuple(v[t] // tcs2[t] for t in range(3))
    loc = tuple(v[t] % tcs2[t] for t in range(3))
    shape = tuple(min(tcs2[t],
                      tuple(LSHAPE[q2[t]] for t in range(3))[t]
                      - tc[t] * tcs2[t]) for t in range(3))
    off = loc[0] * shape[1] * shape[2] + loc[1] * shape[2] + loc[2]
    rc = idxc.locate(tc, off)
    check(f"chain point {coord}", rc.coord == coord
          and rc.origin in ("data", "unobserved"))

# ---------------------------------------------------------------
# 命名空间/导入别名
# ---------------------------------------------------------------
import chunk_export
from chunk_export.core import Merger as CoreMerger
from chunk_export.index import MergeIndex as IdxAlias
from chunk_export.request import MergeRequest as Req2
check("namespace aliases",
      CoreMerger is Merger and IdxAlias is MergeIndex and Req2 is MergeRequest
      and hasattr(chunk_export, "MergeIndex"))
try:
    import chunk_export.merge as m
    check("functional aliases",
          callable(m.merge_packages) and callable(m.merge_directories)
          and callable(m.MergedIndex) and callable(m.MergeResultIndex))
except Exception as e:
    check("functional aliases", False, str(e))

# ---------------------------------------------------------------
# 旧 query CLI 输出键集合严格不变
# ---------------------------------------------------------------
import io
import contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = cli_main(["query", "--export", str(pA),
                   "--chunk", "0,0,0", "--offset", "0"])
old = json.loads(buf.getvalue())
check("old query exact keys/rc",
      rc == 0 and set(old.keys()) == {"coord", "chunk", "origin",
                                     "value", "is_fill"}, old)

# 旧导出 CLI 摘要键不变
reqfile = ROOT / "req.json"
reqfile.write_text(json.dumps(reqA))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = cli_main(["export", "--catalog", str(cat),
                   "--request", str(reqfile),
                   "--out", str(ROOT / "cli_old_out")])
summ = json.loads(buf.getvalue())
check("old export summary keys",
      rc == 0 and set(summ.keys()) == {"out", "chunks",
                                       "source_chunks_opened",
                                       "bytes_read"}, summ)

# ---------------------------------------------------------------
# 越界查询行为
# ---------------------------------------------------------------
try:
    idx.locate((0, 0, 0), 10 ** 12)
    check("offset bounds raises", False)
except IndexError:
    check("offset bounds raises", True)
try:
    idx.locate((99, 99, 99), 0)
    check("unknown chunk raises", False)
except KeyError:
    check("unknown chunk raises", True)

# merge-query CLI 越界 -> rc4
rc = cli_main(["merge-query", "--merge", str(out),
               "--chunk", "0,0,0", "--offset", "99999999"])
check("merge-query out of range rc4", rc == 4, rc)

print(f"\nROUND2 ALL PASSED: {PASS}")
