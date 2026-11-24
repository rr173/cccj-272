"""测试夹具：构造带目录的三轴分块数据集，及期望值/交付核对工具。"""
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chunk_export import Exporter, Request, load_catalog  # noqa: E402

FILL = -9


def value_fn(a, b, c):
    """可解码的确定性取值：值本身编码逻辑坐标。"""
    return a * 100 + b * 10 + c


def build_dataset(root, logical_shape, chunk_shape, *, dtype="int32",
                  fill_value=FILL, max_decode_elements=10 ** 6,
                  axis_orders=None, fill_cells=(), corrupt=(), omit=(),
                  unsampled=()):
    """在 root 下生成 chunks/*.npy 与 catalog.json，返回 catalog 路径。

    axis_orders: {(i,j,k): 文件轴序排列}，缺省 (0,1,2)
    fill_cells:  写入缺值标志的逻辑坐标（真实缺值）
    corrupt:     目录摘要写错的块（命中即拒交付）
    omit:        既不写入也不声明的块（命中即拒交付）
    unsampled:   明确标为未采的块（产缺值，不打开）
    """
    root = Path(root)
    (root / "chunks").mkdir(parents=True, exist_ok=True)
    grid_shape = tuple(-(-l // c) for l, c in zip(logical_shape, chunk_shape))
    dt = np.dtype(dtype)
    corrupt = {tuple(c) for c in corrupt}
    omit = {tuple(c) for c in omit}
    unsampled = {tuple(c) for c in unsampled}
    fill_cells = {tuple(c) for c in fill_cells}

    chunks = {}
    for idx in itertools.product(*[range(g) for g in grid_shape]):
        if idx in omit or idx in unsampled:
            continue
        region = [(i * cs, min((i + 1) * cs, l))
                  for i, cs, l in zip(idx, chunk_shape, logical_shape)]
        shape = tuple(hi - lo for lo, hi in region)
        arr = np.empty(shape, dtype=dt)
        for local in np.ndindex(*shape):
            coord = tuple(region[a][0] + local[a] for a in range(3))
            arr[local] = fill_value if coord in fill_cells else value_fn(*coord)
        order = tuple((axis_orders or {}).get(idx, (0, 1, 2)))
        stored = np.transpose(arr, axes=order)  # 文件轴 j 存逻辑轴 order[j]
        path = root / "chunks" / ("c%d_%d_%d.npy" % idx)
        np.save(path, stored)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if idx in corrupt:
            sha = "0" * 64  # 摘要对不上 => 坏块
        chunks["%d,%d,%d" % idx] = {
            "path": "chunks/" + path.name,
            "shape": list(shape),
            "file_axis_order": list(order),
            "value_count": int(arr.size),
            "sha256": sha,
        }

    catalog = {
        "logical_shape": list(logical_shape),
        "chunk_shape": list(chunk_shape),
        "dtype": dtype,
        "fill_value": fill_value,
        "max_decode_elements": max_decode_elements,
        "chunks": chunks,
        "unsampled": sorted(list(c) for c in unsampled),
    }
    catalog_path = root / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return catalog_path


def run_export(catalog_path, request, out_dir, **kw):
    cat = load_catalog(catalog_path)
    req = Request.from_dict(request)
    return Exporter(cat, req, out_dir, **kw).run()


def expected_target_grid(logical_shape, chunk_shape, request, *,
                         fill_value=FILL, unsampled=(), fill_cells=()):
    """完整逻辑网格 -> 取等差子窗 -> 换轴，得到目标网格期望值。"""
    grid = np.empty(logical_shape, dtype=np.int64)
    fill_cells = {tuple(c) for c in fill_cells}
    for coord in np.ndindex(*logical_shape):
        grid[coord] = fill_value if coord in fill_cells else value_fn(*coord)
    for idx in unsampled:
        slices = tuple(slice(i * c, min((i + 1) * c, l))
                       for i, c, l in zip(idx, chunk_shape, logical_shape))
        grid[slices] = fill_value
    coords = [np.array([a["start"] + i * a["step"] for i in range(a["count"])],
                       dtype=np.int64)
              for a in request["axes"]]
    sub = grid[np.ix_(*coords)]
    return np.transpose(sub, axes=request["target_axis_order"])


def check_delivery(manifest, out_dir, expected_grid, target_chunk_shape):
    """核对交付：块序为字典序、块内 C 序、边缘仅含实际点、数值一致。"""
    out_dir = Path(out_dir)
    metas = manifest["chunks"]
    assert [m["chunk"] for m in metas] == sorted(m["chunk"] for m in metas), \
        "交付顺序必须是目标块号字典序"
    dims = tuple(-(-g // c) for g, c in zip(expected_grid.shape, target_chunk_shape))
    assert len(metas) == dims[0] * dims[1] * dims[2]
    for m in metas:
        tc = m["chunk"]
        arr = np.load(out_dir / m["file"])
        slices = tuple(
            slice(tc[t] * target_chunk_shape[t],
                  min((tc[t] + 1) * target_chunk_shape[t], expected_grid.shape[t]))
            for t in range(3))
        want = expected_grid[slices]
        assert tuple(m["shape"]) == want.shape == arr.shape, f"块 {tc} 形状"
        assert arr.flags["C_CONTIGUOUS"], f"块 {tc} 必须末轴最快"
        np.testing.assert_array_equal(arr, want)
