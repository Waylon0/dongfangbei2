"""将赛题 test.dat 导入 GeoEast 数据库。

用法:
    D:\GeoEastRC\support\miniconda3\envs\nv\python.exe import_test_data.py
"""

import sys
import time
import numpy as np

sys.path.insert(0, r"d:\东方杯-v2\src")
from pybo_importer import get_root, get_project, pybo

# 配置
PROJECT_NAME = "DFB_S7"          # 东方杯赛题7 项目
SURVEY_NAME = "survey1"          # 工区
GRID_NAME = "fault_attr"         # 沿层属性网格名
DAT_FILE = r"references\test.dat"


def main():
    t0 = time.perf_counter()

    # ── 1. 解析 test.dat ──────────────────────────────────────────
    print("[1/4] 读取 test.dat ...")
    raw = np.loadtxt(DAT_FILE, skiprows=1)  # (N, 5): line, cmp, x, y, value
    lines = raw[:, 0].astype(np.int32)
    cmps = raw[:, 1].astype(np.int32)
    x_vals = raw[:, 2]
    y_vals = raw[:, 3]
    values = raw[:, 4]

    unique_lines = np.unique(lines)
    unique_cmps = np.unique(cmps)
    n_lines = len(unique_lines)
    n_cmps = len(unique_cmps)

    line_to_idx = {int(l): i for i, l in enumerate(unique_lines)}
    cmp_to_idx = {int(c): j for j, c in enumerate(unique_cmps)}

    # 原始: row=Line (X), col=CMP (Y) → (1501, 701)
    grid_orig = np.zeros((n_lines, n_cmps), dtype=np.float64)
    for k in range(len(raw)):
        li = line_to_idx[int(lines[k])]
        cj = cmp_to_idx[int(cmps[k])]
        grid_orig[li, cj] = values[k]

    # 转置为 GeoEast 标准: row=CMP (Y), col=Line (X) → (701, 1501)
    # 使得 X 沿列方向、Y 沿行方向
    grid_2d = grid_orig.T  # shape: (n_cmps, n_lines)
    ny, nx = grid_2d.shape  # ny=701, nx=1501

    dx = 10.0   # X 间距 (每列 = 每 Line 步)
    dy = 10.0   # Y 间距 (每行 = 每 CMP 步)
    sx = float(np.round(x_vals.min(), 2))
    sy = float(np.round(y_vals.min(), 2))

    print(f"    原始: {n_lines} Lines x {n_cmps} CMPs")
    print(f"    GeoEast网格: {ny} 行 x {nx} 列")
    print(f"    X: {sx} ~ {x_vals.max():.1f}  dx={dx}")
    print(f"    Y: {sy} ~ {y_vals.max():.1f}  dy={dy}")
    print(f"    值范围: [{values.min():.4f}, {values.max():.4f}]")
    print(f"    非零占比: {(values > 0.001).sum() / len(values) * 100:.1f}%")

    # ── 2. 创建项目 + 工区 ──────────────────────────────────────
    print("[2/4] 创建 GeoEast 项目 ...")
    root = get_root()
    if root.hasProject(PROJECT_NAME):
        print(f"    项目 {PROJECT_NAME} 已存在，复用")
    else:
        root.createProject(PROJECT_NAME)
        print(f"    已创建项目: {PROJECT_NAME}")

    project = root.getProject(PROJECT_NAME)

    surveys = project.listSurvey()
    if any(s.getName() == SURVEY_NAME for s in surveys):
        print(f"    工区 {SURVEY_NAME} 已存在，复用")
    else:
        project.createSurvey(SURVEY_NAME)
        print(f"    已创建工区: {SURVEY_NAME}")

    # ── 3. 写入网格数据 ─────────────────────────────────────────
    print(f"[3/4] 构建网格数据 ({ny} x {nx} = {ny * nx} 点) ...")

    if project.hasMapGrid(GRID_NAME):
        project.eraseMapGrid(GRID_NAME)
    grid_obj = project.createMapGrid(GRID_NAME)

    head = pybo.PyBAMapGridHead()
    head.nx = nx
    head.ny = ny
    head.dx = dx
    head.dy = dy
    head.sx = sx
    head.sy = sy
    head.minValue = float(values.min())
    head.maxValue = float(values.max())
    head.surveyName = SURVEY_NAME

    # 构建 2D 点数组
    data_2d = []
    report_every = max(1, ny // 10)
    for row in range(ny):
        row_pts = []
        row_y = sy + row * dy       # 每行 Y 固定 (CMP 方向)
        for col in range(nx):
            pt = pybo.PyBAMapGridPoint()
            pt.x = sx + col * dx    # X 随列变化 (Line 方向)
            pt.y = row_y
            pt.z = float(grid_2d[row, col])
            row_pts.append(pt)
        data_2d.append(row_pts)
        if (row + 1) % report_every == 0:
            pct = (row + 1) / ny * 100
            print(f"    ... {pct:.0f}%")

    print("    正在写入数据库 (saveData)...")
    grid_obj.saveData(head, data_2d)
    print(f"    写入完成: {ny} x {nx} = {ny * nx} 点")

    # ── 4. 验证 ──────────────────────────────────────────────────
    print("[4/4] 验证 ...")
    grid2 = project.getMapGrid(GRID_NAME)
    grid2.readData()
    d2 = grid2.getData()
    h2 = grid2.getDataHead()

    if hasattr(d2, 'data'):
        arr = np.array(d2.data, dtype=np.float64).reshape(h2.ny, h2.nx)
    elif hasattr(d2, 'getData'):
        arr = np.array(d2.getData(), dtype=np.float64).reshape(h2.ny, h2.nx)
    else:
        print("    [!] 无法解析验证数据")
        arr = None

    if arr is not None:
        diff = np.abs(arr - grid_2d).max()
        print(f"    读回尺寸: {arr.shape[0]} x {arr.shape[1]}")
        print(f"    最大误差: {diff:.6f}")
        if diff < 1e-6:
            print("    验证通过!")

    elapsed = time.perf_counter() - t0
    print(f"\n导入完成，总耗时 {elapsed:.1f}s")
    print(f"项目名: {PROJECT_NAME}")
    print(f"网格名: {GRID_NAME}")
    print(f"\n下一步:")
    print(f"  D:\\GeoEastRC\\support\\miniconda3\\envs\\nv\\python.exe run_geoeast.py --project {PROJECT_NAME} --grid {GRID_NAME} --output fault_polygons")


if __name__ == "__main__":
    main()
