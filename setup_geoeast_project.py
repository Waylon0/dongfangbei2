r"""一键搭建 GeoEast 项目 — 创建项目、导入数据、运行算法、写回结果。

用法:
    D:\GeoEastRC\support\miniconda3\envs\nv\python.exe setup_geoeast_project.py
"""

import sys
import time
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from pybo_importer import get_root, get_project, pybo

PROJECT_NAME = "DFB_S7_WZY"
SURVEY_NAME = "survey1"
GRID_NAME = "fault_attr"
OUTPUT_NAME = "fault_polygons"
DAT_FILE = os.path.join(os.path.dirname(__file__), "references", "test.dat")


def step1_parse_dat():
    print("=" * 60)
    print("[1/6] 解析 test.dat ...")
    raw = np.loadtxt(DAT_FILE, skiprows=1)
    lines = raw[:, 0].astype(np.int32)
    cmps = raw[:, 1].astype(np.int32)
    x_vals = raw[:, 2]
    values = raw[:, 4]

    unique_lines = np.unique(lines)
    unique_cmps = np.unique(cmps)
    n_lines = len(unique_lines)
    n_cmps = len(unique_cmps)

    line_to_idx = {int(l): i for i, l in enumerate(unique_lines)}
    cmp_to_idx = {int(c): j for j, c in enumerate(unique_cmps)}

    grid_orig = np.zeros((n_lines, n_cmps), dtype=np.float64)
    for k in range(len(raw)):
        li = line_to_idx[int(lines[k])]
        cj = cmp_to_idx[int(cmps[k])]
        grid_orig[li, cj] = values[k]

    grid_2d = grid_orig.T
    ny, nx = grid_2d.shape
    dx = 10.0
    dy = 10.0
    sx = float(np.round(x_vals.min(), 2))
    sy = float(np.round(values[np.argmin(x_vals)], 2))

    # 实际 sx/sy 从坐标直接获取
    sy_val = float(np.round(raw[np.argmin(raw[:, 1]), 3], 2))
    sx_val = float(np.round(raw[np.argmin(raw[:, 0]), 2], 2))

    print(f"    网格: {ny} rows x {nx} cols")
    print(f"    X: {sx_val} ~ {x_vals.max():.1f}  dx={dx}")
    print(f"    Y: {sy_val} ~ {raw[:, 3].max():.1f}  dy={dy}")
    print(f"    值范围: [{values.min():.4f}, {values.max():.4f}]")

    return grid_2d, nx, ny, dx, dy, sx_val, sy_val, values.min(), values.max()


def step2_create_project(root):
    print("=" * 60)
    print(f"[2/6] 创建项目: {PROJECT_NAME}")

    if root.hasProject(PROJECT_NAME):
        print(f"    项目已存在，删除重建...")
        root.eraseProject(PROJECT_NAME)

    root.createProject(PROJECT_NAME)
    print(f"    项目已创建: {PROJECT_NAME}")

    project = root.getProject(PROJECT_NAME)

    if any(s.getName() == SURVEY_NAME for s in project.listSurvey()):
        print(f"    工区已存在: {SURVEY_NAME}")
    else:
        project.createSurvey(SURVEY_NAME)
        print(f"    工区已创建: {SURVEY_NAME}")

    return project


def step3_import_grid(project, grid_2d, nx, ny, dx, dy, sx, sy, vmin, vmax):
    print("=" * 60)
    print(f"[3/6] 导入网格数据: {GRID_NAME} ({ny}x{nx}={ny*nx} 点)")

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
    head.minValue = float(vmin)
    head.maxValue = float(vmax)
    head.surveyName = SURVEY_NAME

    data_2d = []
    for row in range(ny):
        row_pts = []
        row_y = sy + row * dy
        for col in range(nx):
            pt = pybo.PyBAMapGridPoint()
            pt.x = sx + col * dx
            pt.y = row_y
            pt.z = float(grid_2d[row, col])
            row_pts.append(pt)
        data_2d.append(row_pts)
        if (row + 1) % 200 == 0:
            print(f"    ... {int((row+1)/ny*100)}%")

    grid_obj.saveData(head, data_2d)
    print(f"    写入完成!")


def step4_run_algorithm(grid_2d):
    print("=" * 60)
    print("[4/6] 运行断层多边形追踪...")

    from src.pipeline import run_pipeline
    from config import Config

    cfg = Config()
    result = run_pipeline(grid_2d, cfg)

    polygons_px = result['filtered']
    areas = result['areas']
    print(f"    检测到 {len(polygons_px)} 个断层多边形")
    print(f"    耗时: {result['elapsed']}s")
    if areas:
        print(f"    面积范围: [{min(areas):.1f}, {max(areas):.1f}]")

    return polygons_px, areas


def step5_write_results(project, polygons_px, areas):
    print("=" * 60)
    print(f"[5/6] 写入断层多边形: {OUTPUT_NAME}")

    from src.geoeast_io import write_fault_polygons

    # 像素坐标 → 物理坐标
    geom = {'sx': 622993.9, 'sy': 4334420.1, 'dx': 10.0, 'dy': 10.0,
            'nx': 1501, 'ny': 701}
    from src.geoeast_io import pixels_to_physical
    polygons_phys = pixels_to_physical(polygons_px, geom)

    write_fault_polygons(project, OUTPUT_NAME, polygons_phys, areas)
    print(f"    已写入 {len(polygons_phys)} 条断层多边形")


def step6_verify(project):
    print("=" * 60)
    print("[6/6] 验证...")

    from src.geoeast_io import read_fault_polygons
    try:
        results = read_fault_polygons(project, OUTPUT_NAME)
        print(f"    读回 {len(results)} 条断层多边形 — 验证成功!")
    except Exception as e:
        print(f"    验证跳过: {e}")

    # 统计
    grids = project.listMapGrid()
    faults = project.listFaultPolygon()
    surveys = project.listSurvey()
    print(f"\n项目 [{PROJECT_NAME}] 包含:")
    print(f"  工区: {[s.getName() for s in surveys]}")
    print(f"  网格: {[g.getName() for g in grids]}")
    print(f"  断层多边形: {[f.getName() for f in faults]}")


def main():
    t0 = time.perf_counter()
    print("GeoEast 项目一键搭建 & 断层多边形追踪")
    print(f"项目: {PROJECT_NAME}  |  数据库: ndp  |  用户: wzy")
    print()

    grid_2d, nx, ny, dx, dy, sx, sy, vmin, vmax = step1_parse_dat()

    root = get_root()
    project = step2_create_project(root)

    step3_import_grid(project, grid_2d, nx, ny, dx, dy, sx, sy, vmin, vmax)

    polygons_px, areas = step4_run_algorithm(grid_2d)

    step5_write_results(project, polygons_px, areas)

    step6_verify(project)

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 60}")
    print(f"全部完成! 总耗时 {elapsed:.1f}s")
    print(f"\n在 GeoEast 中查看:")
    print(f"  1. 文件 → 打开已有项目 → 选择 ndp → {PROJECT_NAME}")
    print(f"  2. 左侧数据树 → Fault Polygons → {OUTPUT_NAME}")
    print(f"  3. 右键 → LayerDisplay")
    print(f"  4. 或在 构造解释 → 工区底图 中拖入断层多边形图层")


if __name__ == "__main__":
    main()
