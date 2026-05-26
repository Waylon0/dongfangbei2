"""GeoEast 集成入口 — 读取属性数据，运行断层追踪，写回结果。

两种输入模式:
  --file test.dat  → 直接从 .dat 文件读取 (推荐，PyBO 网格读接口受限)
  --grid 网格名    → 从 GeoEast 数据库读取 (需要 PyBO 支持)

用法:
    D:\GeoEastRC\support\miniconda3\envs\nv\python.exe run_geoeast.py --list-projects
    D:\GeoEastRC\support\miniconda3\envs\nv\python.exe run_geoeast.py --project DFB_S7 --file references\test.dat --output fault_polygons
    D:\GeoEastRC\support\miniconda3\envs\nv\python.exe run_geoeast.py --project DFB_S7 --file references\test.dat --output fault_polygons --visualize
"""

import argparse
import sys
import time
import os

import numpy as np

# matplotlib 必须在 pybo_importer 之前导入
# GeoEast bin/ 下的 libexpat.dll 会污染 PATH，导致 pyexpat 加载失败
import matplotlib
matplotlib.use('Agg')  # 非交互后端，保存为 PNG 文件
import matplotlib.pyplot as plt

from src.pybo_importer import get_root, get_project
from src.geoeast_io import write_fault_polygons, list_grids, pixels_to_physical
from src.pipeline import run_pipeline
from config import Config


# ── test.dat 网格参数（由 import_test_data.py 分析得出） ──────
GRID_PARAMS = {
    'nx': 1501, 'ny': 701,
    'dx': 10.0, 'dy': 10.0,
    'sx': 622993.9, 'sy': 4334420.1,
}


def load_dat_file(filepath: str):
    """读取赛题 test.dat，返回 (data_2d, geometry_dict)。"""
    raw = np.loadtxt(filepath, skiprows=1)  # (N, 5): line, cmp, x, y, value
    lines = raw[:, 0].astype(np.int32)
    cmps = raw[:, 1].astype(np.int32)
    values = raw[:, 4]

    unique_lines = np.unique(lines)
    unique_cmps = np.unique(cmps)
    n_lines = len(unique_lines)
    n_cmps = len(unique_cmps)

    line_to_idx = {int(l): i for i, l in enumerate(unique_lines)}
    cmp_to_idx = {int(c): j for j, c in enumerate(unique_cmps)}

    # 原始: row=Line (X), col=CMP (Y) → 转置为 row=CMP (Y), col=Line (X)
    grid_orig = np.zeros((n_lines, n_cmps), dtype=np.float64)
    for k in range(len(raw)):
        li = line_to_idx[int(lines[k])]
        cj = cmp_to_idx[int(cmps[k])]
        grid_orig[li, cj] = values[k]
    grid_2d = grid_orig.T  # (701, 1501)

    geom = dict(GRID_PARAMS)
    return grid_2d, geom


def cmd_list_projects(dpname: str, user: str, password: str):
    """列出所有 GeoEast 项目。"""
    root = get_root(dpname, user, password)
    projects = root.listProject()
    if not projects:
        print("(无项目)")
        return
    print(f"{'项目名':<40} {'类型'}")
    print("-" * 60)
    for p in projects:
        name = p.getName()
        try:
            ptype = p.getType()
        except Exception:
            ptype = "-"
        print(f"{name:<40} {ptype}")


def cmd_list_grids(dpname: str, user: str, password: str, project_name: str):
    """列出项目下所有网格数据。"""
    project = get_project(project_name, dpname, user, password)
    grids = list_grids(project)
    if not grids:
        print("(无网格数据)")
        return
    print(f"项目 [{project_name}] 下的网格数据:")
    for g in grids:
        print(f"  - {g}")


def cmd_run(dpname: str, user: str, password: str,
            project_name: str, output_name: str,
            file_path: str = None, grid_name: str = None,
            visualize: bool = False):
    """运行完整流水线：读数据 → 断层追踪 → 写回 GeoEast。"""
    t0 = time.perf_counter()

    # 1. 连接项目
    print(f"[1/5] 连接项目: {project_name}")
    project = get_project(project_name, dpname, user, password)

    # 2. 读取数据
    if file_path:
        print(f"[2/5] 读取数据文件: {file_path}")
        data, geom = load_dat_file(file_path)
    elif grid_name:
        print(f"[2/5] 读取 GeoEast 网格: {grid_name}")
        from src.geoeast_io import read_grid_data, read_grid_geometry
        data = read_grid_data(project, grid_name)
        geom = read_grid_geometry(project, grid_name)
    else:
        print("错误: 需要 --file 或 --grid 指定输入数据")
        sys.exit(1)

    print(f"      网格尺寸: {data.shape[0]} rows x {data.shape[1]} cols")
    print(f"      物理范围: X=[{geom['sx']:.1f}, {geom['sx'] + geom['nx'] * geom['dx']:.1f}]")
    print(f"                Y=[{geom['sy']:.1f}, {geom['sy'] + geom['ny'] * geom['dy']:.1f}]")
    print(f"      值范围: [{data.min():.4f}, {data.max():.4f}]")

    # 3. 运行断层追踪
    print(f"[3/5] 运行断层追踪...")
    cfg = Config()
    result = run_pipeline(data, cfg)

    polygons_px = result['filtered']
    areas = result['areas']
    polygons = pixels_to_physical(polygons_px, geom)
    print(f"      耗时: {result['elapsed']}s")
    print(f"      检测到 {len(polygons)} 个断层多边形")
    if areas:
        print(f"      面积范围: [{min(areas):.1f}, {max(areas):.1f}]")
        print(f"      总面积: {sum(areas):.1f}")

    # 4. 写回 GeoEast
    print(f"[4/5] 写入断层多边形到 GeoEast: {output_name}")
    write_fault_polygons(project, output_name, polygons, areas)
    print(f"      写入 {len(polygons)} 个多边形")

    # 5. 验证
    print(f"[5/5] 验证读取...")
    from src.geoeast_io import read_fault_polygons
    try:
        verify = read_fault_polygons(project, output_name)
        print(f"      读回 {len(verify)} 个多边形")
        if len(verify) == len(polygons):
            print(f"      数量一致，写入成功!")
    except Exception as e:
        print(f"      验证跳过: {e}")

    elapsed = time.perf_counter() - t0
    print(f"\n完成，总耗时 {elapsed:.1f}s")

    if visualize and polygons:
        _quick_view(data, polygons, geom, output_name)


def _quick_view(data, polygons, geom, title):
    """保存可视化结果到 PNG 文件。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14)

    ax1.imshow(data, cmap='seismic', aspect='auto')
    ax1.set_title("Attribute Data")
    ax1.axis('off')

    ax2.imshow(data, cmap='seismic', aspect='auto')
    sx, sy = geom['sx'], geom['sy']
    dx, dy = geom['dx'], geom['dy']
    for poly in polygons:
        if len(poly) >= 2:
            col = (poly[:, 0] - sx) / dx
            row = (poly[:, 1] - sy) / dy
            ax2.plot(col, row, 'lime', linewidth=1.2)
    ax2.set_title(f"Fault Polygons ({len(polygons)} detected)")
    ax2.axis('off')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'output', f'{title}.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n可视化已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="GeoEast 断层多边形追踪 — PyBO 集成入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --list-projects
  %(prog)s --list-grids --project DFB_S7
  %(prog)s --project DFB_S7 --file references\\test.dat --output fault_polygons
  %(prog)s --project DFB_S7 --file references\\test.dat --output fault_polygons --visualize
        """,
    )

    parser.add_argument("--project", "-p", help="GeoEast 项目名")
    parser.add_argument("--file", "-f", help="输入数据文件路径 (.dat)")
    parser.add_argument("--grid", "-g", help="GeoEast 网格名 (PyBO 读受限，推荐用 --file)")
    parser.add_argument("--output", "-o", help="输出断层多边形对象名")
    parser.add_argument("--dpname", default="ndp", help="数据平台名 (default: ndp)")
    parser.add_argument("--user", default="admin1", help="数据库用户名")
    parser.add_argument("--password", default="admin1", help="数据库密码")
    parser.add_argument("--list-projects", action="store_true", help="列出所有项目")
    parser.add_argument("--list-grids", action="store_true", help="列出项目下的网格数据")
    parser.add_argument("--visualize", "-V", action="store_true",
                        help="运行后用 matplotlib 快速预览结果")

    args = parser.parse_args()

    if args.list_projects:
        cmd_list_projects(args.dpname, args.user, args.password)
    elif args.list_grids:
        if not args.project:
            print("错误: --list-grids 需要指定 --project")
            sys.exit(1)
        cmd_list_grids(args.dpname, args.user, args.password, args.project)
    elif args.project and args.output and (args.file or args.grid):
        cmd_run(args.dpname, args.user, args.password,
                args.project, args.output,
                file_path=args.file, grid_name=args.grid,
                visualize=args.visualize)
    else:
        parser.print_help()
        print("\n提示: 需要指定 --project, --output, 以及 --file 或 --grid")


if __name__ == "__main__":
    main()
