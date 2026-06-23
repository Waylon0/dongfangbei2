"""断层多边形自动追踪 — 标准化算法入口（供 QProcess_Run_Python 框架调用）

接口规范（前端开发者只需阅读此 docstring 即可集成）:

    run(project_name, survey_name, grid_name, output_name,
        threshold_mode, otsu_scale, gaussian_sigma, closing_radius,
        opening_radius, min_polygon_area, dp_epsilon, polygon_mode)

参数说明:
    project_name  : str   — GeoEast 项目名
    survey_name   : str   — 工区名
    grid_name     : str   — 输入沿层属性网格名
    output_name   : str   — 输出断层多边形对象名
    threshold_mode: str   — 二值化模式 [otsu/fixed/adaptive/hysteresis]，默认 otsu
    otsu_scale    : float — Otsu 阈值缩放 (<1灵敏, >1保守)，范围 0.1~3.0，默认 0.8
    gaussian_sigma: float — 高斯滤波 σ，范围 0.0~10.0，默认 1.5
    closing_radius: int   — 闭运算半径，范围 0~20，默认 2
    opening_radius: int   — 开运算半径，范围 0~10，默认 1
    min_polygon_area: float — 最小多边形面积(像素²)，范围 0~500，默认 15
    dp_epsilon     : float — DP 简化容差(像素)，范围 0.1~20.0，默认 1.0
    polygon_mode   : str   — 多边形模式 [skeleton/region]，默认 skeleton

返回:
    算法结果写入 GeoEast 数据库，同时在 stdout 输出进度和统计信息。
    C++ 前端通过 QProcess::readyReadStandardOutput 读取。
"""

import sys
import os
import time
import json
import threading
import signal

# GeoEast DLL 路径
NGP_DIR = os.environ.get("NGP", r"D:\GeoEastRC\GeoEast-RC-V2.2")
GEOEAST_DIR = os.environ.get("GEOEAST", r"D:\GeoEastRC\iEcoV2.1")
os.environ.setdefault("NGP", NGP_DIR)
os.environ.setdefault("GEOEAST", GEOEAST_DIR)
os.environ.setdefault("PG_HOME", NGP_DIR)

_DPNAME = os.environ.get("NDP_DBNAME", "ndp")
_USER = os.environ.get("NDP_USER", "admin1")
_PASS = os.environ.get("NDP_PASS", "admin1")

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# ── 退出监听 ──

exit_requested = False

def _input_listener():
    global exit_requested
    while not exit_requested:
        try:
            line = sys.stdin.readline().strip()
            if line == "exit":
                print("exiting...", flush=True)
                exit_requested = True
                break
        except Exception:
            break

_input_thread = threading.Thread(target=_input_listener, daemon=True)
_input_thread.start()

def _handle_terminate(signum, frame):
    print("Termination signal received", flush=True)
    sys.exit(0)

signal.signal(signal.SIGTERM, _handle_terminate)


def run(project_name: str, survey_name: str = "survey1",
        grid_name: str = "fault_attr", output_name: str = "fault_polygons",
        threshold_mode: str = "otsu", otsu_scale: float = 0.8,
        gaussian_sigma: float = 1.5, closing_radius: int = 2,
        opening_radius: int = 1, min_polygon_area: float = 15.0,
        dp_epsilon: float = 1.0, polygon_mode: str = "skeleton"):
    """标准算法入口 — 读取 GeoEast 数据 → 运行追踪 → 写回结果。"""
    t0 = time.perf_counter()

    print(f"[pyalgo] 开始运行断层多边形追踪", flush=True)
    print(f"[pyalgo] 项目={project_name} 工区={survey_name} 网格={grid_name}", flush=True)

    # 1. 连接 GeoEast 数据库
    print(f"[pyalgo] 连接数据库 {_DPNAME} ...", flush=True)
    from pybo_importer import get_project
    project = get_project(project_name, _DPNAME, _USER, _PASS)

    # 2. 读取网格数据
    print(f"[pyalgo] 读取网格数据: {grid_name}", flush=True)
    from geoeast_io import read_grid_data, read_grid_geometry
    data = read_grid_data(project, grid_name)
    geom = read_grid_geometry(project, grid_name)
    print(f"[pyalgo] 网格尺寸: {data.shape[0]}×{data.shape[1]}", flush=True)
    print(f"[pyalgo] 值范围: [{data.min():.4f}, {data.max():.4f}]", flush=True)

    # 3. 配置参数
    from config import Config
    cfg = Config()
    cfg.threshold_mode = threshold_mode
    cfg.otsu_scale = otsu_scale
    cfg.gaussian_sigma = gaussian_sigma
    cfg.closing_radius = closing_radius
    cfg.opening_radius = opening_radius
    cfg.min_polygon_area = min_polygon_area
    cfg.dp_epsilon = dp_epsilon
    cfg.polygon_mode = polygon_mode
    cfg.use_confidence = True

    # 4. 运行算法
    print(f"[pyalgo] 运行断层追踪算法...", flush=True)
    if exit_requested:
        sys.exit(0)
    from pipeline import run_pipeline
    result = run_pipeline(data, cfg)
    polygons = result['filtered']
    areas = result['areas']
    confidences = result.get('confidences', [])

    elapsed = time.perf_counter() - t0
    print(f"[pyalgo] 检测到 {len(polygons)} 条断层多边形", flush=True)
    print(f"[pyalgo] 耗时: {elapsed:.1f}s", flush=True)
    if areas:
        print(f"[pyalgo] 面积范围: [{min(areas):.1f}, {max(areas):.1f}]", flush=True)

    # 5. 坐标转换 + 写回 GeoEast
    from geoeast_io import pixels_to_physical, write_fault_polygons
    polygons_phys = pixels_to_physical(polygons, geom)
    print(f"[pyalgo] 写回 GeoEast: {output_name}", flush=True)
    write_fault_polygons(project, output_name, polygons_phys, areas)
    print(f"[pyalgo] 写入完成!", flush=True)

    # 6. 输出统计结果（JSON，方便 C++ 前端解析）
    stats = {
        "status": "success",
        "count": len(polygons),
        "elapsed": round(elapsed, 2),
        "total_area": round(result.get('total_area', 0), 2),
        "total_length": round(result.get('total_length', 0), 2),
        "min_area": round(result.get('min_area', 0), 2),
        "max_area": round(result.get('max_area', 0), 2),
    }
    if confidences:
        high = sum(1 for c in confidences if c >= 0.6)
        med = sum(1 for c in confidences if 0.3 <= c < 0.6)
        low = sum(1 for c in confidences if c < 0.3)
        stats["confidence_high"] = high
        stats["confidence_medium"] = med
        stats["confidence_low"] = low
    print(f"[pyalgo] RESULT_JSON: {json.dumps(stats, ensure_ascii=False)}", flush=True)

    return stats


# 模块自测
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="断层多边形追踪 — 标准化入口")
    parser.add_argument("--project", required=True, help="GeoEast 项目名")
    parser.add_argument("--survey", default="survey1", help="工区名")
    parser.add_argument("--grid", default="fault_attr", help="输入网格名")
    parser.add_argument("--output", default="fault_polygons", help="输出对象名")
    parser.add_argument("--threshold-mode", default="otsu")
    parser.add_argument("--otsu-scale", type=float, default=0.8)
    parser.add_argument("--gaussian-sigma", type=float, default=1.5)
    parser.add_argument("--closing-radius", type=int, default=2)
    parser.add_argument("--opening-radius", type=int, default=1)
    parser.add_argument("--min-polygon-area", type=float, default=15.0)
    parser.add_argument("--dp-epsilon", type=float, default=1.0)
    parser.add_argument("--polygon-mode", default="skeleton")
    args = parser.parse_args()

    run(args.project, args.survey, args.grid, args.output,
        args.threshold_mode, args.otsu_scale, args.gaussian_sigma,
        args.closing_radius, args.opening_radius, args.min_polygon_area,
        args.dp_epsilon, args.polygon_mode)
