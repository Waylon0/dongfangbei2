"""GeoEast 集成入口 — 命令行调用

用法:
    python run.py --project "xx工区" --grid "沿层属性"
    python run.py --project "xx工区" --grid "沿层属性" --output "断层多边形结果"
    python run.py --project "xx工区" --grid "沿层属性" --config my_config.py

GeoEast 主控通过 QProcess 调用此脚本，传入数据库连接参数。
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="基于平面断层属性的断层多边形自动追踪 (GeoEast 集成版)",
    )
    parser.add_argument(
        "--project", required=True,
        help="GeoEast 项目名",
    )
    parser.add_argument(
        "--grid", required=True,
        help="沿层属性网格数据名称",
    )
    parser.add_argument(
        "--output", default=None,
        help="输出断层多边形名称（默认: {grid}_fault_polygon）",
    )
    parser.add_argument(
        "--config", default=None,
        help="自定义 Config .py 文件路径",
    )
    args = parser.parse_args()

    # 加载配置
    cfg = Config()
    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("user_config", args.config)
        user_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_module)
        for attr in dir(user_module):
            if not attr.startswith('_') and hasattr(cfg, attr):
                setattr(cfg, attr, getattr(user_module, attr))

    output_name = args.output or f"{args.grid}_fault_polygon"

    # --- PyBO 读取数据 ---
    print(f"[run.py] 连接 GeoEast 项目: {args.project}")
    print(f"[run.py] 读取网格数据: {args.grid}")

    try:
        from src.geoeast_io import read_grid_data, write_fault_polygons, \
            list_grids, remove_fault_polygon

        # TODO: 实际 PyBO 项目对象由 GeoEast 主控传入或通过环境变量获取
        # project = connect_to_geoeast_project(args.project)
        # data = read_grid_data(project, args.grid)
        raise NotImplementedError(
            "PyBO 环境未就绪。请安装 GeoEast 后:\n"
            "  1. 在 geoeast_io.py 中实现 PyBO 读写函数\n"
            "  2. 在此处获取 BOProject 对象\n"
            "  3. 删除此 raise 语句"
        )

    except NotImplementedError as e:
        print(f"[run.py] 错误: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 运行流水线 ---
    print(f"[run.py] 运行断层追踪流水线...")
    result = run_pipeline(data, cfg)
    print(f"[run.py] 完成，耗时 {result['elapsed']:.3f}s")
    print(f"[run.py] 提取到 {len(result['filtered'])} 条断层多边形")

    # --- PyBO 写回结果 ---
    print(f"[run.py] 写入断层多边形: {output_name}")
    try:
        n_written = write_fault_polygons(None, output_name,  # TODO: 替换 project
                                         result['filtered'],
                                         result['areas'])
        print(f"[run.py] 成功写入 {n_written} 条断层多边形")
    except NotImplementedError as e:
        print(f"[run.py] 错误: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[run.py] 完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
