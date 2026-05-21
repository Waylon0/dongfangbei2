"""任务2：用真实 test.dat 数据跑一遍流水线，快速验证算法泛化能力

105万点全读太慢 → 裁中心区域快速验证，秒级跑完。
想看全图效果：把 CROP 改成 None
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from config import Config
from src.preprocess import load_attribute_data
from src.pipeline import run_pipeline
from src.vectorize import export_geojson

# 裁切范围：None=全图, (r1, r2, c1, c2)=子区域
CROP = (500, 900, 150, 550)  # 中心 400x400

print("=" * 50)
print("任务2：真实数据验证")
print("=" * 50)

# 1. 加载
print("\n[1/3] 加载 data/test.dat ...")
t0 = time.perf_counter()
data_full = load_attribute_data("data/test.dat")
t_load = time.perf_counter() - t0
print(f"  全图尺寸: {data_full.shape[0]} x {data_full.shape[1]} ({data_full.shape[0]*data_full.shape[1]:,} 点)")
print(f"  全图值域: [{data_full.min():.4f}, {data_full.max():.4f}]")

if CROP:
    r1, r2, c1, c2 = CROP
    data = data_full[r1:r2, c1:c2].copy()
    print(f"  裁切区域: [{r1}:{r2}, {c1}:{c2}] → {data.shape[0]}x{data.shape[1]}")
else:
    data = data_full
    print(f"  使用全图")
print(f"  加载耗时: {t_load:.2f}s")

# 2. 流水线
print("\n[2/3] 运行断层追踪流水线...")
cfg = Config()
t0 = time.perf_counter()
result = run_pipeline(data, cfg)
t_pipe = time.perf_counter() - t0

print(f"  流水线耗时: {t_pipe:.2f}s")
print(f"  二值图前景占比: {result['binary'].mean()*100:.1f}%")
print(f"  骨架交叉点: {len(result['junctions'])} 个")
print(f"  提取多边形: {len(result['filtered'])} 条")

if result['areas']:
    areas_sorted = sorted(result['areas'], reverse=True)
    print(f"  面积范围: {min(areas_sorted):.1f} ~ {max(areas_sorted):.1f} px^2")
    print(f"  前10大面积: {[round(a,1) for a in areas_sorted[:10]]}")

# 3. 导出
print("\n[3/3] 导出 GeoJSON ...")
export_geojson(result['filtered'], "data/test_output.geojson", result['areas'])
kb = os.path.getsize("data/test_output.geojson")
print(f"  已导出: data/test_output.geojson ({kb/1024:.1f} KB)")

print("\n" + "=" * 50)
print(f"总耗时: {t_load + t_pipe:.2f}s")
print("=" * 50)
