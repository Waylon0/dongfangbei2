"""多边形矢量化与导出模块

对提取的多边形轮廓进行简化、平滑，并按面积阈值过滤。
导出为 GeoJSON / CSV / Shapefile / 简单文本格式。

DP 简化支持两种模式：
- absolute: epsilon = 绝对像素值
- relative: epsilon = 周长 × dp_ratio（推荐 0.001~0.01）
"""

import json
import csv
import os
import numpy as np
from typing import List, Tuple


def douglas_peucker(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Douglas-Peucker 折线简化算法（支持闭合多边形）"""
    if len(points) < 4:
        return points

    is_closed = np.allclose(points[0], points[-1], atol=1e-6)
    if is_closed:
        work = points[:-1]
    else:
        work = points

    simplified = _dp_recursive(work, epsilon)

    if is_closed:
        simplified = np.vstack([simplified, simplified[0:1]])

    return simplified


def _dp_recursive(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) < 3:
        return points

    start, end = points[0], points[-1]
    vec = end - start
    vec_norm = np.linalg.norm(vec)
    if vec_norm < 1e-10:
        return np.array([start, end])

    t = np.sum((points - start) * vec, axis=1) / (vec_norm ** 2)
    t = np.clip(t, 0, 1)
    proj = start + t[:, np.newaxis] * vec
    dists = np.linalg.norm(points - proj, axis=1)

    max_idx = np.argmax(dists)
    max_dist = dists[max_idx]

    if max_dist > epsilon:
        left = _dp_recursive(points[:max_idx + 1], epsilon)
        right = _dp_recursive(points[max_idx:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.array([start, end])


def chaikin_smooth(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Chaikin 角点切割平滑"""
    if len(points) < 3:
        return points

    for _ in range(iterations):
        new_pts = [points[0]]
        for i in range(len(points) - 1):
            p0, p1 = points[i], points[i + 1]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_pts.append(q)
            new_pts.append(r)
        new_pts.append(points[-1])
        points = np.array(new_pts)

    return points


def polygon_area(points: np.ndarray) -> float:
    """Shoelace 公式计算多边形面积"""
    if len(points) < 3:
        return 0.0
    x = points[:, 1]
    y = points[:, 0]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_perimeter(points: np.ndarray) -> float:
    """计算多边形周长"""
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1)))


def simplify_polygon(contour: List[Tuple[float, float]],
                      dp_epsilon: float = 3.0,
                      smooth_iterations: int = 2,
                      dp_mode: str = 'absolute',
                      dp_ratio: float = 0.005) -> np.ndarray:
    """对单个多边形轮廓做简化和平滑。

    参数：
        contour: 轮廓点列表 [(row, col), ...]
        dp_epsilon: DP容差（dp_mode='absolute'时绝对像素值）
        smooth_iterations: Chaikin平滑迭代次数
        dp_mode: 'absolute' 绝对像素 / 'relative' 周长比例
        dp_ratio: dp_mode='relative'时 epsilon = 周长 × dp_ratio
                  建议 0.001~0.01（典型值 0.005）

    返回：
        (N, 2) numpy数组 [row, col]
    """
    pts = np.array(contour, dtype=np.float64)

    if dp_mode == 'relative':
        perimeter = float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))
        epsilon = max(0.1, perimeter * dp_ratio)
    else:
        epsilon = dp_epsilon

    if epsilon > 0:
        pts = douglas_peucker(pts, epsilon)

    if smooth_iterations > 0:
        pts = chaikin_smooth(pts, smooth_iterations)

    return pts


def filter_by_area(polygons: List[np.ndarray],
                    min_area: float = 50) -> List[np.ndarray]:
    """按面积阈值过滤多边形"""
    result = []
    for poly in polygons:
        area = polygon_area(poly)
        if area >= min_area:
            result.append(poly)
    return result


def export_geojson(polygons: List[np.ndarray],
                    output_path: str,
                    areas: List[float] = None):
    """导出为 GeoJSON (Polygon 类型)"""
    features = []
    for i, poly in enumerate(polygons):
        if len(poly) < 3:
            continue
        coords = [[float(pt[1]), float(pt[0])] for pt in poly]
        area = float(polygon_area(poly)) if areas is None else float(areas[i])
        perimeter = polygon_perimeter(poly)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            },
            "properties": {
                "id": i,
                "area_pixels": round(area, 2),
                "perimeter_pixels": round(perimeter, 2),
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)


def export_csv(polygons: List[np.ndarray],
               output_path: str,
               areas: List[float] = None):
    """导出统计结果到 CSV 文件"""
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['编号', '面积(像素²)', '周长(像素)', '紧致度', '顶点数'])
        for i, poly in enumerate(polygons):
            if len(poly) < 3:
                continue
            area = float(polygon_area(poly)) if areas is None else float(areas[i])
            perimeter = polygon_perimeter(poly)
            comp = (perimeter * perimeter) / (4 * np.pi * area) if area > 0 else 0
            writer.writerow([i, round(area, 2), round(perimeter, 2),
                           round(comp, 3), len(poly)])
        writer.writerow([])
        writer.writerow(['总计', len(polygons), '', '', ''])
        if polygons:
            total_area = sum(
                float(polygon_area(p)) if areas is None else float(areas[i])
                for i, p in enumerate(polygons) if len(p) >= 3
            )
            writer.writerow(['总面积', round(total_area, 2), '', '', ''])


def export_shapefile(polygons: List[np.ndarray],
                     output_path: str,
                     areas: List[float] = None):
    """导出为 ESRI Shapefile 格式。

    需要 pyshp 库：pip install pyshp
    """
    try:
        import shapefile
    except ImportError:
        raise ImportError("导出 Shapefile 需要 pyshp 库。请运行: pip install pyshp")

    with shapefile.Writer(output_path, shapeType=shapefile.POLYGON) as shp:
        shp.field('id', 'N', decimal=0)
        shp.field('area', 'F', decimal=2)
        shp.field('perimeter', 'F', decimal=2)

        for i, poly in enumerate(polygons):
            if len(poly) < 3:
                continue
            # Shapefile: [lon, lat] = [col, row]
            coords = [(float(p[1]), float(p[0])) for p in poly]
            area = float(polygon_area(poly)) if areas is None else float(areas[i])
            perimeter = polygon_perimeter(poly)
            shp.poly([coords])
            shp.record(i, round(area, 2), round(perimeter, 2))

    # 生成 .prj 投影文件 (WGS84 作为默认)
    prj_path = output_path.replace('.shp', '.prj')
    if not prj_path.endswith('.prj'):
        prj_path = output_path + '.prj'
    with open(prj_path, 'w') as f:
        f.write('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
                'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')


def export_polygons_txt(polygons: List[np.ndarray],
                          output_path: str):
    """导出为简单文本格式"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, poly in enumerate(polygons):
            if len(poly) < 3:
                continue
            area = polygon_area(poly)
            f.write(f"> polygon_{i}  area={area:.2f}\n")
            for pt in poly:
                f.write(f"{pt[0]:.2f} {pt[1]:.2f}\n")
            if not np.allclose(poly[0], poly[-1], atol=1e-6):
                f.write(f"{poly[0, 0]:.2f} {poly[0, 1]:.2f}\n")
