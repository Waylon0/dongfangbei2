"""多边形矢量化与导出模块

对提取的多边形轮廓进行简化、平滑，并按面积阈值过滤。
导出为 GeoJSON (Polygon) 或简单文本格式。
"""

import json
import numpy as np
from typing import List, Tuple


def douglas_peucker(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Douglas-Peucker 折线简化算法（支持闭合多边形）"""
    if len(points) < 4:
        return points

    # 对于闭合多边形，最后一点 = 第一点
    is_closed = np.allclose(points[0], points[-1], atol=1e-6)
    if is_closed:
        work = points[:-1]
    else:
        work = points

    simplified = _dp_recursive(work, epsilon)

    if is_closed:
        # 重新闭合
        simplified = np.vstack([simplified, simplified[0:1]])

    return simplified


def _dp_recursive(points: np.ndarray, epsilon: float) -> np.ndarray:
    """DP递归简化"""
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
    """Chaikin 角点切割平滑（对闭合多边形友好）"""
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
    """用 Shoelace 公式计算多边形面积"""
    if len(points) < 3:
        return 0.0
    x = points[:, 1]  # col
    y = points[:, 0]  # row
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def simplify_polygon(contour: List[Tuple[float, float]],
                      dp_epsilon: float = 3.0,
                      smooth_iterations: int = 2) -> np.ndarray:
    """对单个多边形轮廓做简化和平滑。

    返回 (N, 2) numpy数组，[row, col]。
    """
    pts = np.array(contour, dtype=np.float64)

    # DP 简化
    if dp_epsilon > 0:
        pts = douglas_peucker(pts, dp_epsilon)

    # Chaikin 平滑
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
    """导出为 GeoJSON (Polygon 类型)

    参数：
        polygons: 多边形列表，每个为 (N, 2) 数组 [row, col]
        output_path: 输出文件路径
        areas: 对应的面积列表（可选）
    """
    features = []
    for i, poly in enumerate(polygons):
        if len(poly) < 3:
            continue
        # GeoJSON: [lon, lat] = [col, row]
        coords = [[float(pt[1]), float(pt[0])] for pt in poly]
        area = float(polygon_area(poly)) if areas is None else float(areas[i])
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            },
            "properties": {
                "id": i,
                "area_pixels": area,
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)


def export_polygons_txt(polygons: List[np.ndarray],
                          output_path: str):
    """导出为简单文本格式，兼容 GeoEast 底图模块。

    格式：
        > polygon_0  area=123.4
        row1 col1
        row2 col2
        ...
        row1 col1  (闭合)
        > polygon_1  area=56.7
        ...
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, poly in enumerate(polygons):
            if len(poly) < 3:
                continue
            area = polygon_area(poly)
            f.write(f"> polygon_{i}  area={area:.2f}\n")
            for pt in poly:
                f.write(f"{pt[0]:.2f} {pt[1]:.2f}\n")
            # 确保闭合
            if not np.allclose(poly[0], poly[-1], atol=1e-6):
                f.write(f"{poly[0, 0]:.2f} {poly[0, 1]:.2f}\n")
