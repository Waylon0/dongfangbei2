"""多尺度融合模块

在不同高斯σ下分别运行流水线，合并结果并用IoU去重。
细尺度捕捉细节，粗尺度连接断缝，融合后更鲁棒。
"""

import numpy as np
from typing import List, Tuple
from shapely.geometry import Polygon
from shapely.validation import make_valid


def compute_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """计算两个多边形的IoU（交并比）"""
    try:
        # (N, 2) [row, col] -> shapely (x, y) = (col, row)
        coords_a = [(float(p[1]), float(p[0])) for p in poly_a]
        coords_b = [(float(p[1]), float(p[0])) for p in poly_b]
        a = Polygon(coords_a)
        b = Polygon(coords_b)
        if not a.is_valid:
            a = make_valid(a)
        if not b.is_valid:
            b = make_valid(b)
        if a.is_empty or b.is_empty:
            return 0.0
        inter = a.intersection(b).area
        union = a.union(b).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def deduplicate_polygons(polygons: List[np.ndarray],
                          areas: List[float],
                          iou_threshold: float = 0.5) -> Tuple[List[np.ndarray], List[float]]:
    """按IoU去重：保留面积大的，移除与之高度重叠的。

    策略：按面积从大到小排序，依次检查是否与已保留的多边形IoU > threshold。
    """
    if not polygons:
        return [], []

    # 按面积降序排列
    indexed = sorted(enumerate(polygons), key=lambda x: areas[x[0]], reverse=True)

    kept = []
    kept_areas = []

    for idx, poly in indexed:
        is_dup = False
        for kept_poly in kept:
            iou = compute_iou(poly, kept_poly)
            if iou > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(poly)
            kept_areas.append(areas[idx])

    return kept, kept_areas


def merge_multiscale_results(all_polygons: List[List[np.ndarray]],
                              all_areas: List[List[float]],
                              iou_threshold: float = 0.5) -> Tuple[List[np.ndarray], List[float]]:
    """合并多个尺度的结果。

    将所有尺度的多边形汇总，按面积从大到小去重。
    尺度顺序：先细后粗（粗尺度的结果更可能被细尺度覆盖）。
    """
    merged_polygons = []
    merged_areas = []

    for polygons, areas in zip(all_polygons, all_areas):
        merged_polygons.extend(polygons)
        merged_areas.extend(areas)

    return deduplicate_polygons(merged_polygons, merged_areas, iou_threshold)
