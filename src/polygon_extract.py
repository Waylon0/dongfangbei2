"""多边形轮廓提取模块

从断层区域掩膜中提取闭合的多边形轮廓：
1. 连通域分析 → 分离不同断层区域
2. 交叉断层分离 → 骨架化+交叉点检测分拆
3. 闭合轮廓追踪 → 每个区域的边界
"""

import numpy as np
from scipy.ndimage import label
from skimage.measure import find_contours
from skimage.morphology import skeletonize, disk
from skimage.segmentation import clear_border
from typing import List, Tuple


def extract_fault_polygons(binary_mask: np.ndarray,
                            min_component_area: int = 100,
                            separate_intersections: bool = True,
                            smooth_sigma: float = 2.0) -> List[List[Tuple[float, float]]]:
    """从断层区域掩膜提取闭合多边形轮廓。

    参数：
        binary_mask: 断层区域二值掩膜 (2D, 0/1)
        min_component_area: 最小连通域面积，小于此值的区域被丢弃
        separate_intersections: 是否在骨架交叉处分离不同断层
        smooth_sigma: 轮廓高斯平滑σ

    返回：
        多边形列表，每个多边形为 [(row, col), ...] 闭合轮廓点
    """
    # 步骤1：连通域分析
    labeled, n_features = label(binary_mask)

    contours = []
    for region_id in range(1, n_features + 1):
        region = (labeled == region_id)

        # 面积过滤
        if region.sum() < min_component_area:
            continue

        if separate_intersections:
            sub_regions = _separate_intersecting_faults(region)
            for sub in sub_regions:
                if sub.sum() >= min_component_area:
                    c = _extract_contour(sub, smooth_sigma)
                    if c is not None and len(c) >= 4:
                        contours.append(c)
        else:
            c = _extract_contour(region, smooth_sigma)
            if c is not None and len(c) >= 4:
                contours.append(c)

    return contours


def _separate_intersecting_faults(region: np.ndarray) -> List[np.ndarray]:
    """在骨架交叉点处分离连在一起的断层区域。

    改进：用方向聚类代替粗暴删除交叉点像素。
    1. 骨架化 → 找交叉点
    2. 在交叉点断开骨架，得到独立片段
    3. 对每个片段估算走向
    4. 按走向聚类：方向一致的片段归为同一断层
    5. 每组片段膨胀回原区域
    """
    skel = skeletonize(region)
    junctions = _find_junction_points(skel)

    if len(junctions) == 0:
        return [region]

    # 在交叉点处断开骨架
    skel_split = skel.copy().astype(np.uint8)
    h, w = skel.shape
    for r, c in junctions:
        skel_split[r, c] = 0
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    if (rr, cc) in set(map(tuple, junctions)):
                        skel_split[rr, cc] = 0

    # 连通域标记 → 独立片段
    labeled_skel, n = label(skel_split)
    if n < 2:
        return [region]

    # 估算每段走向
    segment_dirs = []
    segment_masks = []
    for skel_id in range(1, n + 1):
        skel_part = (labeled_skel == skel_id)
        if skel_part.sum() < 5:
            segment_dirs.append(None)
            segment_masks.append(skel_part)
            continue
        direction = _estimate_skeleton_direction(skel_part)
        segment_dirs.append(direction)
        segment_masks.append(skel_part)

    # 按方向聚类
    valid = [(i, d) for i, d in enumerate(segment_dirs) if d is not None]
    if len(valid) < 2:
        # 不足2个有效方向，回退到简单膨胀
        sub_regions = []
        for skel_id in range(1, n + 1):
            skel_part = (labeled_skel == skel_id)
            if skel_part.sum() < 5:
                continue
            dilated = _dilate_to_original(skel_part, region)
            if dilated.sum() > 5:
                sub_regions.append(dilated)
        return sub_regions if sub_regions else [region]

    # 贪心聚类：方向夹角 < 45° 的片段归为一组
    groups = _cluster_by_direction(segment_masks, segment_dirs)

    # 每组片段膨胀回原区域
    sub_regions = []
    for group in groups:
        combined_skel = np.zeros_like(skel, dtype=bool)
        for idx in group:
            combined_skel |= segment_masks[idx]
        if combined_skel.sum() < 5:
            continue
        dilated = _dilate_to_original(combined_skel, region)
        if dilated.sum() > 5:
            sub_regions.append(dilated)

    return sub_regions if sub_regions else [region]


def _estimate_skeleton_direction(skel_part: np.ndarray) -> np.ndarray:
    """估算骨架片段的主方向（PCA），返回单位方向向量 (dr, dc)"""
    coords = np.argwhere(skel_part)  # (N, 2) [row, col]
    if len(coords) < 5:
        return None
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered[:, 1], centered[:, 0])  # (col, row) -> (x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, -1]
    direction = np.array([principal[1], principal[0]])  # 转回 (row, col)
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-10 else None


def _cluster_by_direction(segment_masks: list, segment_dirs: list,
                           angle_threshold: float = 45.0) -> list:
    """按方向对片段做贪心聚类。返回 [[idx, ...], ...]"""
    n = len(segment_masks)
    # 每个片段初始独立成组
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        if segment_dirs[i] is None:
            continue
        for j in range(i + 1, n):
            if segment_dirs[j] is None:
                continue
            # 检查片段是否在空间上相邻（膨胀后重叠）
            from scipy.ndimage import binary_dilation
            dilated_i = binary_dilation(segment_masks[i], disk(3))
            dilated_j = binary_dilation(segment_masks[j], disk(3))
            if not (dilated_i & dilated_j).any():
                continue
            # 方向一致性
            dot = abs(np.dot(segment_dirs[i], segment_dirs[j]))
            angle = np.degrees(np.arccos(np.clip(dot, 0, 1)))
            if angle < angle_threshold:
                union(i, j)

    # 收集分组
    groups_dict = {}
    for i in range(n):
        root = find(i)
        groups_dict.setdefault(root, []).append(i)

    return list(groups_dict.values())


def _dilate_to_original(skel_part: np.ndarray,
                         original_region: np.ndarray) -> np.ndarray:
    """将骨架片段膨胀回原始区域的大小，但被原始区域边界裁剪"""
    from scipy.ndimage import distance_transform_edt, binary_dilation
    # 计算原始区域内各点到骨架的距离
    dist = distance_transform_edt(original_region)
    # 膨胀，但不超过原始区域边界
    dilated = binary_dilation(skel_part, structure=disk(3), iterations=5)
    return dilated & original_region


def _find_junction_points(skel: np.ndarray) -> List[Tuple[int, int]]:
    """找骨架图中的交叉点（度数 >= 3）"""
    coords = np.argwhere(skel > 0)
    junctions = []
    h, w = skel.shape
    for r, c in coords:
        r, c = int(r), int(c)
        cnt = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and skel[rr, cc]:
                    cnt += 1
        if cnt >= 3:
            junctions.append((r, c))
    return junctions


def _extract_contour(region: np.ndarray,
                      smooth_sigma: float = 2.0) -> List[Tuple[float, float]]:
    """从二值区域提取最外层闭合轮廓。

    使用 marching squares 算法 (skimage.measure.find_contours)，
    返回最长的轮廓（即外围边界）。
    """
    from scipy.ndimage import gaussian_filter

    # 对区域边界做轻微平滑，减少锯齿
    region_f = region.astype(np.float64)
    if smooth_sigma > 0:
        region_f = gaussian_filter(region_f, sigma=smooth_sigma)

    contours = find_contours(region_f, level=0.5)

    if not contours:
        return None

    # 取最长的轮廓（外围边界）
    longest = max(contours, key=len)

    # 确保闭合：如果首尾距离 > 1像素，手动闭合
    pts = [(float(p[0]), float(p[1])) for p in longest]
    if len(pts) > 1:
        d = np.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1])
        if d > 2.0:
            pts.append(pts[0])

    return pts
