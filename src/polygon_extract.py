"""多边形轮廓提取模块

从断层区域掩膜中提取闭合的多边形轮廓。

支持两种模式：
- 'region' (默认): 从二值区域提取外轮廓 → 粗多边形
- 'skeleton' (推荐): 骨架化 → 窄缓冲 → 提取轮廓 → 精细线条多边形，
  沿断层中心线勾勒，输出窄条带而非粗块
"""

import numpy as np
from scipy.ndimage import label, binary_dilation
from skimage.measure import find_contours
from skimage.morphology import skeletonize, disk, remove_small_objects
from typing import List, Tuple, Dict


def _extract_skeleton_polygons(binary_mask: np.ndarray,
                                  min_component_area: int = 30,
                                  buffer_radius: int = 2,
                                  smooth_sigma: float = 1.0,
                                  max_aspect_ratio: float = 20.0,
                                  max_compactness: float = 5.0,
                                  precomputed_skel: np.ndarray = None) -> tuple:
    """骨架化 → 在交叉点拆分为独立线段 → 窄缓冲 → 精细轮廓。

    关键：骨架分叉处断开，保证每条断层独立输出，不会因为二值区域
    相互接触就把多条断层合并为一个多边形。
    """
    # 骨架像素阈值
    skel_min_pixels = max(5, min_component_area // 5)
    se8 = np.ones((3, 3), dtype=bool)

    # 1. 骨架化
    if precomputed_skel is not None:
        skel = precomputed_skel
    else:
        skel = skeletonize(binary_mask.astype(bool))
    if skel.sum() == 0:
        return [], []

    # 2. 找交叉点（度数 >= 3）并在交叉点断开
    junctions = _find_junction_points(skel)

    skel_split = skel.copy().astype(np.uint8)
    if junctions:
        for r, c in junctions:
            skel_split[r, c] = 0
            # 断开交叉点周围
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < skel_split.shape[0] and 0 <= cc < skel_split.shape[1]:
                        if (rr, cc) in junctions:
                            skel_split[rr, cc] = 0

    # 3. 标注断开后的独立片段
    labeled_skel, n_seg = label(skel_split, structure=se8)

    contours = []
    components = []

    for seg_id in range(1, n_seg + 1):
        seg = (labeled_skel == seg_id)
        seg_pixels = seg.sum()

        if seg_pixels < skel_min_pixels:
            continue

        # 缓冲这条骨架片段
        if buffer_radius > 0:
            se = disk(buffer_radius)
            buffered = binary_dilation(seg, structure=se, iterations=1)
        else:
            buffered = seg

        area = float(buffered.sum())
        if area < min_component_area:
            continue

        coords = np.argwhere(buffered)
        centroid = None
        if len(coords) > 0:
            centroid = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))

        c = _extract_contour(buffered, smooth_sigma=0)
        if c is not None and len(c) >= 4:
            contours.append(c)
            components.append({
                'id': seg_id,
                'area': area,
                'centroid': centroid,
                'aspect_ratio': 0.0,
                'compactness': 0.0,
            })

    return contours, components


def extract_fault_polygons(binary_mask: np.ndarray,
                            min_component_area: int = 100,
                            separate_intersections: bool = True,
                            smooth_sigma: float = 2.0,
                            max_aspect_ratio: float = 20.0,
                            max_compactness: float = 5.0,
                            polygon_mode: str = 'region',
                            skeleton_buffer: int = 2,
                            precomputed_skel: np.ndarray = None) -> tuple:
    """从断层区域掩膜提取闭合多边形轮廓。

    参数：
        binary_mask: 断层区域二值掩膜 (2D, 0/1)
        min_component_area: 最小连通域面积
        separate_intersections: 是否分离交叉断层（仅 region 模式）
        smooth_sigma: 轮廓高斯平滑σ
        max_aspect_ratio: 最大长宽比过滤
        max_compactness: 最大紧致度过滤
        polygon_mode: 'region' 粗区域轮廓 / 'skeleton' 骨架细线多边形
        skeleton_buffer: skeleton 模式下骨架缓冲半径（像素，1~3）

    返回：
        (多边形列表, 组件属性列表)
    """
    if polygon_mode == 'skeleton':
        # 骨架模式的区域很薄（缓冲后约5px宽），平滑σ必须较小
        skel_smooth = min(smooth_sigma, 1.0)
        return _extract_skeleton_polygons(
            binary_mask, min_component_area=min_component_area,
            buffer_radius=skeleton_buffer, smooth_sigma=skel_smooth,
            max_aspect_ratio=max_aspect_ratio, max_compactness=max_compactness,
            precomputed_skel=precomputed_skel)

    # --- region 模式（原有逻辑） ---
    labeled, n_features = label(binary_mask)

    contours = []
    components = []

    for region_id in range(1, n_features + 1):
        region = (labeled == region_id)
        area = float(region.sum())

        if area < min_component_area:
            continue

        from .utils import aspect_ratio as ar_func, compactness as comp_func

        ar = ar_func(region)
        comp = comp_func(region)

        # 形状过滤
        if ar > max_aspect_ratio or comp > max_compactness:
            continue

        centroid = None
        coords = np.argwhere(region)
        if len(coords) > 0:
            centroid = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))

        comp_info = {
            'id': region_id,
            'area': area,
            'centroid': centroid,
            'aspect_ratio': ar,
            'compactness': comp,
        }

        if separate_intersections:
            sub_regions = _separate_intersecting_faults(region)
            added = False
            for sub in sub_regions:
                sub_area = float(sub.sum())
                if sub_area >= min_component_area:
                    c = _extract_contour(sub, smooth_sigma)
                    if c is not None and len(c) >= 4:
                        contours.append(c)
                        comp_info['area'] = sub_area
                        components.append(dict(comp_info))
                        added = True
            if not added:
                c = _extract_contour(region, smooth_sigma)
                if c is not None and len(c) >= 4:
                    contours.append(c)
                    components.append(comp_info)
        else:
            c = _extract_contour(region, smooth_sigma)
            if c is not None and len(c) >= 4:
                contours.append(c)
                components.append(comp_info)

    return contours, components


def _separate_intersecting_faults(region: np.ndarray) -> List[np.ndarray]:
    """在骨架交叉点处分离连在一起的断层区域。"""
    skel = skeletonize(region)
    junctions = _find_junction_points(skel)

    if len(junctions) == 0:
        return [region]

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

    labeled_skel, n = label(skel_split)
    if n < 2:
        return [region]

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

    valid = [(i, d) for i, d in enumerate(segment_dirs) if d is not None]
    if len(valid) < 2:
        sub_regions = []
        for skel_id in range(1, n + 1):
            skel_part = (labeled_skel == skel_id)
            if skel_part.sum() < 5:
                continue
            dilated = _dilate_to_original(skel_part, region)
            if dilated.sum() > 5:
                sub_regions.append(dilated)
        return sub_regions if sub_regions else [region]

    groups = _cluster_by_direction(segment_masks, segment_dirs)

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
    coords = np.argwhere(skel_part)
    if len(coords) < 5:
        return None
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered[:, 1], centered[:, 0])
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, -1]
    direction = np.array([principal[1], principal[0]])
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-10 else None


def _cluster_by_direction(segment_masks: list, segment_dirs: list,
                           angle_threshold: float = 45.0) -> list:
    """按方向对片段做贪心聚类。"""
    n = len(segment_masks)
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

    # 预计算 cos 阈值，避免循环内 arccos → degrees 转换
    cos_threshold = np.cos(np.radians(angle_threshold))

    # 预膨胀所有片段，避免 O(n²) 次重复 binary_dilation 调用
    d3 = disk(3)
    dilated_masks = [binary_dilation(m, d3) if segment_dirs[i] is not None else None
                     for i, m in enumerate(segment_masks)]

    for i in range(n):
        if segment_dirs[i] is None:
            continue
        for j in range(i + 1, n):
            if segment_dirs[j] is None:
                continue
            if not (dilated_masks[i] & dilated_masks[j]).any():
                continue
            dot = abs(np.dot(segment_dirs[i], segment_dirs[j]))
            if dot > cos_threshold:
                union(i, j)

    groups_dict = {}
    for i in range(n):
        root = find(i)
        groups_dict.setdefault(root, []).append(i)

    return list(groups_dict.values())


def _dilate_to_original(skel_part: np.ndarray,
                         original_region: np.ndarray) -> np.ndarray:
    """将骨架片段膨胀回原始区域的大小"""
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

    优先使用 cv2.findContours（C++ 实现，比 skimage 快 5-10x），
    skimage 作为 fallback。
    """
    # 尝试 cv2（更快）
    try:
        import cv2
        region_u8 = region.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(region_u8, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        # 取最长轮廓
        longest = max(cnts, key=cv2.contourArea)
        # cv2 返回 (N,1,2) [col,row]，转为 [(row, col)]
        pts = [(float(p[0][1]), float(p[0][0])) for p in longest]
        if len(pts) > 1:
            d = np.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1])
            if d > 2.0:
                pts.append(pts[0])
        return pts
    except ImportError:
        pass

    # fallback: skimage
    from scipy.ndimage import gaussian_filter
    if smooth_sigma > 0:
        region_f = region.astype(np.float64)
        region_f = gaussian_filter(region_f, sigma=smooth_sigma)
    else:
        region_f = region.astype(np.float64)

    contours = find_contours(region_f, level=0.5)

    if not contours:
        return None

    longest = max(contours, key=len)

    pts = [(float(p[0]), float(p[1])) for p in longest]
    if len(pts) > 1:
        d = np.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1])
        if d > 2.0:
            pts.append(pts[0])

    return pts
