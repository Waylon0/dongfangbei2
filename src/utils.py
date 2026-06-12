"""公共工具函数"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple


def normalize(data: np.ndarray) -> np.ndarray:
    """归一化到 [0, 1]"""
    mn, mx = data.min(), data.max()
    if mx == mn:
        return np.zeros_like(data)
    return (data - mn) / (mx - mn)


def gradient_orientation(data: np.ndarray, sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """计算梯度的幅值和方向。返回 (magnitude, orientation_radians)"""
    smoothed = gaussian_filter(data, sigma=sigma)
    gy, gx = np.gradient(smoothed)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ori = np.arctan2(gy, gx)
    return mag, ori


def compute_local_direction(skel: np.ndarray, r: int, c: int, window: int = 5) -> Tuple[float, float]:
    """在骨架点 (r,c) 处估计局部方向，返回单位方向向量 (dr, dc)"""
    half = window // 2
    h, w = skel.shape
    ys, xs = [], []
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and skel[rr, cc]:
                ys.append(rr - r)
                xs.append(cc - c)
    if len(ys) < 2:
        return (0.0, 0.0)

    ys, xs = np.array(ys), np.array(xs)
    cov = np.cov(xs, ys)
    if cov.shape != (2, 2):
        return (0.0, 0.0)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, -1]
    return (float(principal[1]), float(principal[0]))


def angle_between(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """两个向量之间的最小夹角（度）"""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
    n2 = np.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 180.0
    cos = np.clip(dot / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def compactness(region: np.ndarray) -> float:
    """计算区域的紧致度 = 周长² / (4π × 面积)，圆形=1，线状>1"""
    from skimage.measure import perimeter
    area = region.sum()
    if area < 1:
        return float('inf')
    p = perimeter(region)
    return (p * p) / (4 * np.pi * area)


def aspect_ratio(region: np.ndarray) -> float:
    """计算区域的宽高比（通过 PCA 主方向）"""
    coords = np.argwhere(region)
    if len(coords) < 3:
        return 1.0
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered[:, 1], centered[:, 0])
    eigenvalues, _ = np.linalg.eigh(cov)
    if eigenvalues.min() < 1e-10:
        return float('inf')
    return float(eigenvalues.max() / eigenvalues.min())


def fault_length_from_contour(contour: list) -> float:
    """从多边形轮廓估算断层长度（取轮廓周长的一半作为中线近似）"""
    if len(contour) < 2:
        return 0.0
    pts = np.array(contour)
    perimeter = float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))
    return perimeter / 2.0


def fault_length_from_binary(binary: np.ndarray) -> float:
    """从二值断层区域估算总长度（骨架化后像素数）"""
    from skimage.morphology import skeletonize
    skel = skeletonize(binary.astype(bool))
    return float(skel.sum())


def group_fault_polygons(polygons: list,
                          max_gap_distance: float = 30.0,
                          angle_threshold: float = 30.0,
                          lateral_threshold: float = 20.0) -> list:
    """将属于同一条地质断层的多边形碎片归为一组。

    一条断层可能因为噪声、遮挡或阈值原因被断成多个不连通的多边形。
    此函数按空间邻近度 + 方向一致性 + 共线性将碎片聚类，
    同组的多边形视为同一条断层，应在可视化中使用相同颜色。

    算法：
    1. 对每个多边形计算质心、PCA 主方向和两个端点
    2. 两两比较：端点距离 < max_gap_distance 且
       方向夹角 < angle_threshold 且
       横向偏移 < lateral_threshold → 归为同一断层
    3. 并查集连通分量 → 输出分组标签

    参数：
        polygons: 多边形列表，每个为 (N,2) numpy 数组 [row, col]
        max_gap_distance: 两个碎片端点间的最大允许距离（像素）
        angle_threshold: 主方向夹角阈值（度）
        lateral_threshold: 横向偏移阈值（像素），限制碎片不能偏离太远

    返回：
        groups: 列表，每个元素为该组的多边形索引列表 [[idx, idx], ...]
    """
    n = len(polygons)
    if n <= 1:
        return [list(range(n))] if n == 1 else []

    # --- 1. 提取每个多边形的特征 ---
    centroids = []
    directions = []   # 单位主方向向量 (dr, dc)
    endpoints = []    # 两个端点 [(r1,c1), (r2,c2)]

    for poly in polygons:
        arr = np.array(poly)
        # 去掉闭合点（首尾相同）
        if len(arr) > 1 and np.allclose(arr[0], arr[-1], atol=1e-6):
            arr = arr[:-1]
        if len(arr) < 2:
            centroids.append(None)
            directions.append(None)
            endpoints.append(None)
            continue

        centroid = arr.mean(axis=0)
        centroids.append(centroid)

        # PCA 主方向
        centered = arr - centroid
        cov = np.cov(centered[:, 1], centered[:, 0])
        if cov.shape == (2, 2):
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            principal = eigenvectors[:, -1]  # (dc, dr)
            direction = np.array([principal[1], principal[0]])  # (dr, dc)
        else:
            direction = np.array([0.0, 0.0])
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
        directions.append(direction)

        # 找到沿主方向的两个最远端点
        if norm > 1e-10:
            proj = np.dot(centered, direction)
            idx1, idx2 = np.argmin(proj), np.argmax(proj)
            endpoints.append((tuple(arr[idx1]), tuple(arr[idx2])))
        else:
            endpoints.append(None)

    # --- 2. 并查集聚类 ---
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

    cos_angle = np.cos(np.radians(angle_threshold))

    for i in range(n):
        if directions[i] is None:
            continue
        for j in range(i + 1, n):
            if directions[j] is None:
                continue
            if find(i) == find(j):
                continue

            # 方向一致性
            dot = abs(np.dot(directions[i], directions[j]))
            if dot < cos_angle:
                continue

            # 端点间最小距离
            ep_i = endpoints[i]
            ep_j = endpoints[j]
            if ep_i is None or ep_j is None:
                continue
            d1 = np.linalg.norm(np.array(ep_i[0]) - np.array(ep_j[0]))
            d2 = np.linalg.norm(np.array(ep_i[0]) - np.array(ep_j[1]))
            d3 = np.linalg.norm(np.array(ep_i[1]) - np.array(ep_j[0]))
            d4 = np.linalg.norm(np.array(ep_i[1]) - np.array(ep_j[1]))
            min_dist = min(d1, d2, d3, d4)
            if min_dist > max_gap_distance:
                continue

            # 横向偏移：两个质心在垂直方向上的距离
            avg_dir = directions[i] + directions[j]
            avg_norm = np.linalg.norm(avg_dir)
            if avg_norm < 1e-10:
                continue
            avg_dir = avg_dir / avg_norm
            # 垂直方向
            perp = np.array([-avg_dir[1], avg_dir[0]])
            centroid_vec = centroids[j] - centroids[i]
            lateral = abs(np.dot(centroid_vec, perp))
            if lateral > lateral_threshold:
                continue

            union(i, j)

    # --- 3. 收集分组 ---
    groups_dict = {}
    for i in range(n):
        root = find(i)
        groups_dict.setdefault(root, []).append(i)

    # 按组内最大面积排序（大断层优先）
    from .vectorize import polygon_area
    groups = list(groups_dict.values())
    groups.sort(key=lambda g: max(polygon_area(polygons[i]) for i in g), reverse=True)

    return groups
