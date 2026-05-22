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
