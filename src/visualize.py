"""可视化模块

叠加显示原始属性数据、分割结果和多边形追踪结果。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List
from .vectorize import polygon_area



def plot_result(attr_data: np.ndarray,
                polygons: List[np.ndarray],
                binary_mask: np.ndarray = None,
                output_path: str = None,
                title: str = "Fault Polygon Extraction Result"):
    """可视化多边形提取结果。

    参数：
        attr_data: 原始属性数据 (2D)
        polygons: 多边形列表，每个为 (N, 2) numpy数组 [row, col]
        binary_mask: 断层区域掩膜（可选）
        output_path: 保存路径（可选）
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # 属性底图
    ax.imshow(attr_data, cmap='seismic', aspect='auto', origin='upper', alpha=0.6)

    # 断层区域掩膜轮廓
    if binary_mask is not None:
        from skimage.measure import find_contours
        contours = find_contours(binary_mask.astype(float), level=0.5)
        for c in contours:
            ax.plot(c[:, 1], c[:, 0], 'gray', linewidth=0.5, alpha=0.4)

    # 多边形 — 统一红色
    for i, poly in enumerate(polygons):
        ax.plot(poly[:, 1], poly[:, 0], color='#e6194b', linewidth=2.0, label=f'Fault {i}')
        ax.fill(poly[:, 1], poly[:, 0], color='#e6194b', alpha=0.1)

    ax.set_title(title)
    ax.set_xlabel('Col')
    ax.set_ylabel('Row')
    if len(polygons) <= 10:
        ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_intermediate(attr_data: np.ndarray,
                       binary_mask: np.ndarray,
                       output_path: str = None):
    """可视化中间结果：原始属性 + 断层区域掩膜"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(attr_data, cmap='seismic', aspect='auto', origin='upper')
    axes[0].set_title('Original Attribute Data')

    axes[1].imshow(binary_mask, cmap='gray', aspect='auto', origin='upper')
    axes[1].set_title('Fault Region Mask (after segmentation)')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_polygon_stats(polygons: List[np.ndarray],
                        output_path: str = None):
    """绘制多边形面积分布统计"""
    areas = [polygon_area(p) for p in polygons]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(range(len(areas)), sorted(areas, reverse=True))
    axes[0].set_xlabel('Polygon Rank')
    axes[0].set_ylabel('Area (pixels)')
    axes[0].set_title('Polygon Area Distribution')
    axes[0].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='min_area=50')
    axes[0].legend()

    axes[1].hist(areas, bins=20, edgecolor='black')
    axes[1].set_xlabel('Area (pixels)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Area Histogram')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
