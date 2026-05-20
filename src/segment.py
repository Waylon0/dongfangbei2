"""断层区域分割模块

从断层属性数据中提取断层区域掩膜：
1. 阈值二值化 → 断层/非断层（全局Otsu或局部自适应）
2. 形态学闭运算 → 填充小孔洞、连接断缝
3. 形态学开运算 → 去除噪点
"""

import numpy as np
from scipy.ndimage import gaussian_filter, binary_closing, binary_opening
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import disk
from .utils import normalize


def segment_fault_regions(data: np.ndarray,
                           sigma: float = 1.5,
                           otsu_scale: float = 1.0,
                           closing_radius: int = 5,
                           opening_radius: int = 2,
                           use_adaptive_threshold: bool = False,
                           adaptive_block_size: int = 35,
                           adaptive_c: float = 0.0) -> np.ndarray:
    """从属性数据中分割出断层区域。

    参数：
        data: 归一化后的属性数据 (2D)
        sigma: 高斯滤波σ
        otsu_scale: Otsu阈值缩放因子
        closing_radius: 形态学闭运算半径
        opening_radius: 形态学开运算半径
        use_adaptive_threshold: 使用局部自适应阈值
        adaptive_block_size: 自适应阈值局部窗口大小（奇数）
        adaptive_c: 自适应阈值偏移常数

    返回：
        二值掩膜 (0/1)，1 表示断层区域
    """
    data = normalize(data)

    # 高斯滤波去噪
    smoothed = gaussian_filter(data, sigma=sigma)

    # 二值化
    if use_adaptive_threshold:
        # 局部自适应阈值：对响应强度空间变化大的数据更鲁棒
        block = max(3, adaptive_block_size)
        if block % 2 == 0:
            block += 1
        img_uint8 = (smoothed * 255).astype(np.uint8)
        local_thresh = threshold_local(img_uint8, block, method='gaussian',
                                        offset=adaptive_c * 255)
        binary = smoothed >= (local_thresh / 255.0)
    else:
        # 全局 Otsu 阈值
        thresh = threshold_otsu(smoothed) * otsu_scale
        binary = smoothed >= thresh

    # 形态学闭运算 — 填充断层区域内部小孔洞，连接断缝
    if closing_radius > 0:
        se_close = disk(closing_radius)
        binary = binary_closing(binary, structure=se_close, iterations=1)

    # 形态学开运算 — 去除孤立噪点
    if opening_radius > 0:
        se_open = disk(opening_radius)
        binary = binary_opening(binary, structure=se_open, iterations=1)

    return binary.astype(np.uint8)
