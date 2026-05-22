"""断层区域分割模块

从断层属性数据中提取断层区域掩膜：
1. 可选 Gabor 方向性滤波增强
2. 可选中值滤波 / 高斯滤波去噪
3. 四种阈值分割模式：Otsu / 固定 / 自适应 / 双阈值滞后
4. 形态学处理（disk/ellipse/cross 核，先开后闭或先闭后开）
"""

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.ndimage import binary_closing, binary_opening
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import disk, diamond
from .utils import normalize


def _make_kernel(radius: int, shape: str = 'disk'):
    """创建形态学结构元素。

    参数：
        radius: 核半径
        shape: 'disk'圆形 / 'ellipse'椭圆 / 'cross'十字

    返回：
        二维 bool 数组
    """
    if radius <= 0:
        return None

    if shape == 'cross':
        s = 2 * radius + 1
        kern = np.zeros((s, s), dtype=bool)
        kern[radius, :] = True
        kern[:, radius] = True
        return kern
    elif shape == 'ellipse':
        s = 2 * radius + 1
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        rx = max(radius, 1)
        ry = max(radius // 2, 1)
        return (x*x)/(rx*rx) + (y*y)/(ry*ry) <= 1.0
    else:
        return disk(radius)


def _hysteresis_threshold(data: np.ndarray, low: float, high: float) -> np.ndarray:
    """双阈值滞后分割。

    原理：
    1. 高于高阈值的像素 → 强种子点（确定为断层）
    2. 高于低阈值且与种子点连通的像素 → 也保留为断层
    3. 低于低阈值的像素 → 丢弃

    这对断层响应弱但不连续的区域特别有效。
    """
    from scipy.ndimage import label, binary_dilation

    strong = data >= high
    weak = (data >= low) & (~strong)

    if strong.sum() == 0:
        return np.zeros_like(data, dtype=np.uint8)

    # 将 weak 中与 strong 连通的像素标记为断层
    labeled, n = label(weak)
    # 膨胀强种子使接触更多弱像素
    dilated_strong = binary_dilation(strong, structure=np.ones((3, 3)))
    connected_labels = np.unique(labeled[dilated_strong & weak])
    result = np.isin(labeled, connected_labels[connected_labels > 0])

    return ((strong | result) > 0).astype(np.uint8)


def segment_fault_regions(data: np.ndarray,
                           sigma: float = 1.5,
                           otsu_scale: float = 1.0,
                           closing_radius: int = 5,
                           opening_radius: int = 2,
                           use_adaptive_threshold: bool = False,
                           adaptive_block_size: int = 35,
                           adaptive_c: float = 0.0,
                           use_median_filter: bool = False,
                           median_filter_size: int = 3,
                           use_gabor: bool = False,
                           gabor_frequency: float = 0.15,
                           gabor_angles: int = 4,
                           threshold_mode: str = 'otsu',
                           fixed_threshold: float = 0.6,
                           hysteresis_low: float = 0.3,
                           hysteresis_high: float = 0.6,
                           morph_order: str = 'open_first',
                           morph_kernel_shape: str = 'disk') -> np.ndarray:
    """从属性数据中分割出断层区域。

    参数：
        data: 属性数据 (2D)
        sigma: 高斯滤波σ
        otsu_scale: Otsu阈值缩放因子
        closing_radius: 形态学闭运算半径
        opening_radius: 形态学开运算半径
        use_adaptive_threshold: (兼容旧接口)
        adaptive_block_size: 自适应阈值局部窗口大小
        adaptive_c: 自适应阈值偏移常数
        use_median_filter: 是否使用中值滤波
        median_filter_size: 中值滤波核大小
        use_gabor: 是否使用 Gabor 方向性滤波
        gabor_frequency: Gabor 频率
        gabor_angles: Gabor 方向数
        threshold_mode: 阈值模式 'otsu'/'fixed'/'adaptive'/'hysteresis'
        fixed_threshold: 固定阈值 (threshold_mode='fixed')
        hysteresis_low: 滞后分割低阈值
        hysteresis_high: 滞后分割高阈值
        morph_order: 形态学操作顺序 'open_first'先开后闭 / 'close_first'先闭后开
        morph_kernel_shape: 结构元素形状 'disk'/'ellipse'/'cross'

    返回：
        二值掩膜 (0/1)，1 表示断层区域
    """
    data = normalize(data)

    # --- 可选 Gabor 方向性滤波 ---
    if use_gabor:
        from .preprocess import apply_gabor_filter
        data = apply_gabor_filter(data, frequency=gabor_frequency,
                                   n_angles=gabor_angles)

    # --- 可选中值滤波 ---
    if use_median_filter:
        size = max(3, median_filter_size)
        if size % 2 == 0:
            size += 1
        data = median_filter(data, size=size)

    # --- 高斯滤波去噪 ---
    smoothed = gaussian_filter(data, sigma=sigma)

    # --- 二值化 ---
    # 兼容旧接口
    mode = threshold_mode
    if use_adaptive_threshold and mode == 'otsu':
        mode = 'adaptive'

    if mode == 'fixed':
        binary = smoothed >= fixed_threshold
    elif mode == 'hysteresis':
        binary = _hysteresis_threshold(smoothed, hysteresis_low, hysteresis_high)
    elif mode == 'adaptive':
        block = max(3, adaptive_block_size)
        if block % 2 == 0:
            block += 1
        img_uint8 = (smoothed * 255).astype(np.uint8)
        local_thresh = threshold_local(img_uint8, block, method='gaussian',
                                        offset=adaptive_c * 255)
        binary = smoothed >= (local_thresh / 255.0)
    else:  # 'otsu'
        thresh = threshold_otsu(smoothed) * otsu_scale
        binary = smoothed >= thresh

    # --- 形态学处理 ---
    se_open = _make_kernel(opening_radius, morph_kernel_shape)
    se_close = _make_kernel(closing_radius, morph_kernel_shape)

    if morph_order == 'close_first':
        # 先闭后开：先填充孔洞连接断缝，再去除噪点
        if closing_radius > 0 and se_close is not None:
            binary = binary_closing(binary, structure=se_close, iterations=1)
        if opening_radius > 0 and se_open is not None:
            binary = binary_opening(binary, structure=se_open, iterations=1)
    else:
        # 先开后闭（默认）：先去噪点，再填充孔洞
        if opening_radius > 0 and se_open is not None:
            binary = binary_opening(binary, structure=se_open, iterations=1)
        if closing_radius > 0 and se_close is not None:
            binary = binary_closing(binary, structure=se_close, iterations=1)

    return binary.astype(np.uint8)
