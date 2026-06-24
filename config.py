"""可调参数配置"""


class Config:
    # --- 预处理 ---
    gaussian_sigma: float = 1.5       # 高斯滤波σ
    use_median_filter: bool = False   # 是否先使用中值滤波
    median_filter_size: int = 3       # 中值滤波核大小（奇数）
    use_clahe: bool = False           # CLAHE对比度增强
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8

    # --- Gabor 方向性滤波 ---
    use_gabor: bool = False           # 是否使用Gabor方向性滤波
    gabor_frequency: float = 0.15     # Gabor频率
    gabor_angles: int = 4             # Gabor方向数（等分180°）

    # --- UNet 深度学习分割 ---
    use_unet: bool = False            # 使用UNet替代传统二值化
    unet_model_path: str = ''         # UNet模型文件路径

    # --- 二值化 ---
    threshold_mode: str = 'otsu'      # 'otsu' / 'fixed' / 'adaptive' / 'hysteresis'
    otsu_scale: float = 0.8           # Otsu阈值缩放（>1更保守，<1更敏感）
    fixed_threshold: float = 0.6      # 固定阈值（threshold_mode='fixed'时生效）
    use_adaptive_threshold: bool = False  # 使用局部自适应阈值（已废弃，改用threshold_mode）
    adaptive_block_size: int = 35     # 自适应阈值局部窗口大小（奇数）
    adaptive_c: float = 0.0           # 自适应阈值偏移常数

    # --- 双阈值滞后分割 ---
    hysteresis_low: float = 0.3       # 低阈值
    hysteresis_high: float = 0.6      # 高阈值

    # --- 形态学处理 ---
    morph_order: str = 'open_first'   # 操作顺序：'open_first'先开后闭 / 'close_first'先闭后开
    closing_radius: int = 2           # 闭运算半径（skeleton模式宜小，避免过度合并断层）
    opening_radius: int = 1           # 开运算半径（去除噪点）
    morph_kernel_shape: str = 'disk'  # 结构元素形状：'disk'圆形 / 'ellipse'椭圆 / 'cross'十字

    # --- 断层分离 ---
    min_component_area: int = 30      # 最小连通域面积（skeleton模式下为缓冲后面积）
    separate_intersections: bool = True  # 是否在交叉处分离不同断层

    # --- 形状过滤 ---
    max_aspect_ratio: float = 30.0    # 最大长宽比（超出视为噪声）
    max_compactness: float = 10.0     # 最大紧致度（周长²/面积，越大越不规则）

    # --- 轮廓提取 ---
    contour_smooth_sigma: float = 0.0 # 轮廓平滑σ（0=不平滑，保留原始断层边界棱角）
    polygon_mode: str = 'skeleton'    # 多边形模式：'region'粗区域 / 'skeleton'骨架细线
    skeleton_buffer: int = 2          # skeleton模式下骨架缓冲半径（1~3像素）

    # --- 多边形过滤 ---
    min_polygon_area: float = 15      # 最小多边形面积（像素），skeleton模式通常较小

    # --- 多边形简化 ---
    dp_mode: str = 'absolute'         # 'absolute'绝对像素 / 'relative'周长比例
    dp_epsilon: float = 1.0           # DP简化容差（dp_mode='absolute'时像素值，越小顶点越多）
    dp_ratio: float = 0.005           # DP简化比例（dp_mode='relative'时周长×该比例）
    smooth_iterations: int = 0        # Chaikin平滑迭代次数（0=不平滑，保留断层多边形棱角）

    # --- 多尺度 ---
    scales: list = []                 # 多尺度σ列表（单尺度用 []）
    dedup_overlap_threshold: float = 0.5  # 去重重合度（IoU）

    # --- 断层追踪 ---
    track_max_link_distance: float = 12.0   # 最大连接距离（像素）
    track_angle_weight: float = 2.0         # 方向一致性权重（越大越严格）
    track_min_segment_length: int = 10      # 最小参与连接的片段长度
    track_dilate_radius: int = 1            # 连接后膨胀半径（skeleton模式用1，防止过度合并）
    track_dilate_iterations: int = 2        # 膨胀迭代次数（skeleton模式用2，防止过度合并）

    # --- 导出 ---
    output_format: str = 'geojson'    # geojson / txt / csv / shp


# ── 参数提示文本（UI tooltip 用） ──────────────────────────────

PARAM_HINTS = {
    'gaussian_sigma': '高斯滤波标准差。值越大去噪越强但会模糊断层边缘，建议 1.0~2.0。',
    'use_median_filter': '先执行中值滤波去除椒盐噪声。数据噪声多时建议开启。',
    'median_filter_size': '中值滤波核大小，必须为奇数。值越大去噪越强，建议 3~5。',
    'use_clahe': 'CLAHE对比度增强，提升弱断层可见度。数据对比度低时开启。',
    'clahe_clip_limit': 'CLAHE对比度剪切限。值越大增强越强但可能放大噪声，建议 1.5~3.0。',
    'clahe_grid_size': 'CLAHE分块大小。值越大增强越全局，建议 8~16。',
    'use_gabor': 'Gabor方向性滤波，增强线性断层特征。有明显方向性断层时开启。',
    'gabor_frequency': 'Gabor滤波器频率参数。控制检测的断层频率尺度，建议 0.1~0.2。',
    'gabor_angles': 'Gabor方向数，在180°内等分。方向越多越精细但计算量增大，建议 4~6。',
    'use_unet': '使用UNet深度学习模型替代传统二值化。需要先训练并加载模型文件。',
    'threshold_mode': "二值化方法。Otsu自动阈值最常用；固定阈值适合数据稳定时；"
                      "自适应适合光照不均；滞后分割适合弱边缘。",
    'otsu_scale': 'Otsu阈值缩放系数。<1检测更多断层（灵敏），>1更保守（减少假阳性）。',
    'fixed_threshold': '固定二值化阈值（0~1）。仅在 threshold_mode=fixed 时生效。',
    'hysteresis_low': '滞后分割低阈值。低于此值的像素直接舍弃。仅在 threshold_mode=hysteresis 时生效。',
    'hysteresis_high': '滞后分割高阈值。高于此值的像素直接保留。仅在 threshold_mode=hysteresis 时生效。',
    'adaptive_block_size': '自适应阈值局部窗口大小（奇数）。仅在 threshold_mode=adaptive 时生效，建议 31~51。',
    'adaptive_c': '自适应阈值偏移常数。负值检测更多，正值更保守。仅在 threshold_mode=adaptive 时生效。',
    'morph_order': '形态学操作顺序。先开后闭：先去噪再填充缝隙（推荐）；先闭后开：先连接再清理。',
    'opening_radius': '开运算半径，去除孤立噪声点。值越大去除越多但可能丢失小断层，建议 0~3。',
    'closing_radius': '闭运算半径，填充断层间隙、连接断裂段。值越大连接越强但可能过度合并，建议 1~4。',
    'morph_kernel_shape': '形态学结构元素形状。disk圆形（通用）、ellipse椭圆（方向性）、cross十字（正交）。',
    'min_component_area': '最小连通域面积（像素）。小于此值的连通域被视为噪声丢弃。',
    'separate_intersections': '在断层交叉处分离不同断层。开启后每条断层更清晰独立。',
    'max_aspect_ratio': '最大长宽比过滤。超出此值的连通域视为非断层噪声丢弃。',
    'max_compactness': '最大紧致度过滤（周长²/面积）。值越大越不规则，超出视为噪声丢弃。',
    'contour_smooth_sigma': '轮廓高斯平滑σ。0=不平滑保留断层棱角（推荐），>0产生更圆滑的多边形边界。',
    'polygon_mode': '多边形提取模式。skeleton骨架细线（推荐，更精确）；region区域轮廓（更粗犷）。',
    'skeleton_buffer': '骨架缓冲半径（像素）。skeleton模式下在骨架周围膨胀的宽度，建议 1~3。',
    'min_polygon_area': '最小多边形面积（像素²）。小于此值的多边形被丢弃，skeleton模式通常设 10~30。',
    'dp_mode': 'Douglas-Peucker简化模式。absolute绝对像素容差；relative按周长比例。',
    'dp_epsilon': 'DP简化容差（像素）。越小多边形顶点越多越精细。仅在 dp_mode=absolute 时生效。',
    'dp_ratio': 'DP简化比例（周长×此值=容差）。仅在 dp_mode=relative 时生效，建议 0.003~0.01。',
    'smooth_iterations': 'Chaikin平滑迭代次数。0=不平滑保留断层棱角（推荐），>0产生更圆滑的边界。',
    'track_max_link_distance': '断层追踪最大连接距离（像素）。只有在此距离内的断裂段才被考虑连接。',
    'track_angle_weight': '方向一致性权重（0~10）。越大越严格要求断裂段方向一致才连接。',
    'track_min_segment_length': '最小参与连接的片段长度（像素）。短于此值的孤立片段不参与连接。',
    'track_dilate_radius': '连接后膨胀半径。skeleton模式下用1，防止过度合并。',
    'track_dilate_iterations': '膨胀迭代次数。skeleton模式下用2，防止过度合并。',
}
