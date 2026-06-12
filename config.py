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
    otsu_scale: float = 1.0           # Otsu阈值缩放（>1更保守，<1更敏感）
    fixed_threshold: float = 0.6      # 固定阈值（threshold_mode='fixed'时生效）
    use_adaptive_threshold: bool = False  # 使用局部自适应阈值（已废弃，改用threshold_mode）
    adaptive_block_size: int = 35     # 自适应阈值局部窗口大小（奇数）
    adaptive_c: float = 0.0           # 自适应阈值偏移常数

    # --- 双阈值滞后分割 ---
    hysteresis_low: float = 0.3       # 低阈值
    hysteresis_high: float = 0.6      # 高阈值

    # --- 形态学处理 ---
    morph_order: str = 'open_first'   # 操作顺序：'open_first'先开后闭 / 'close_first'先闭后开
    closing_radius: int = 3           # 闭运算半径（skeleton模式宜小，避免过度合并断层）
    opening_radius: int = 2           # 开运算半径（去除噪点）
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
    scales: list = [1.0, 2.0]         # 多尺度σ列表（单尺度用 []）
    dedup_overlap_threshold: float = 0.5  # 去重重合度（IoU）

    # --- 断层追踪 ---
    track_max_link_distance: float = 30.0   # 最大连接距离（像素）
    track_angle_weight: float = 2.0         # 方向一致性权重（越大越严格）
    track_min_segment_length: int = 10      # 最小参与连接的片段长度
    track_dilate_radius: int = 1            # 连接后膨胀半径（skeleton模式用1，防止过度合并）
    track_dilate_iterations: int = 2        # 膨胀迭代次数（skeleton模式用2，防止过度合并）

    # --- 导出 ---
    output_format: str = 'geojson'    # geojson / txt / csv / shp
