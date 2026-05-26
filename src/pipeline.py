"""断层多边形追踪流水线 — UI 无关的核心逻辑"""

import time
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import skeletonize

from config import Config
from .preprocess import normalize, apply_gabor_filter
from .segment import segment_fault_regions
from .tracker import track_faults
from .polygon_extract import extract_fault_polygons
from .vectorize import simplify_polygon, filter_by_area, polygon_area
from .multiscale import merge_multiscale_results


def _find_junctions(skel: np.ndarray) -> list:
    """在骨架上找到交叉点（邻域内 ≥3 个前景像素）"""
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


def run_pipeline(data: np.ndarray, cfg: Config) -> dict:
    """运行完整断层多边形追踪流水线。

    Args:
        data: 2D numpy 数组，断层属性响应值
        cfg: Config 参数配置

    Returns:
        dict: 包含所有处理步骤的中间结果和最终统计
    """
    t0 = time.perf_counter()
    step_times = {}

    # --- 步骤 1: 预处理（归一化 + 可选Gabor + 去噪） ---
    t1 = time.perf_counter()
    data_norm = normalize(data)

    if cfg.use_gabor:
        data_norm = apply_gabor_filter(data_norm, frequency=cfg.gabor_frequency,
                                        n_angles=cfg.gabor_angles)

    if cfg.use_median_filter:
        size = max(3, cfg.median_filter_size)
        if size % 2 == 0:
            size += 1
        data_norm = median_filter(data_norm, size=size)

    smoothed = gaussian_filter(data_norm, sigma=cfg.gaussian_sigma)

    # --- 步骤 1.5: 可选 UNet 分割 ---
    if cfg.use_unet and cfg.unet_model_path:
        from .preprocess import UNetFault
        try:
            unet = UNetFault(cfg.unet_model_path)
            prob = unet.predict(data)
            smoothed = gaussian_filter(prob, sigma=0.5)
        except Exception as e:
            print(f"[pipeline] UNet 预测失败，回退到传统方法: {e}")

    step_times['preprocess'] = round(time.perf_counter() - t1, 3)

    # --- 步骤 2: 二值化 ---
    t2 = time.perf_counter()
    mode = cfg.threshold_mode
    if cfg.use_adaptive_threshold and mode == 'otsu':
        mode = 'adaptive'

    if mode == 'fixed':
        binary_before_morph = (smoothed >= cfg.fixed_threshold).astype(np.uint8)
    elif mode == 'hysteresis':
        from .segment import _hysteresis_threshold
        binary_before_morph = _hysteresis_threshold(
            smoothed, cfg.hysteresis_low, cfg.hysteresis_high)
    elif mode == 'adaptive':
        block = max(3, cfg.adaptive_block_size)
        if block % 2 == 0:
            block += 1
        img_uint8 = (smoothed * 255).astype(np.uint8)
        local_thresh = threshold_local(img_uint8, block, method='gaussian',
                                        offset=cfg.adaptive_c * 255)
        binary_before_morph = (smoothed >= (local_thresh / 255.0)).astype(np.uint8)
    else:  # 'otsu'
        thresh = threshold_otsu(smoothed) * cfg.otsu_scale
        binary_before_morph = (smoothed >= thresh).astype(np.uint8)

    step_times['binarize'] = round(time.perf_counter() - t2, 3)

    # --- 步骤 3: 形态学处理 ---
    t3 = time.perf_counter()
    binary = segment_fault_regions(
        data, sigma=cfg.gaussian_sigma, otsu_scale=cfg.otsu_scale,
        closing_radius=cfg.closing_radius, opening_radius=cfg.opening_radius,
        use_adaptive_threshold=cfg.use_adaptive_threshold,
        adaptive_block_size=cfg.adaptive_block_size, adaptive_c=cfg.adaptive_c,
        use_median_filter=cfg.use_median_filter,
        median_filter_size=cfg.median_filter_size,
        use_gabor=cfg.use_gabor,
        gabor_frequency=cfg.gabor_frequency,
        gabor_angles=cfg.gabor_angles,
        threshold_mode=cfg.threshold_mode,
        fixed_threshold=cfg.fixed_threshold,
        hysteresis_low=cfg.hysteresis_low,
        hysteresis_high=cfg.hysteresis_high,
        morph_order=cfg.morph_order,
        morph_kernel_shape=cfg.morph_kernel_shape,
        precomputed_binary=binary_before_morph,
    )

    step_times['morph'] = round(time.perf_counter() - t3, 3)

    # --- 步骤 3.5: 断层追踪 ---
    # skeleton模式跳过track_faults：segment的闭运算已连接断缝，
    # track的膨胀反而合并所有断层，之后又被skeletonize一次浪费计算
    t_track = 0.0
    binary_before_track = binary.copy()
    if cfg.polygon_mode != 'skeleton':
        t_track_start = time.perf_counter()
        binary = track_faults(
            binary, max_link_distance=cfg.track_max_link_distance,
            angle_weight=cfg.track_angle_weight,
            min_segment_length=cfg.track_min_segment_length,
            dilate_radius=cfg.track_dilate_radius,
            dilate_iterations=cfg.track_dilate_iterations,
            raw_data=None,
        )
        t_track = round(time.perf_counter() - t_track_start, 3)
    step_times['track'] = t_track

    # --- 步骤 4: 轮廓提取（含形状过滤） ---
    t4 = time.perf_counter()
    # 预先骨架化一次，避免 extract_fault_polygons 内部重复 skeletonize
    skel = skeletonize(binary.astype(bool))
    contours, components = extract_fault_polygons(
        binary, min_component_area=cfg.min_component_area,
        separate_intersections=cfg.separate_intersections,
        smooth_sigma=cfg.contour_smooth_sigma,
        max_aspect_ratio=cfg.max_aspect_ratio,
        max_compactness=cfg.max_compactness,
        polygon_mode=cfg.polygon_mode,
        skeleton_buffer=cfg.skeleton_buffer,
        precomputed_skel=skel if cfg.polygon_mode == 'skeleton' else None,
    )

    # --- 步骤 5: 交叉点 ---
    junctions = _find_junctions(skel)

    step_times['contour_extract'] = round(time.perf_counter() - t4, 3)

    # --- 步骤 6: 矢量化 + 过滤 ---
    t5 = time.perf_counter()
    vectorized = [
        simplify_polygon(c, cfg.dp_epsilon, cfg.smooth_iterations,
                         dp_mode=cfg.dp_mode, dp_ratio=cfg.dp_ratio)
        for c in contours
    ]
    filtered = filter_by_area(vectorized, cfg.min_polygon_area)
    areas = [polygon_area(p) for p in filtered]

    step_times['vectorize'] = round(time.perf_counter() - t5, 3)

    # --- 多尺度融合 ---
    t_ms = 0.0
    if cfg.scales and len(cfg.scales) > 1:
        t_ms_start = time.perf_counter()
        all_polygons = [filtered]
        all_areas_list = [areas]
        for scale_sigma in cfg.scales[1:]:
            binary_s = segment_fault_regions(
                data, sigma=scale_sigma, otsu_scale=cfg.otsu_scale,
                closing_radius=cfg.closing_radius, opening_radius=cfg.opening_radius,
                use_median_filter=cfg.use_median_filter,
                median_filter_size=cfg.median_filter_size,
                use_gabor=cfg.use_gabor,
                gabor_frequency=cfg.gabor_frequency,
                gabor_angles=cfg.gabor_angles,
                threshold_mode=cfg.threshold_mode,
                fixed_threshold=cfg.fixed_threshold,
                hysteresis_low=cfg.hysteresis_low,
                hysteresis_high=cfg.hysteresis_high,
                morph_order=cfg.morph_order,
                morph_kernel_shape=cfg.morph_kernel_shape,
            )
            if cfg.polygon_mode != 'skeleton':
                binary_s = track_faults(
                    binary_s, max_link_distance=cfg.track_max_link_distance,
                    angle_weight=cfg.track_angle_weight,
                    min_segment_length=cfg.track_min_segment_length,
                    dilate_radius=cfg.track_dilate_radius,
                    dilate_iterations=cfg.track_dilate_iterations,
                    raw_data=None,
                )
            contours_s, _ = extract_fault_polygons(
                binary_s, min_component_area=cfg.min_component_area,
                separate_intersections=cfg.separate_intersections,
                smooth_sigma=cfg.contour_smooth_sigma,
                max_aspect_ratio=cfg.max_aspect_ratio,
                max_compactness=cfg.max_compactness,
                polygon_mode=cfg.polygon_mode,
                skeleton_buffer=cfg.skeleton_buffer,
            )
            vectorized_s = [simplify_polygon(c, cfg.dp_epsilon, cfg.smooth_iterations,
                                             dp_mode=cfg.dp_mode, dp_ratio=cfg.dp_ratio)
                           for c in contours_s]
            filtered_s = filter_by_area(vectorized_s, cfg.min_polygon_area)
            areas_s = [polygon_area(p) for p in filtered_s]
            all_polygons.append(filtered_s)
            all_areas_list.append(areas_s)
        filtered, areas = merge_multiscale_results(
            all_polygons, all_areas_list, cfg.dedup_overlap_threshold)
        t_ms = round(time.perf_counter() - t_ms_start, 3)
    step_times['multiscale'] = t_ms

    # --- 统计 ---
    elapsed = time.perf_counter() - t0

    total_area = sum(areas) if areas else 0.0
    from .vectorize import polygon_perimeter
    total_length = sum(polygon_perimeter(p) / 2.0 for p in filtered) if filtered else 0.0
    min_area = min(areas) if areas else 0.0
    max_area = max(areas) if areas else 0.0

    return {
        'data_smoothed': smoothed,
        'binary_before_morph': binary_before_morph,
        'binary': binary,
        'binary_before_track': binary_before_track,
        'skeleton': skel,
        'junctions': junctions,
        'contours': contours,
        'vectorized': vectorized,
        'filtered': filtered,
        'areas': areas,
        'components': components,
        'elapsed': round(elapsed, 3),
        'total_area': round(total_area, 2),
        'total_length': round(total_length, 2),
        'min_area': round(min_area, 2),
        'max_area': round(max_area, 2),
        'count': len(filtered),
        'step_times': step_times,
    }
