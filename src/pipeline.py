"""断层多边形追踪流水线 — UI 无关的核心逻辑"""

import time
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import skeletonize

from config import Config
from .preprocess import normalize
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
        dict 包含每步中间结果：data_smoothed, binary_before_morph, binary,
        binary_before_track, skeleton, junctions, contours, vectorized,
        filtered, areas, elapsed
    """
    t0 = time.perf_counter()

    # --- 步骤 1: 预处理（归一化 + 去噪） ---
    data_norm = normalize(data)
    smoothed = gaussian_filter(data_norm, sigma=cfg.gaussian_sigma)

    # --- 步骤 2: 二值化（形态学处理前） ---
    if cfg.use_adaptive_threshold:
        block = max(3, cfg.adaptive_block_size)
        if block % 2 == 0:
            block += 1
        img_uint8 = (smoothed * 255).astype(np.uint8)
        local_thresh = threshold_local(img_uint8, block, method='gaussian',
                                        offset=cfg.adaptive_c * 255)
        binary_before_morph = (smoothed >= (local_thresh / 255.0)).astype(np.uint8)
    else:
        thresh = threshold_otsu(smoothed) * cfg.otsu_scale
        binary_before_morph = (smoothed >= thresh).astype(np.uint8)

    # --- 步骤 3: 形态学处理 ---
    binary = segment_fault_regions(
        data, sigma=cfg.gaussian_sigma, otsu_scale=cfg.otsu_scale,
        closing_radius=cfg.closing_radius, opening_radius=cfg.opening_radius,
        use_adaptive_threshold=cfg.use_adaptive_threshold,
        adaptive_block_size=cfg.adaptive_block_size, adaptive_c=cfg.adaptive_c,
    )

    # --- 步骤 3.5: 断层追踪 ---
    binary_before_track = binary.copy()
    binary = track_faults(
        binary, max_link_distance=cfg.track_max_link_distance,
        angle_weight=cfg.track_angle_weight,
        min_segment_length=cfg.track_min_segment_length,
        dilate_radius=cfg.track_dilate_radius,
        dilate_iterations=cfg.track_dilate_iterations,
        raw_data=data,
    )

    # --- 步骤 4: 轮廓提取 ---
    contours = extract_fault_polygons(
        binary, min_component_area=cfg.min_component_area,
        separate_intersections=cfg.separate_intersections,
        smooth_sigma=cfg.contour_smooth_sigma,
    )

    # --- 步骤 5: 骨架 + 交叉点 ---
    skel = skeletonize(binary.astype(bool))
    junctions = _find_junctions(skel)

    # --- 步骤 6: 矢量化 + 过滤 ---
    vectorized = [
        simplify_polygon(c, cfg.dp_epsilon, cfg.smooth_iterations)
        for c in contours
    ]
    filtered = filter_by_area(vectorized, cfg.min_polygon_area)
    areas = [polygon_area(p) for p in filtered]

    # --- 多尺度融合 ---
    if cfg.scales and len(cfg.scales) > 1:
        all_polygons = [filtered]
        all_areas_list = [areas]
        for scale_sigma in cfg.scales[1:]:
            binary_s = segment_fault_regions(
                data, sigma=scale_sigma, otsu_scale=cfg.otsu_scale,
                closing_radius=cfg.closing_radius, opening_radius=cfg.opening_radius,
            )
            binary_s = track_faults(
                binary_s, max_link_distance=cfg.track_max_link_distance,
                angle_weight=cfg.track_angle_weight,
                min_segment_length=cfg.track_min_segment_length,
                dilate_radius=cfg.track_dilate_radius,
                dilate_iterations=cfg.track_dilate_iterations,
                raw_data=data,
            )
            contours_s = extract_fault_polygons(
                binary_s, min_component_area=cfg.min_component_area,
                separate_intersections=cfg.separate_intersections,
                smooth_sigma=cfg.contour_smooth_sigma,
            )
            vectorized_s = [simplify_polygon(c, cfg.dp_epsilon, cfg.smooth_iterations)
                           for c in contours_s]
            filtered_s = filter_by_area(vectorized_s, cfg.min_polygon_area)
            areas_s = [polygon_area(p) for p in filtered_s]
            all_polygons.append(filtered_s)
            all_areas_list.append(areas_s)
        filtered, areas = merge_multiscale_results(
            all_polygons, all_areas_list, cfg.dedup_overlap_threshold)

    elapsed = time.perf_counter() - t0

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
        'elapsed': round(elapsed, 3),
    }
