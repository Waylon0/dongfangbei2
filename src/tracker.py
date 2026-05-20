"""断层追踪模块 — 方向引导的片段连接

在二值化之后、轮廓提取之前，将断续的断层片段沿走向连接成完整断层。
核心思路：
1. 骨架化提取断层中心线
2. 在交叉点断开，得到独立曲线片段
3. PCA 估计每段走向
4. 图模型匹配：距离 + 方向一致性 → 连接代价
5. 贪心连接，膨胀恢复宽度
"""

import numpy as np
from scipy.ndimage import label, binary_dilation, distance_transform_edt, gaussian_filter
from skimage.morphology import skeletonize, disk
from typing import List, Tuple, Optional


class SkeletonSegment:
    """骨架片段：一段连续的1像素宽曲线"""

    def __init__(self, points: np.ndarray, seg_id: int,
                 binary_mask: np.ndarray = None):
        self.points = points  # (N, 2) [row, col]
        self.seg_id = seg_id
        self.length = len(points)

        self.start = points[0].copy()
        self.end = points[-1].copy()

        self.start_dir = self._direction_at(end=False)
        self.end_dir = self._direction_at(end=True)

        # 将端点延伸到二值区域边界
        if binary_mask is not None:
            self.start = self._extend_to_boundary(self.start, self.start_dir, binary_mask)
            self.end = self._extend_to_boundary(self.end, self.end_dir, binary_mask)

    def _direction_at(self, end: bool, window: int = 8) -> np.ndarray:
        """估算端点处的走向向量（单位向量）"""
        n = len(self.points)
        w = min(window, n // 2)
        if w < 2:
            return np.array([0.0, 0.0])

        if end:
            pts = self.points[-w:]
        else:
            pts = self.points[:w]

        # PCA 主方向
        centered = pts - pts.mean(axis=0)
        if len(centered) < 2:
            return np.array([0.0, 0.0])
        cov = np.cov(centered[:, 1], centered[:, 0])  # (col, row) -> (x, y)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal = eigenvectors[:, -1]  # 最大特征值方向

        # 确保方向指向端点外侧
        direction = np.array([principal[1], principal[0]])  # 转回 (row, col)
        tip = self.end if end else self.start
        center = pts.mean(axis=0)
        if np.dot(direction, tip - center) < 0:
            direction = -direction

        norm = np.linalg.norm(direction)
        return direction / norm if norm > 1e-10 else np.array([0.0, 0.0])

    def _extend_to_boundary(self, tip: np.ndarray, direction: np.ndarray,
                             binary_mask: np.ndarray, max_steps: int = 50) -> np.ndarray:
        """从骨架端点沿方向外推到二值区域边界，返回边界上的点。"""
        if np.linalg.norm(direction) < 1e-10:
            return tip
        h, w = binary_mask.shape
        pos = tip.copy().astype(float)
        step = direction / np.linalg.norm(direction)

        for _ in range(max_steps):
            next_pos = pos + step
            r, c = int(round(next_pos[0])), int(round(next_pos[1]))
            if r < 0 or r >= h or c < 0 or c >= w:
                break
            if binary_mask[r, c] == 0:
                break
            pos = next_pos

        return pos


def _skeleton_to_segments(skel: np.ndarray,
                           min_length: int = 10,
                           binary_mask: np.ndarray = None) -> List[SkeletonSegment]:
    """将骨架图拆分为独立片段（在交叉点处断开）"""
    h, w = skel.shape
    skel_coords = set(map(tuple, np.argwhere(skel > 0)))

    # 找交叉点（度数 >= 3）和端点（度数 == 1）
    junctions = set()
    for r, c in skel_coords:
        cnt = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in skel_coords:
                    cnt += 1
        if cnt >= 3:
            junctions.add((r, c))

    # 断开交叉点
    skel_split = skel.copy().astype(np.uint8)
    for r, c in junctions:
        skel_split[r, c] = 0
        # 交叉点周围也断开，确保彻底分离
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if (r + dr, c + dc) in junctions:
                    skel_split[r + dr, c + dc] = 0

    # 连通域标记 → 每个域就是一个片段
    labeled, n_features = label(skel_split)
    segments = []
    for seg_id in range(1, n_features + 1):
        coords = np.argwhere(labeled == seg_id)  # (N, 2) [row, col]
        if len(coords) < min_length:
            continue
        # 沿曲线排序
        ordered = _order_points(coords)
        if ordered is not None:
            segments.append(SkeletonSegment(ordered, seg_id, binary_mask))

    return segments


def _order_points(coords: np.ndarray) -> Optional[np.ndarray]:
    """将无序的骨架点集排序成一条连续曲线"""
    if len(coords) < 2:
        return None

    coords_set = set(map(tuple, coords))
    visited = set()

    # 找端点（只有一个邻居的点）
    endpoints = []
    for r, c in coords:
        neighbors = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in coords_set:
                    neighbors += 1
        if neighbors == 1:
            endpoints.append((r, c))

    # 如果没有端点（环形），从任意点开始
    start = endpoints[0] if endpoints else tuple(coords[0])

    # DFS 遍历排序
    ordered = [start]
    visited.add(start)

    for _ in range(len(coords) - 1):
        r, c = ordered[-1]
        best = None
        best_dist = 999
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) in coords_set and (nr, nc) not in visited:
                    # 优先选择8-连通最近邻
                    dist = abs(dr) + abs(dc)
                    if dist < best_dist:
                        best_dist = dist
                        best = (nr, nc)
        if best is None:
            break
        ordered.append(best)
        visited.add(best)

    result = np.array(ordered)
    if len(result) < 2:
        return None
    return result


def _angle_cost(dir1: np.ndarray, dir2: np.ndarray) -> float:
    """两个方向向量之间的角度代价 [0, 1]，越接近代价越低"""
    dot = np.clip(np.dot(dir1, dir2), -1.0, 1.0)
    # cos(angle) 接近 1 或 -1 都表示方向一致
    return 1.0 - abs(dot)


def _connection_cost(seg_a: SkeletonSegment, end_a: bool,
                     seg_b: SkeletonSegment, end_b: bool,
                     max_distance: float, angle_weight: float,
                     gradient_orientation: np.ndarray = None) -> float:
    """计算两个片段端点之间的连接代价。

    可选 gradient_orientation: 全局梯度方向图 (H, W)，用于验证连接线方向与梯度一致性。
    """
    pt_a = seg_a.end if end_a else seg_a.start
    pt_b = seg_b.end if end_b else seg_b.start

    dist = np.linalg.norm(pt_a - pt_b)
    if dist > max_distance or dist < 1.0:
        return float('inf')

    dir_a = seg_a.end_dir if end_a else seg_a.start_dir
    dir_b = seg_b.end_dir if end_b else seg_b.start_dir

    # 方向代价：端点方向应指向对方
    vec_ab = (pt_b - pt_a)
    norm_ab = np.linalg.norm(vec_ab)
    if norm_ab < 1e-10:
        return float('inf')
    vec_ab_unit = vec_ab / norm_ab

    # seg_a 的端点方向应大致指向 pt_b
    align_a = np.dot(dir_a, vec_ab_unit)
    # seg_b 的端点方向应大致指向 pt_a（反向）
    align_b = np.dot(dir_b, -vec_ab_unit)

    # 两个对齐都应 > 0
    if align_a < -0.3 or align_b < -0.3:
        return float('inf')

    angle_penalty = (1.0 - align_a) + (1.0 - align_b)
    cost = dist + angle_weight * angle_penalty * dist

    # 梯度方向验证：沿连接线采样梯度方向，检查是否一致
    if gradient_orientation is not None:
        h, w = gradient_orientation.shape
        n_samples = max(3, int(dist // 5))
        grad_consistency = _check_gradient_consistency(
            pt_a, pt_b, gradient_orientation, n_samples)
        # grad_consistency 越低越不一致，增加代价
        cost += (1.0 - grad_consistency) * dist * 0.5

    return cost


def _check_gradient_consistency(pt_a: np.ndarray, pt_b: np.ndarray,
                                 gradient_orientation: np.ndarray,
                                 n_samples: int = 5) -> float:
    """沿连接线采样梯度方向，检查是否与连接方向一致。返回 [0, 1] 一致性分数。"""
    h, w = gradient_orientation.shape
    vec = pt_b - pt_a
    link_angle = np.arctan2(vec[0], vec[1])  # (row, col) -> atan2(dr, dc)

    orientations = []
    for t in np.linspace(0.1, 0.9, n_samples):
        pt = pt_a + t * vec
        r, c = int(round(pt[0])), int(round(pt[1]))
        if 0 <= r < h and 0 <= c < w:
            orientations.append(gradient_orientation[r, c])

    if not orientations:
        return 0.5

    # 计算每个采样点梯度方向与连接方向的最小夹角
    angles = np.array(orientations)
    diffs = np.abs(np.cos(angles) * np.cos(link_angle) + np.sin(angles) * np.sin(link_angle))
    # |cos(diff)| 接近1表示方向一致（不管正反）
    consistencies = np.abs(diffs)
    return float(np.mean(consistencies))


def track_faults(binary: np.ndarray,
                  max_link_distance: float = 30.0,
                  angle_weight: float = 2.0,
                  min_segment_length: int = 10,
                  dilate_radius: int = 3,
                  dilate_iterations: int = 5,
                  raw_data: np.ndarray = None) -> np.ndarray:
    """断层追踪主函数：将断续的断层片段沿走向连接。

    参数：
        binary: 断层区域二值掩膜 (0/1)
        max_link_distance: 最大连接距离（像素）
        angle_weight: 方向一致性权重
        min_segment_length: 最小片段长度（过短的片段不参与连接）
        dilate_radius: 膨胀半径（恢复宽度）
        dilate_iterations: 膨胀迭代次数
        raw_data: 原始属性数据（可选，用于梯度方向验证）

    返回：
        连接后的二值掩膜
    """
    if binary.sum() == 0:
        return binary

    skel = skeletonize(binary.astype(bool))

    # 1. 提取片段
    segments = _skeleton_to_segments(skel, min_length=min_segment_length,
                                      binary_mask=binary)

    if len(segments) < 2:
        return binary

    # 2. 计算梯度方向图（用于验证连接方向一致性）
    grad_ori = None
    if raw_data is not None:
        from src.utils import normalize
        data_norm = normalize(raw_data)
        smoothed_data = gaussian_filter(data_norm, sigma=2.0)
        gy, gx = np.gradient(smoothed_data)
        grad_ori = np.arctan2(gy, gx)

    # 3. 构建候选连接

    # 2. 构建候选连接
    candidates = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            for end_a in (False, True):
                for end_b in (False, True):
                    cost = _connection_cost(
                        segments[i], end_a,
                        segments[j], end_b,
                        max_link_distance, angle_weight,
                        gradient_orientation=grad_ori,
                    )
                    if cost < float('inf'):
                        pt_a = segments[i].end if end_a else segments[i].start
                        pt_b = segments[j].end if end_b else segments[j].start
                        candidates.append((cost, i, end_a, j, end_b, pt_a, pt_b))

    candidates.sort(key=lambda x: x[0])

    # 3. 贪心连接（每个端点最多连一次）
    used_endpoints = set()
    connected_skel = skel.copy().astype(np.uint8)

    for cost, idx_a, end_a, idx_b, end_b, pt_a, pt_b in candidates:
        key_a = (idx_a, end_a)
        key_b = (idx_b, end_b)
        if key_a in used_endpoints or key_b in used_endpoints:
            continue

        # 画连接线（Bresenham）
        _draw_line(connected_skel, int(pt_a[0]), int(pt_a[1]),
                   int(pt_b[0]), int(pt_b[1]))

        used_endpoints.add(key_a)
        used_endpoints.add(key_b)

    # 4. 膨胀恢复宽度
    # 连接后的骨架整体膨胀，裁剪到原始区域 + 连接线走廊
    connected_dilated = binary_dilation(
        connected_skel.astype(bool),
        structure=disk(dilate_radius),
        iterations=dilate_iterations,
    )

    # 限制：膨胀区域不能离原始区域太远
    # dist_to_original = 每个非原始区域像素到最近原始区域像素的距离
    dist_to_original = distance_transform_edt(~binary.astype(bool))
    # 连接走廊宽度 = max_link_distance，允许连接处在这个范围内
    result = binary.astype(bool) | (connected_dilated & (dist_to_original <= max_link_distance))

    return result.astype(np.uint8)


def _draw_line(img: np.ndarray, r1: int, c1: int, r2: int, c2: int):
    """Bresenham 画线"""
    h, w = img.shape
    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    sr = 1 if r1 < r2 else -1
    sc = 1 if c1 < c2 else -1
    err = dr - dc
    r, c = r1, c1

    while True:
        if 0 <= r < h and 0 <= c < w:
            img[r, c] = 1
            # 稍微加粗，保证连通
            for wr in (-1, 0, 1):
                for wc in (-1, 0, 1):
                    rr, cc = r + wr, c + wc
                    if 0 <= rr < h and 0 <= cc < w:
                        img[rr, cc] = 1
        if r == r2 and c == c2:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
