"""合成断层属性数据生成模块"""

import numpy as np


def _draw_thick_line(data: np.ndarray, r1: int, c1: int,
                     r2: int, c2: int, strength: float = 1.0,
                     width: int = 3):
    """在数组上画具有一定宽度的线（模拟断层区域）"""
    h, w = data.shape
    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    sr = 1 if r1 < r2 else -1
    sc = 1 if c1 < c2 else -1
    err = dr - dc
    r, c = r1, c1

    while True:
        for wr in range(-width, width + 1):
            for wc in range(-width, width + 1):
                rr, cc = r + wr, c + wc
                if 0 <= rr < h and 0 <= cc < w:
                    dist = np.sqrt(wr ** 2 + wc ** 2)
                    wgt = np.exp(-0.5 * (dist / max(width, 1)) ** 2)
                    data[rr, cc] = max(data[rr, cc], strength * wgt)
        if r == r2 and c == c2:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


def generate_synthetic_data(shape: tuple = (300, 400),
                             n_faults: int = 5,
                             noise_level: float = 0.03,
                             seed: int = 42) -> np.ndarray:
    """生成包含弯曲断层、断缝、分支的合成测试数据"""
    rng = np.random.default_rng(seed)
    data = np.full(shape, 0.05)
    data += rng.normal(0, noise_level, shape)

    for _ in range(n_faults):
        r1 = rng.integers(20, shape[0] - 20)
        c1 = rng.integers(0, shape[1] // 5)
        r2 = rng.integers(20, shape[0] - 20)
        c2 = rng.integers(4 * shape[1] // 5, shape[1] - 1)

        strength = rng.uniform(0.5, 0.95)
        width = rng.integers(3, 10)
        curvature_amp = rng.uniform(0, 30)
        curvature_freq = rng.uniform(0.005, 0.03)

        n_steps = int(max(abs(r2 - r1), abs(c2 - c1))) * 2
        for t in np.linspace(0, 1, n_steps):
            c_pos = c1 + t * (c2 - c1)
            r_base = r1 + t * (r2 - r1)
            r_offset = curvature_amp * np.sin(2 * np.pi * curvature_freq * (c_pos - c1))
            r_pos = r_base + r_offset

            for wr in range(-width, width + 1):
                for wc in range(-width, width + 1):
                    rr = int(round(r_pos)) + wr
                    cc = int(round(c_pos)) + wc
                    if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                        dist = np.sqrt(wr ** 2 + wc ** 2)
                        wgt = np.exp(-0.5 * (dist / max(width, 1)) ** 2)
                        grad = 0.7 + 0.3 * np.sin(np.pi * t)
                        data[rr, cc] = max(data[rr, cc], strength * wgt * grad)

        n_gaps = rng.integers(0, 3)
        for _ in range(n_gaps):
            gap_center = rng.uniform(0.2, 0.8)
            gap_width = rng.uniform(0.02, 0.06)
            t_lo, t_hi = gap_center - gap_width, gap_center + gap_width
            for t in np.linspace(max(0, t_lo), min(1, t_hi), 20):
                c_pos = c1 + t * (c2 - c1)
                r_base = r1 + t * (r2 - r1)
                r_offset = curvature_amp * np.sin(2 * np.pi * curvature_freq * (c_pos - c1))
                r_pos = r_base + r_offset
                for wr in range(-width - 2, width + 3):
                    for wc in range(-width - 2, width + 3):
                        rr = int(round(r_pos)) + wr
                        cc = int(round(c_pos)) + wc
                        if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                            data[rr, cc] = min(data[rr, cc], 0.1)

        if rng.random() > 0.5:
            branch_t = rng.uniform(0.3, 0.7)
            br = int(r1 + branch_t * (r2 - r1))
            bc = int(c1 + branch_t * (c2 - c1))
            mr = br + rng.integers(-40, 40)
            mc = bc + rng.integers(-40, 40)
            mr = np.clip(mr, 20, shape[0] - 20)
            mc = np.clip(mc, 10, shape[1] - 10)
            _draw_thick_line(data, br, bc, mr, mc,
                             strength=rng.uniform(0.3, 0.6),
                             width=rng.integers(2, 5))

    return np.clip(data, 0, 1)
