"""数据加载与预处理

支持多种数据格式：
- .npy / .npz（NumPy）
- .dat（GeoEast ASCII 网格）
- .png / .tiff / .bmp / .jpg（图像文件）
- .txt（空格/逗号分隔的矩阵文本）

去噪与增强：
- 中值滤波 / 高斯滤波 / CLAHE / Gabor 方向性滤波
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage import exposure
from .utils import normalize


def apply_gabor_filter(data: np.ndarray, frequency: float = 0.15,
                        n_angles: int = 4) -> np.ndarray:
    """Gabor 方向性滤波：增强特定方向的线性结构（断层）。

    对多个方向做 Gabor 滤波，取每个像素在所有方向上的最大响应。
    适合增强有方向性条带的弱断层信号。

    参数：
        data: 归一化后的属性数据 (2D, 0~1)
        frequency: Gabor 核频率（控制条纹宽度）
        n_angles: 方向数量（等分 180°）

    返回：
        增强后的图像 (2D, 0~1)
    """
    from scipy.ndimage import correlate
    h, w = data.shape
    result = np.zeros_like(data)

    for k in range(n_angles):
        theta = k * np.pi / n_angles
        # 构建 Gabor 核
        sigma_x = 1.0 / frequency * 0.7
        sigma_y = sigma_x * 3.0
        kernel_size = int(4.0 / frequency)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(7, min(kernel_size, min(h, w) // 4))

        ks2 = kernel_size // 2
        y, x = np.mgrid[-ks2:ks2+1, -ks2:ks2+1]
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        gb = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2))
        gb *= np.cos(2 * np.pi * frequency * x_theta)
        # 去均值，使其成为零和滤波器
        gb -= gb.mean()
        gb /= np.abs(gb).sum() + 1e-10

        filtered = correlate(data.astype(np.float64), gb, mode='reflect')
        result = np.maximum(result, np.abs(filtered))

    # 归一化回 [0, 1]
    mn, mx = result.min(), result.max()
    if mx > mn:
        result = (result - mn) / (mx - mn)
    return result


def _load_dat(filepath: str) -> np.ndarray:
    """解析 GeoEast 导出的 ASCII .dat 格式属性网格文件。

    格式：空格分隔，首行为 # 注释头，后续每行：
    Line  CMP  X  Y  Value
    根据 Line 和 CMP 的唯一个数构建规则网格。
    """
    lines_set = set()
    cmps_set = set()

    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            lines_set.add(int(parts[0]))
            cmps_set.add(int(parts[1]))

    lines_vals = sorted(lines_set)
    cmps_vals = sorted(cmps_set)
    line_to_idx = {v: i for i, v in enumerate(lines_vals)}
    cmp_to_idx = {v: i for i, v in enumerate(cmps_vals)}

    data = np.zeros((len(lines_vals), len(cmps_vals)), dtype=np.float64)

    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            li = int(parts[0])
            ci = int(parts[1])
            val = float(parts[4])
            data[line_to_idx[li], cmp_to_idx[ci]] = val

    return data


def _load_image(filepath: str) -> np.ndarray:
    """从图像文件加载为灰度二维数组 (float64)"""
    try:
        import cv2
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像文件: {filepath}")
        return img.astype(np.float64)
    except ImportError:
        from skimage import io
        img = io.imread(filepath, as_gray=True)
        return img.astype(np.float64)


def _load_txt_matrix(filepath: str) -> np.ndarray:
    """从 .txt 文件加载数值矩阵（空格或逗号分隔）"""
    try:
        data = np.loadtxt(filepath)
        if data.ndim != 2:
            raise ValueError(f"TXT 文件不是二维矩阵: shape={data.shape}")
        return data.astype(np.float64)
    except Exception:
        pass
    # 尝试用 csv reader 处理更复杂的格式
    import csv
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=None)
        for row in reader:
            nums = []
            for val in row:
                try:
                    nums.append(float(val))
                except ValueError:
                    continue
            if nums:
                rows.append(nums)
    if len(set(len(r) for r in rows)) != 1:
        raise ValueError("TXT 文件各行列数不一致")
    return np.array(rows, dtype=np.float64)


def load_attribute_data(filepath: str) -> np.ndarray:
    """加载沿层属性数据。

    支持格式：.npy / .npz / .dat / .png / .tiff / .tif / .bmp / .jpg / .jpeg / .txt
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.dat':
        return _load_dat(filepath)
    elif ext in ('.png', '.tiff', '.tif', '.bmp', '.jpg', '.jpeg', '.webp'):
        return _load_image(filepath)
    elif ext == '.txt':
        return _load_txt_matrix(filepath)
    elif ext == '.npz':
        data = np.load(filepath)
        key = list(data.keys())[0]
        return data[key].astype(np.float64)
    else:
        return np.load(filepath).astype(np.float64)


def preprocess(data: np.ndarray,
               sigma: float = 1.0,
               use_median_filter: bool = False,
               median_filter_size: int = 3,
               use_clahe: bool = False,
               clahe_clip_limit: float = 2.0,
               clahe_grid_size: int = 8,
               otsu_scale: float = 1.0) -> np.ndarray:
    """经典预处理：归一化 → 可选中值滤波 → CLAHE(可选) → 高斯滤波 → Otsu 二值化。

    返回二值图像 (0/1)。
    """
    data = normalize(data)

    if use_median_filter:
        size = max(3, median_filter_size)
        if size % 2 == 0:
            size += 1
        data = median_filter(data, size=size)

    if use_clahe:
        img_uint8 = (data * 255).astype(np.uint8)
        data = exposure.equalize_adapthist(
            img_uint8,
            kernel_size=(clahe_grid_size, clahe_grid_size),
            clip_limit=clahe_clip_limit,
        )

    smoothed = gaussian_filter(data, sigma=sigma)

    from skimage.filters import threshold_otsu
    thresh = threshold_otsu(smoothed) * otsu_scale
    binary = (smoothed >= thresh).astype(np.uint8)

    return binary


class UNetFault:
    """轻量 3 级 UNet，用于断层属性图语义分割。

    参数量 ~0.8M，CPU 推理毫秒级。
    """

    def __init__(self, model_path: str = None):
        self.model = None
        self._device = 'cpu'
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def load(self, model_path: str):
        import torch
        self.model = _build_unet()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu',
                                              weights_only=True))
        self.model.eval()
        self.model.to(self._device)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """对单张属性图做推理，返回概率图 (0~1)。"""
        import torch
        if self.model is None:
            raise RuntimeError("UNet 模型未加载")

        data_norm = normalize(data)
        tensor = torch.from_numpy(data_norm).float().unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self._device)

        with torch.no_grad():
            prob = self.model(tensor).squeeze().cpu().numpy()

        return prob.astype(np.float64)


def _build_unet():
    """构建轻量 UNet，返回未训练的模型。"""
    import torch
    import torch.nn as nn

    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.conv(x)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = DoubleConv(1, 32)
            self.enc2 = DoubleConv(32, 64)
            self.enc3 = DoubleConv(64, 128)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = DoubleConv(128, 256)
            self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.dec3 = DoubleConv(256 + 128, 128)
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.dec2 = DoubleConv(128 + 64, 64)
            self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.dec1 = DoubleConv(64 + 32, 32)
            self.out_conv = nn.Conv2d(32, 1, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.sigmoid(self.out_conv(d1))

    return UNet()
