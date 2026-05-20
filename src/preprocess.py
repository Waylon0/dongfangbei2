"""数据加载与预处理

支持两种分割路径：
1. UNet（推荐）：合成数据预训练，输出概率图，边界更鲁棒
2. Otsu / 自适应阈值（fallback）：不需要模型文件
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import exposure
from .utils import normalize

# --- 文件加载（不变） ---


def _load_dat(filepath: str) -> np.ndarray:
    """解析 GeoEast 导出的 ASCII .dat 格式属性网格文件。

    格式：空格分隔，首行为 # 注释头，后续每行：
    Line  CMP  X  Y  Value
    根据 Line 和 CMP 的唯一个数构建规则网格。
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    rows_data = []
    for line in lines:
        if line.startswith('#'):
            continue
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        rows_data.append((int(parts[0]), int(parts[1]), float(parts[4])))

    lines_vals = sorted(set(r[0] for r in rows_data))
    cmps_vals = sorted(set(r[1] for r in rows_data))
    line_to_idx = {v: i for i, v in enumerate(lines_vals)}
    cmp_to_idx = {v: i for i, v in enumerate(cmps_vals)}

    data = np.zeros((len(lines_vals), len(cmps_vals)), dtype=np.float64)
    for line, cmp, val in rows_data:
        data[line_to_idx[line], cmp_to_idx[cmp]] = val

    return data


def load_attribute_data(filepath: str) -> np.ndarray:
    """加载沿层属性数据。支持 .npy / .npz / .dat 格式。"""
    if filepath.endswith('.dat'):
        return _load_dat(filepath)
    if filepath.endswith('.npz'):
        data = np.load(filepath)
        key = list(data.keys())[0]
        return data[key].astype(np.float64)
    else:
        return np.load(filepath).astype(np.float64)


# --- 经典预处理流水线 ---


def preprocess(data: np.ndarray, sigma: float = 1.0,
               use_clahe: bool = False,
               clahe_clip_limit: float = 2.0,
               clahe_grid_size: int = 8,
               otsu_scale: float = 1.0) -> np.ndarray:
    """经典预处理：归一化 → CLAHE(可选) → 高斯滤波 → Otsu 二值化。

    返回二值图像 (0/1)。
    """
    data = normalize(data)

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


# --- UNet 分割路径 ---


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
