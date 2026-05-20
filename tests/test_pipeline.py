"""流水线集成测试 — 使用合成数据验证端到端流程"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.synthetic import generate_synthetic_data
from src.pipeline import run_pipeline


def test_pipeline_runs():
    """验证流水线能从合成数据中提取断层多边形"""
    cfg = Config()
    data = generate_synthetic_data(shape=(200, 300), n_faults=5, noise_level=0.03, seed=42)

    result = run_pipeline(data, cfg)

    # 每步都有产出
    assert isinstance(result['data_smoothed'], np.ndarray)
    assert result['data_smoothed'].shape == data.shape

    assert isinstance(result['binary'], np.ndarray)
    assert result['binary'].shape == data.shape
    assert set(np.unique(result['binary'])).issubset({0, 1})

    assert isinstance(result['skeleton'], np.ndarray)
    assert isinstance(result['junctions'], list)

    # 至少提取到一些多边形（合成数据有 5 条断层）
    assert len(result['filtered']) > 0, "合成数据应至少提取到 1 条断层多边形"
    assert len(result['areas']) == len(result['filtered'])
    assert all(a > 0 for a in result['areas'])

    assert result['elapsed'] > 0


def test_pipeline_single_scale():
    """单尺度模式不报错"""
    cfg = Config()
    cfg.scales = []  # 单尺度
    data = generate_synthetic_data(shape=(150, 200), n_faults=3, seed=99)

    result = run_pipeline(data, cfg)
    assert len(result['filtered']) > 0


def test_pipeline_adaptive_threshold():
    """自适应阈值模式不报错"""
    cfg = Config()
    cfg.use_adaptive_threshold = True
    data = generate_synthetic_data(shape=(150, 200), n_faults=3, seed=77)

    result = run_pipeline(data, cfg)
    assert len(result['filtered']) > 0


if __name__ == "__main__":
    test_pipeline_runs()
    test_pipeline_single_scale()
    test_pipeline_adaptive_threshold()
    print("所有测试通过")
