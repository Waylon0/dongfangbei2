"""PyQt6 断层追踪调试工具

双击运行，提供：
- 参数可视化调节
- 中间步骤查看（概率图 / 二值图 / 骨架 / 多边形）
- 一键运行完整流水线

依赖：PyQt6, matplotlib
"""

import sys
import os
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QScrollArea,
    QSpinBox, QDoubleSpinBox, QCheckBox, QFormLayout, QGroupBox,
    QSplitter, QStatusBar, QMessageBox, QProgressBar,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from src.pipeline import run_pipeline
from src.preprocess import load_attribute_data
from src.synthetic import generate_synthetic_data
from src.visualize import plot_overlay


# --- matplotlib 画布 ---

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 5), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()

    def clear(self):
        self.ax.clear()
        self.fig.tight_layout()
        self.draw_idle()

    def imshow(self, data: np.ndarray, title: str = "", cmap: str = "gray"):
        self.ax.clear()
        self.ax.imshow(data, cmap=cmap, origin='upper', aspect='auto')
        self.ax.set_title(title)
        self.fig.tight_layout()
        self.draw_idle()

    def plot_polygons(self, data: np.ndarray, polygons: list, title: str = ""):
        self.ax.clear()
        self.ax.imshow(data, cmap='gray', origin='upper', aspect='auto')
        for poly in polygons:
            arr = np.array(poly)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                self.ax.plot(arr[:, 1], arr[:, 0], 'r-', linewidth=1.5)
        self.ax.set_title(title)
        self.fig.tight_layout()
        self.draw_idle()


# --- 后台流水线线程 ---

class PipelineWorker(QThread):
    finished = pyqtSignal(object)  # dict result
    error = pyqtSignal(str)

    def __init__(self, data: np.ndarray, cfg: Config):
        super().__init__()
        self.data = data
        self.cfg = cfg

    def run(self):
        try:
            result = run_pipeline(self.data, self.cfg)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# --- 参数面板 ---

_PARAM_META = {
    'gaussian_sigma':          ('float', 0.0, 10.0, 0.1),
    'use_clahe':               ('bool',),
    'clahe_clip_limit':        ('float', 0.5, 10.0, 0.1),
    'clahe_grid_size':         ('int', 2, 64),
    'otsu_scale':              ('float', 0.1, 3.0, 0.05),
    'use_adaptive_threshold':  ('bool',),
    'adaptive_block_size':     ('int', 3, 99, 2),
    'adaptive_c':              ('float', -5.0, 5.0, 0.1),
    'closing_radius':          ('int', 0, 20),
    'opening_radius':          ('int', 0, 10),
    'min_component_area':      ('int', 0, 500),
    'separate_intersections':  ('bool',),
    'contour_smooth_sigma':    ('float', 0.0, 10.0, 0.5),
    'min_polygon_area':        ('float', 0.0, 500.0, 1.0),
    'dp_epsilon':              ('float', 0.1, 20.0, 0.1),
    'smooth_iterations':       ('int', 0, 10),
    'track_max_link_distance': ('float', 0.0, 200.0, 1.0),
    'track_angle_weight':      ('float', 0.0, 10.0, 0.1),
    'track_min_segment_length':('int', 0, 50),
    'track_dilate_radius':     ('int', 0, 20),
    'track_dilate_iterations': ('int', 1, 20),
}


class ParamPanel(QScrollArea):
    configChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cfg = Config()
        self._widgets = {}

        container = QWidget()
        layout = QVBoxLayout(container)

        # 分组：预处理
        pre_grp = QGroupBox("预处理")
        pre_form = QFormLayout()
        for key in ['gaussian_sigma', 'use_clahe', 'clahe_clip_limit',
                     'clahe_grid_size', 'otsu_scale', 'use_adaptive_threshold',
                     'adaptive_block_size', 'adaptive_c']:
            w = self._add_control(pre_form, key)
            self._widgets[key] = w
        pre_grp.setLayout(pre_form)
        layout.addWidget(pre_grp)

        # 分组：形态学
        morph_grp = QGroupBox("形态学")
        morph_form = QFormLayout()
        for key in ['closing_radius', 'opening_radius', 'min_component_area',
                     'separate_intersections']:
            w = self._add_control(morph_form, key)
            self._widgets[key] = w
        morph_grp.setLayout(morph_form)
        layout.addWidget(morph_grp)

        # 分组：矢量化
        vec_grp = QGroupBox("矢量化")
        vec_form = QFormLayout()
        for key in ['contour_smooth_sigma', 'min_polygon_area', 'dp_epsilon',
                     'smooth_iterations']:
            w = self._add_control(vec_form, key)
            self._widgets[key] = w
        vec_grp.setLayout(vec_form)
        layout.addWidget(vec_grp)

        # 分组：断层追踪
        track_grp = QGroupBox("断层追踪")
        track_form = QFormLayout()
        for key in ['track_max_link_distance', 'track_angle_weight',
                     'track_min_segment_length', 'track_dilate_radius',
                     'track_dilate_iterations']:
            w = self._add_control(track_form, key)
            self._widgets[key] = w
        track_grp.setLayout(track_form)
        layout.addWidget(track_grp)

        layout.addStretch()
        self.setWidget(container)
        self.setWidgetResizable(True)
        self.setMinimumWidth(280)

    def _add_control(self, form: QFormLayout, key: str):
        meta = _PARAM_META[key]
        default = getattr(self.cfg, key)

        if meta[0] == 'bool':
            w = QCheckBox()
            w.setChecked(default)
            w.toggled.connect(lambda v, k=key: self._on_change(k, v))
            form.addRow(key, w)
            return w
        elif meta[0] == 'int':
            step = meta[3] if len(meta) > 3 else 1
            w = QSpinBox()
            w.setRange(meta[1], meta[2])
            w.setSingleStep(step)
            w.setValue(int(default))
            w.valueChanged.connect(lambda v, k=key: self._on_change(k, v))
            form.addRow(key, w)
            return w
        elif meta[0] == 'float':
            step = meta[3] if len(meta) > 3 else 0.1
            w = QDoubleSpinBox()
            w.setRange(meta[1], meta[2])
            w.setSingleStep(step)
            w.setDecimals(3)
            w.setValue(float(default))
            w.valueChanged.connect(lambda v, k=key: self._on_change(k, v))
            form.addRow(key, w)
            return w

    def _on_change(self, key, value):
        setattr(self.cfg, key, value)
        self.configChanged.emit()

    def get_config(self) -> Config:
        return self.cfg


# --- 主窗口 ---

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("断层多边形追踪 — 调试工具")
        self.resize(1200, 750)
        self._data = None
        self._result = None
        self._worker = None

        self._setup_menu()
        self._setup_ui()
        self._setup_statusbar()

    def _setup_menu(self):
        menu = self.menuBar()

        file_menu = menu.addMenu("文件(&F)")
        act_open = QAction("打开数据(&O)...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._on_open)
        file_menu.addAction(act_open)

        act_synthetic = QAction("生成合成数据(&G)...", self)
        act_synthetic.setShortcut("Ctrl+G")
        act_synthetic.triggered.connect(self._on_generate)
        file_menu.addAction(act_synthetic)

        file_menu.addSeparator()

        act_export = QAction("导出多边形(&E)...", self)
        act_export.setShortcut("Ctrl+E")
        act_export.triggered.connect(self._on_export)
        file_menu.addAction(act_export)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧参数面板
        self.param_panel = ParamPanel()
        splitter.addWidget(self.param_panel)

        # 右侧结果显示
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self._canvases = {}
        stage_names = [
            ('raw',       "原始数据"),
            ('smoothed',  "平滑后"),
            ('binary_morph', "二值化(形态学前)"),
            ('binary',    "二值化"),
            ('skeleton',  "骨架"),
            ('polygons',  "多边形"),
        ]
        for key, label in stage_names:
            canvas = MplCanvas()
            self.tabs.addTab(canvas, label)
            self._canvases[key] = canvas

        right_layout.addWidget(self.tabs)

        # 底部运行按钮
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("运行流水线")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.clicked.connect(self._on_run)
        self.run_btn.setEnabled(False)
        btn_layout.addStretch()
        btn_layout.addWidget(self.run_btn)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setMaximumWidth(200)
        self._progress.hide()
        btn_layout.addWidget(self._progress)
        btn_layout.addStretch()
        right_layout.addLayout(btn_layout)

        splitter.addWidget(right)
        splitter.setSizes([300, 900])

        main_layout = QHBoxLayout(central)
        main_layout.addWidget(splitter)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self._status_label = QLabel("就绪 — Ctrl+O 打开数据 或 Ctrl+G 生成合成数据")
        self.statusbar.addWidget(self._status_label)

    # --- 槽 ---

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开属性数据", "",
            "属性文件 (*.dat *.npy *.npz);;所有文件 (*)")
        if not path:
            return
        try:
            self._data = load_attribute_data(path)
            self._show_raw()
            self.run_btn.setEnabled(True)
            self._status_label.setText(
                f"已加载: {os.path.basename(path)}  {self._data.shape[0]}×{self._data.shape[1]}")
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    def _on_generate(self):
        from PyQt6.QtWidgets import QInputDialog
        rows, ok = QInputDialog.getInt(self, "合成数据", "行数:", 300, 50, 2000, 50)
        if not ok:
            return
        cols, ok = QInputDialog.getInt(self, "合成数据", "列数:", 400, 50, 2000, 50)
        if not ok:
            return
        n_faults, ok = QInputDialog.getInt(self, "合成数据", "断层数量:", 5, 1, 20)
        if not ok:
            return

        self._data = generate_synthetic_data(
            shape=(rows, cols), n_faults=n_faults, noise_level=0.03)
        self._show_raw()
        self.run_btn.setEnabled(True)
        self._status_label.setText(
            f"合成数据: {rows}×{cols}, {n_faults} 条断层")

    def _show_raw(self):
        self._canvases['raw'].imshow(self._data, "原始属性数据", cmap='viridis')
        for key in ['smoothed', 'binary_morph', 'binary', 'skeleton', 'polygons']:
            self._canvases[key].clear()

    def _on_run(self):
        if self._data is None:
            return
        self._result = None
        self.run_btn.setEnabled(False)
        self._progress.show()

        cfg = self.param_panel.get_config()
        self._worker = PipelineWorker(self._data.copy(), cfg)
        self._worker.finished.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_result(self, result: dict):
        self._result = result
        self._progress.hide()
        self.run_btn.setEnabled(True)

        self._canvases['smoothed'].imshow(
            result['data_smoothed'], "平滑后")
        self._canvases['binary_morph'].imshow(
            result['binary_before_morph'], "二值化(形态学前)", cmap='gray')
        self._canvases['binary'].imshow(
            result['binary'], "二值化", cmap='gray')
        self._canvases['skeleton'].imshow(
            result['skeleton'], f"骨架 ({len(result['junctions'])} 个交叉点)", cmap='gray')
        self._canvases['polygons'].plot_polygons(
            self._data,
            result['filtered'],
            f"断层多边形 ({len(result['filtered'])} 条, {result['elapsed']:.3f}s)")

        self._status_label.setText(
            f"完成 — {len(result['filtered'])} 条多边形, 耗时 {result['elapsed']:.3f}s, "
            f"面积范围: {min(result['areas']) if result['areas'] else 0:.0f}–"
            f"{max(result['areas']) if result['areas'] else 0:.0f} px²")

    def _on_error(self, msg: str):
        self._progress.hide()
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "流水线错误", msg)
        self._status_label.setText("错误, 请查看详情")

    def _on_export(self):
        if self._result is None or not self._result['filtered']:
            QMessageBox.information(self, "导出", "没有可导出的多边形，请先运行流水线。")
            return

        import json
        path, _ = QFileDialog.getSaveFileName(
            self, "导出断层多边形", "fault_polygons.geojson",
            "GeoJSON (*.geojson);;文本 (*.txt);;所有文件 (*)")
        if not path:
            return

        polygons = self._result['filtered']
        areas = self._result['areas']

        if path.endswith('.geojson'):
            features = []
            for i, poly in enumerate(polygons):
                arr = np.array(poly)
                coords = [[float(arr[j, 1]), float(arr[j, 0])] for j in range(len(arr))]
                if len(coords) >= 2:
                    coords.append(coords[0])
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                    "properties": {"id": i, "area": areas[i] if i < len(areas) else 0},
                })
            geojson = {"type": "FeatureCollection", "features": features}
            with open(path, 'w') as f:
                json.dump(geojson, f, indent=2)
        else:
            with open(path, 'w') as f:
                for i, poly in enumerate(polygons):
                    f.write(f"# Polygon {i}\n")
                    for pt in poly:
                        f.write(f"{pt[0]:.3f} {pt[1]:.3f}\n")
                    f.write("\n")

        self._status_label.setText(f"已导出 {len(polygons)} 条多边形到 {os.path.basename(path)}")


# --- 入口 ---

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
