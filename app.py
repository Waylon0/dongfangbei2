"""断层多边形自动追踪系统 — PyQt5 桌面应用程序

功能：
- 读取二维断层属性数据（图像、矩阵、GeoEast .dat）
- 自动识别、连接、提取断层多边形
- 参数可视化调节、多阶段预览
- 支持导出 GeoJSON / CSV / TXT
- AI 助手解释参数作用
"""

import sys
import os
import random
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QScrollArea,
    QSpinBox, QDoubleSpinBox, QCheckBox, QFormLayout, QGroupBox,
    QSplitter, QStatusBar, QMessageBox, QProgressBar,
    QInputDialog, QAction, QTextEdit, QDialog, QComboBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from src.pipeline import run_pipeline
from src.preprocess import load_attribute_data
from src.synthetic import generate_synthetic_data


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
        self.ax.set_title(title, fontsize=11)
        self.fig.tight_layout()
        self.draw_idle()

    def plot_polygons(self, data: np.ndarray, polygons: list, title: str = ""):
        self.ax.clear()
        self.ax.imshow(data, cmap='gray', origin='upper', aspect='auto')
        for poly in polygons:
            arr = np.array(poly)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                self.ax.fill(arr[:, 1], arr[:, 0],
                           color='#e6194b', alpha=0.45, edgecolor='#cc0033',
                           linewidth=1.8)
        self.ax.set_title(title, fontsize=11)
        self.fig.tight_layout()
        self.draw_idle()


# --- 后台流水线线程 ---

class PipelineWorker(QThread):
    finished = pyqtSignal(object)
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
    # 去噪与增强
    'gaussian_sigma':          ('float', 0.0, 10.0, 0.1),
    'use_median_filter':       ('bool',),
    'median_filter_size':      ('int', 3, 15, 2),
    'use_gabor':               ('bool',),
    'gabor_frequency':         ('float', 0.05, 0.5, 0.01),
    'gabor_angles':            ('int', 2, 8),
    'use_clahe':               ('bool',),
    'clahe_clip_limit':        ('float', 0.5, 10.0, 0.1),
    'clahe_grid_size':         ('int', 2, 64),
    # 二值化模式
    'threshold_mode':          ('combo', ['otsu','fixed','adaptive','hysteresis']),
    'otsu_scale':              ('float', 0.1, 3.0, 0.05),
    'fixed_threshold':         ('float', 0.0, 1.0, 0.01),
    'hysteresis_low':          ('float', 0.0, 1.0, 0.01),
    'hysteresis_high':         ('float', 0.0, 1.0, 0.01),
    'adaptive_block_size':     ('int', 3, 99, 2),
    'adaptive_c':              ('float', -5.0, 5.0, 0.1),
    # UNet
    'use_unet':                ('bool',),
    # 形态学处理
    'morph_order':             ('combo', ['open_first','close_first']),
    'opening_radius':          ('int', 0, 10),
    'closing_radius':          ('int', 0, 20),
    'morph_kernel_shape':      ('combo', ['disk','ellipse','cross']),
    # 连通域与形状过滤
    'min_component_area':      ('int', 0, 500),
    'separate_intersections':  ('bool',),
    'max_aspect_ratio':        ('float', 1.0, 100.0, 1.0),
    'max_compactness':         ('float', 1.0, 50.0, 0.5),
    # 轮廓提取与矢量化
    'contour_smooth_sigma':    ('float', 0.0, 10.0, 0.5),
    'polygon_mode':            ('combo', ['skeleton','region']),
    'skeleton_buffer':         ('int', 1, 5),
    'min_polygon_area':        ('float', 0.0, 500.0, 1.0),
    'dp_mode':                 ('combo', ['absolute','relative']),
    'dp_epsilon':              ('float', 0.1, 20.0, 0.1),
    'dp_ratio':                ('float', 0.001, 0.05, 0.001),
    'smooth_iterations':       ('int', 0, 10),
    # 断层追踪连接
    'track_max_link_distance': ('float', 0.0, 200.0, 1.0),
    'track_angle_weight':      ('float', 0.0, 10.0, 0.1),
    'track_min_segment_length':('int', 0, 50),
    'track_dilate_radius':     ('int', 0, 20),
    'track_dilate_iterations': ('int', 1, 20),
}

_GROUP_NAMES = {
    '去噪与增强': ['gaussian_sigma', 'use_median_filter', 'median_filter_size',
                 'use_gabor', 'gabor_frequency', 'gabor_angles',
                 'use_clahe', 'clahe_clip_limit', 'clahe_grid_size'],
    '二值化模式': ['threshold_mode', 'otsu_scale', 'fixed_threshold',
                 'hysteresis_low', 'hysteresis_high',
                 'adaptive_block_size', 'adaptive_c', 'use_unet'],
    '形态学处理': ['morph_order', 'opening_radius', 'closing_radius', 'morph_kernel_shape'],
    '连通域与形状过滤': ['min_component_area', 'separate_intersections',
                     'max_aspect_ratio', 'max_compactness'],
    '矢量化': ['contour_smooth_sigma', 'polygon_mode', 'skeleton_buffer',
              'min_polygon_area', 'dp_mode',
              'dp_epsilon', 'dp_ratio', 'smooth_iterations'],
    '断层追踪连接': ['track_max_link_distance', 'track_angle_weight',
                  'track_min_segment_length', 'track_dilate_radius',
                  'track_dilate_iterations'],
}


class ParamPanel(QScrollArea):
    configChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cfg = Config()
        self._widgets = {}

        container = QWidget()
        layout = QVBoxLayout(container)

        for group_name, keys in _GROUP_NAMES.items():
            grp = QGroupBox(group_name)
            grp.setFont(QFont('Microsoft YaHei', 9))
            form = QFormLayout()
            for key in keys:
                w = self._add_control(form, key)
                self._widgets[key] = w
            grp.setLayout(form)
            layout.addWidget(grp)

        layout.addStretch()
        self.setWidget(container)
        self.setWidgetResizable(True)
        self.setMinimumWidth(300)

    def _add_control(self, form: QFormLayout, key: str):
        meta = _PARAM_META[key]
        default = getattr(self.cfg, key)

        if meta[0] == 'bool':
            w = QCheckBox()
            w.setChecked(default)
            w.setToolTip(self._tr(key))
            w.toggled.connect(lambda v, k=key: self._on_change(k, v))
            form.addRow(self._tr(key), w)
            return w
        elif meta[0] == 'combo':
            options = meta[1]
            w = QComboBox()
            w.addItems([self._tr_opt(o) for o in options])
            idx = options.index(default) if default in options else 0
            w.setCurrentIndex(idx)
            w.currentIndexChanged.connect(
                lambda i, k=key, opts=options: self._on_change(k, opts[i]))
            w.setToolTip(self._tr(key))
            form.addRow(self._tr(key), w)
            return w
        elif meta[0] == 'int':
            step = meta[3] if len(meta) > 3 and isinstance(meta[3], int) else 1
            w = QSpinBox()
            w.setRange(meta[1], meta[2])
            w.setSingleStep(step)
            w.setValue(int(default))
            w.setToolTip(self._tr(key))
            w.valueChanged.connect(lambda v, k=key: self._on_change(k, v))
            form.addRow(self._tr(key), w)
            return w
        elif meta[0] == 'float':
            step = meta[3] if len(meta) > 3 and isinstance(meta[3], float) else 0.1
            w = QDoubleSpinBox()
            w.setRange(meta[1], meta[2])
            w.setSingleStep(step)
            w.setDecimals(3)
            w.setValue(float(default))
            w.setToolTip(self._tr(key))
            w.valueChanged.connect(lambda v, k=key: self._on_change(k, v))
            form.addRow(self._tr(key), w)
            return w

    def _tr(self, key: str) -> str:
        """参数名中英文映射"""
        mapping = {
            'gaussian_sigma': '高斯滤波σ',
            'use_median_filter': '启用中值滤波',
            'median_filter_size': '中值滤波核大小',
            'use_gabor': '启用Gabor方向滤波',
            'gabor_frequency': 'Gabor频率',
            'gabor_angles': 'Gabor方向数',
            'use_clahe': '启用CLAHE增强',
            'clahe_clip_limit': 'CLAHE对比度限制',
            'clahe_grid_size': 'CLAHE网格大小',
            'threshold_mode': '二值化模式',
            'otsu_scale': 'Otsu阈值缩放',
            'fixed_threshold': '固定阈值',
            'hysteresis_low': '滞后分割低阈值',
            'hysteresis_high': '滞后分割高阈值',
            'use_adaptive_threshold': '自适应阈值(旧)',
            'adaptive_block_size': '自适应窗口大小',
            'adaptive_c': '自适应阈值偏移',
            'use_unet': '使用UNet深度学习分割',
            'morph_order': '形态学操作顺序',
            'opening_radius': '开运算半径',
            'closing_radius': '闭运算半径',
            'morph_kernel_shape': '结构元素形状',
            'min_component_area': '最小连通域面积',
            'separate_intersections': '分离交叉断层',
            'max_aspect_ratio': '最大长宽比',
            'max_compactness': '最大紧致度',
            'contour_smooth_sigma': '轮廓平滑σ',
            'polygon_mode': '多边形提取模式',
            'skeleton_buffer': '骨架缓冲半径(像素)',
            'min_polygon_area': '最小多边形面积',
            'dp_mode': 'DP简化模式',
            'dp_epsilon': 'DP简化容差(像素)',
            'dp_ratio': 'DP简化比例(周长×)',
            'smooth_iterations': '平滑迭代次数',
            'track_max_link_distance': '最大连接距离',
            'track_angle_weight': '方向一致性权重',
            'track_min_segment_length': '最小片段长度',
            'track_dilate_radius': '膨胀半径',
            'track_dilate_iterations': '膨胀迭代次数',
        }
        return mapping.get(key, key)

    def _tr_opt(self, opt: str) -> str:
        """下拉选项映射"""
        mapping = {
            'otsu': 'Otsu自动阈值',
            'fixed': '固定阈值',
            'adaptive': '自适应阈值',
            'hysteresis': '双阈值滞后分割',
            'open_first': '先开后闭（去噪→填充）',
            'close_first': '先闭后开（填充→去噪）',
            'disk': '圆形 (Disk)',
            'ellipse': '椭圆形 (Ellipse)',
            'cross': '十字形 (Cross)',
            'absolute': '绝对像素值',
            'relative': '周长比例',
            'skeleton': '骨架精细线条 (推荐)',
            'region': '粗区域轮廓',
        }
        return mapping.get(opt, opt)

    def _on_change(self, key, value):
        setattr(self.cfg, key, value)
        self.configChanged.emit()

    def get_config(self) -> Config:
        return self.cfg


# --- AI 助手对话框 ---

_AI_TIPS = [
    "Otsu阈值缩放 < 1 会检测更多微弱断层，但可能引入噪声；> 1 则更保守。",
    "闭运算半径越大，断开的裂缝越容易被连接，但过度可能导致不同断层粘连。",
    "开运算可以去除细小的椒盐噪声，建议值 1~3。",
    "中值滤波对椒盐噪声特别有效，比高斯滤波更好地保留边缘。",
    "双阈值滞后分割：高于高阈值的为确定断层，与确定断层连通且高于低阈值的也保留。",
    "固定阈值模式适合断层响应整体偏强的数据，参考直方图确定，典型值 0.5~0.7。",
    "Gabor方向滤波能增强特定方向的线性结构，适合弱且走向一致的断层。",
    "椭圆核在走向方向上连接更强，十字核适合正交裂缝网络。",
    "先开后闭 = 先去噪再填充，推荐默认；先闭后开 = 先连接再去除孤立噪点。",
    "DP简化比例模式按周长×比例确定epsilon，推荐 0.005（0.5%周长），自动适应多边形大小。",
    "自适应阈值适合光照/属性值不均匀的数据，比全局Otsu更灵活。",
    "方向一致性权重越大，只有走向接近的片段才会被连接，避免交叉连接。",
    "最大连接距离决定了断开多远的片段仍会被尝试连接，设置过大会导致错误连接。",
    "长宽比过滤可去除细长的非断层噪声（如采集脚印），建议 10~30。",
    "紧致度 = 周长²/(4π×面积)，越接近1越像圆形，断层通常是线状的（紧致度>1）。",
    "UNet模式需要先训练模型（合成数据预训练即可），适合传统方法效果差时使用。",
    "Shapefile 导出需要 pip install pyshp，可导入 ArcGIS / QGIS 等GIS软件。",
    "如果断层数量过多，试着增大 min_polygon_area 或 otsu_scale。",
    "如果断层连成一大片，减小 closing_radius 并增大开运算半径。",
]

_AI_POEMS = [
    "地震波穿岩千尺深，断层面藏万象痕。\n参数调来寻裂隙，一朝识别见乾坤。",
    "数据如山待剖析，断层似线要追踪。\n耐心调整三五处，累累纹理自可窥。",
    "地下迷宫谁人知？属性图中线如丝。\n算子轻扫寻断处，万里地层入画时。",
]


class AiDialog(QDialog):
    def __init__(self, parent=None, cfg: Config = None, result: dict = None):
        super().__init__(parent)
        self.setWindowTitle("AI 助手 — 参数解读与建议")
        self.resize(550, 420)

        layout = QVBoxLayout(self)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont('Microsoft YaHei', 10))
        layout.addWidget(self.text)

        btn_layout = QHBoxLayout()
        tip_btn = QPushButton("参数解读")
        tip_btn.clicked.connect(lambda: self._show_tips(cfg))
        poem_btn = QPushButton("来首地质诗")
        poem_btn.clicked.connect(self._show_poem)
        stats_btn = QPushButton("统计摘要")
        stats_btn.clicked.connect(lambda: self._show_stats(result))
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)

        btn_layout.addWidget(tip_btn)
        btn_layout.addWidget(stats_btn)
        btn_layout.addWidget(poem_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        if result:
            self._show_stats(result)
        else:
            self._show_tips(cfg)

    def _show_tips(self, cfg: Config):
        lines = ["=== 参数解读 ===\n"]
        lines.append(f"当前关键参数：")
        lines.append(f"  二值化模式 = {cfg.threshold_mode}")
        if cfg.threshold_mode == 'otsu':
            lines.append(f"    Otsu缩放 = {cfg.otsu_scale}（{'偏保守' if cfg.otsu_scale >= 1 else '偏敏感'}）")
        elif cfg.threshold_mode == 'fixed':
            lines.append(f"    固定阈值 = {cfg.fixed_threshold}")
        elif cfg.threshold_mode == 'hysteresis':
            lines.append(f"    低阈值={cfg.hysteresis_low}, 高阈值={cfg.hysteresis_high}")
        lines.append(f"  高斯滤波σ = {cfg.gaussian_sigma}")
        lines.append(f"  中值滤波 = {'启用(' + str(cfg.median_filter_size) + ')' if cfg.use_median_filter else '禁用'}")
        lines.append(f"  Gabor滤波 = {'启用(' + str(cfg.gabor_angles) + '方向)' if cfg.use_gabor else '禁用'}")
        lines.append(f"  形态学顺序 = {'先开后闭' if cfg.morph_order == 'open_first' else '先闭后开'}")
        lines.append(f"  结构元素 = {cfg.morph_kernel_shape}（闭{cfg.closing_radius}/开{cfg.opening_radius}）")
        lines.append(f"  DP简化 = {'周长×' + str(cfg.dp_ratio) if cfg.dp_mode == 'relative' else str(cfg.dp_epsilon) + '像素'}")
        lines.append(f"  最大连接距离 = {cfg.track_max_link_distance} 像素")
        lines.append(f"  UNet = {'启用' if cfg.use_unet else '禁用'}")
        lines.append(f"")
        lines.append(random.choice(_AI_TIPS))
        lines.append(random.choice(_AI_TIPS))
        lines.append(f"\n提示：鼠标悬停在参数上可查看说明。")
        self.text.setText('\n'.join(lines))

    def _show_stats(self, result: dict):
        if result is None:
            self.text.setText("请先运行流水线后再查看统计摘要。")
            return
        lines = ["=== 统计摘要 ===\n"]
        lines.append(f"检测到断层多边形数量：{result.get('count', 0)} 条")
        lines.append(f"总面积：{result.get('total_area', 0)} 像素²")
        lines.append(f"总长度（估算）：{result.get('total_length', 0)} 像素")
        lines.append(f"最小面积：{result.get('min_area', 0)} 像素²")
        lines.append(f"最大面积：{result.get('max_area', 0)} 像素²")
        lines.append(f"处理耗时：{result.get('elapsed', 0)} 秒")
        st = result.get('step_times', {})
        if st:
            lines.append("\n--- 各步骤耗时 ---")
            lines.append(f"预处理:      {st.get('preprocess', 0):.3f}s")
            lines.append(f"二值化:      {st.get('binarize', 0):.3f}s")
            lines.append(f"形态学:      {st.get('morph', 0):.3f}s")
            lines.append(f"断层追踪:    {st.get('track', 0):.3f}s")
            lines.append(f"轮廓提取:    {st.get('contour_extract', 0):.3f}s")
            lines.append(f"矢量化:      {st.get('vectorize', 0):.3f}s")
            if st.get('multiscale', 0) > 0:
                lines.append(f"多尺度融合:  {st.get('multiscale', 0):.3f}s")
        if result.get('areas'):
            areas = sorted(result['areas'], reverse=True)
            lines.append(f"\n前5大面积：{areas[:5]}")
            if len(areas) >= 3:
                lines.append(f"面积中位数：{areas[len(areas)//2]:.1f}")
        lines.append(f"\n{random.choice(_AI_TIPS)}")
        self.text.setText('\n'.join(lines))

    def _show_poem(self):
        self.text.setText(random.choice(_AI_POEMS))


# --- 主窗口 ---

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("断层多边形自动追踪系统 v2.0")
        self.resize(1280, 800)
        self._data = None
        self._result = None
        self._worker = None

        self._setup_menu()
        self._setup_ui()
        self._setup_statusbar()

    def _setup_menu(self):
        menu = self.menuBar()

        file_menu = menu.addMenu("文件(&F)")
        act_open = QAction("打开数据文件(&O)...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._on_open)
        file_menu.addAction(act_open)

        act_synthetic = QAction("生成合成测试数据(&G)...", self)
        act_synthetic.setShortcut("Ctrl+G")
        act_synthetic.triggered.connect(self._on_generate)
        file_menu.addAction(act_synthetic)

        file_menu.addSeparator()

        export_menu = file_menu.addMenu("导出结果")

        act_export_geojson = QAction("导出 GeoJSON...", self)
        act_export_geojson.setShortcut("Ctrl+E")
        act_export_geojson.triggered.connect(lambda: self._on_export('geojson'))
        export_menu.addAction(act_export_geojson)

        act_export_csv = QAction("导出 CSV 统计表...", self)
        act_export_csv.triggered.connect(lambda: self._on_export('csv'))
        export_menu.addAction(act_export_csv)

        act_export_txt = QAction("导出 TXT 坐标...", self)
        act_export_txt.triggered.connect(lambda: self._on_export('txt'))
        export_menu.addAction(act_export_txt)

        act_export_shp = QAction("导出 Shapefile...", self)
        act_export_shp.triggered.connect(lambda: self._on_export('shp'))
        export_menu.addAction(act_export_shp)

        file_menu.addSeparator()

        act_exit = QAction("退出(&X)", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        help_menu = menu.addMenu("帮助(&H)")

        act_unet = QAction("加载UNet模型...", self)
        act_unet.triggered.connect(self._on_load_unet)
        help_menu.addAction(act_unet)

        help_menu.addSeparator()

        act_ai = QAction("AI 助手(&A)...", self)
        act_ai.setShortcut("F1")
        act_ai.triggered.connect(self._on_ai_assistant)
        help_menu.addAction(act_ai)

        act_about = QAction("关于(&B)...", self)
        act_about.triggered.connect(self._on_about)
        help_menu.addAction(act_about)

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
        self.tabs.setFont(QFont('Microsoft YaHei', 9))
        self._canvases = {}
        stage_names = [
            ('raw',       "原始数据"),
            ('smoothed',  "平滑/滤波后"),
            ('binary_morph', "二值化(形态学前)"),
            ('binary',    "二值化(最终)"),
            ('skeleton',  "骨架/交叉点"),
            ('polygons',  "断层多边形"),
        ]
        for key, label in stage_names:
            canvas = MplCanvas()
            self.tabs.addTab(canvas, label)
            self._canvases[key] = canvas

        right_layout.addWidget(self.tabs)

        # 裁切区域
        crop_grp = QGroupBox("裁切区域（加载大数据后勾选使用）")
        crop_layout = QHBoxLayout()
        crop_layout.addWidget(QLabel("行:"))
        self._crop_r1 = QSpinBox()
        self._crop_r1.setRange(0, 2000)
        self._crop_r1.setValue(500)
        crop_layout.addWidget(self._crop_r1)
        crop_layout.addWidget(QLabel("–"))
        self._crop_r2 = QSpinBox()
        self._crop_r2.setRange(0, 2000)
        self._crop_r2.setValue(900)
        crop_layout.addWidget(self._crop_r2)
        crop_layout.addWidget(QLabel("  列:"))
        self._crop_c1 = QSpinBox()
        self._crop_c1.setRange(0, 2000)
        self._crop_c1.setValue(150)
        crop_layout.addWidget(self._crop_c1)
        crop_layout.addWidget(QLabel("–"))
        self._crop_c2 = QSpinBox()
        self._crop_c2.setRange(0, 2000)
        self._crop_c2.setValue(550)
        crop_layout.addWidget(self._crop_c2)
        self._crop_enabled = QCheckBox("启用裁切")
        self._crop_enabled.setChecked(False)
        crop_layout.addWidget(self._crop_enabled)
        crop_layout.addStretch()
        crop_grp.setLayout(crop_layout)
        right_layout.addWidget(crop_grp)

        # 底部操作栏
        btn_layout = QHBoxLayout()

        self.run_btn = QPushButton("▶  运行断层追踪流水线")
        self.run_btn.setMinimumHeight(38)
        self.run_btn.setFont(QFont('Microsoft YaHei', 10, QFont.Bold))
        self.run_btn.setStyleSheet(
            "QPushButton { background-color: #0078d4; color: white; border-radius: 6px; padding: 6px 20px; }"
            "QPushButton:hover { background-color: #106ebe; }"
            "QPushButton:disabled { background-color: #ccc; color: #888; }"
        )
        self.run_btn.clicked.connect(self._on_run)
        self.run_btn.setEnabled(False)

        self.ai_btn = QPushButton("🤖 AI 助手")
        self.ai_btn.setMinimumHeight(38)
        self.ai_btn.setFont(QFont('Microsoft YaHei', 9))
        self.ai_btn.setToolTip("查看参数解读、统计摘要或获取地质灵感")
        self.ai_btn.clicked.connect(self._on_ai_assistant)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setMaximumWidth(200)
        self._progress.hide()

        btn_layout.addStretch()
        btn_layout.addWidget(self.ai_btn)
        btn_layout.addSpacing(10)
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self._progress)
        btn_layout.addStretch()
        right_layout.addLayout(btn_layout)

        splitter.addWidget(right)
        splitter.setSizes([320, 960])

        main_layout = QHBoxLayout(central)
        main_layout.addWidget(splitter)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self._status_label = QLabel("就绪 — 默认使用「骨架精细线条」模式，可在矢量化面板切换为「粗区域轮廓」")
        self._status_label.setFont(QFont('Microsoft YaHei', 9))
        self.statusbar.addWidget(self._status_label)

    # --- 槽 ---

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开断层属性数据", "",
            "属性文件 (*.dat *.npy *.npz *.png *.tiff *.tif *.bmp *.jpg *.jpeg *.txt);;"
            "GeoEast网格 (*.dat);;"
            "图像文件 (*.png *.tiff *.tif *.bmp *.jpg *.jpeg);;"
            "NumPy文件 (*.npy *.npz);;"
            "文本矩阵 (*.txt);;"
            "所有文件 (*)")
        if not path:
            return
        try:
            self._data = load_attribute_data(path)
            if self._data.ndim != 2:
                QMessageBox.critical(self, "格式错误",
                    f"数据维度必须为2维，当前为 {self._data.ndim} 维，形状 {self._data.shape}")
                return
            h, w = self._data.shape
            # 自动设裁切范围为图片中心 400x400
            r1, r2 = max(0, h//2-200), min(h, h//2+200)
            c1, c2 = max(0, w//2-200), min(w, w//2+200)
            self._crop_r1.setRange(0, h-1); self._crop_r1.setValue(r1)
            self._crop_r2.setRange(0, h);   self._crop_r2.setValue(r2)
            self._crop_c1.setRange(0, w-1); self._crop_c1.setValue(c1)
            self._crop_c2.setRange(0, w);   self._crop_c2.setValue(c2)
            self._crop_enabled.setChecked(False)
            self._show_raw()
            self.run_btn.setEnabled(True)
            size_mb = self._data.nbytes / (1024 * 1024)
            self._status_label.setText(
                f"已加载: {os.path.basename(path)}  |  "
                f"尺寸: {w}×{h}  |  "
                f"大小: {size_mb:.1f} MB  |  "
                f"值域: [{self._data.min():.3f}, {self._data.max():.3f}]")
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"无法读取文件：\n{str(e)}")

    def _on_generate(self):
        rows, ok = QInputDialog.getInt(self, "合成数据", "图像行数（高度）:", 300, 50, 2000, 50)
        if not ok:
            return
        cols, ok = QInputDialog.getInt(self, "合成数据", "图像列数（宽度）:", 400, 50, 2000, 50)
        if not ok:
            return
        n_faults, ok = QInputDialog.getInt(self, "合成数据", "断层数量:", 5, 1, 30)
        if not ok:
            return
        noise, ok = QInputDialog.getDouble(self, "合成数据", "噪声水平:", 0.03, 0, 0.5, 3)
        if not ok:
            return

        self._data = generate_synthetic_data(
            shape=(rows, cols), n_faults=n_faults, noise_level=noise)
        self._crop_enabled.setChecked(False)
        self._show_raw()
        self.run_btn.setEnabled(True)
        self._status_label.setText(
            f"合成数据: {cols}×{rows}  |  {n_faults} 条模拟断层  |  噪声 {noise:.3f}")

    def _show_raw(self):
        self._canvases['raw'].imshow(self._data, "原始属性数据", cmap='viridis')
        for key in ['smoothed', 'binary_morph', 'binary', 'skeleton', 'polygons']:
            self._canvases[key].clear()

    def _on_run(self):
        if self._data is None:
            return
        self._result = None
        self.run_btn.setEnabled(False)
        self.run_btn.setText("⏳  处理中...")
        self._progress.show()

        # 裁切
        if self._crop_enabled.isChecked():
            r1, r2 = self._crop_r1.value(), self._crop_r2.value()
            c1, c2 = self._crop_c1.value(), self._crop_c2.value()
            self._crop_data = self._data[r1:r2, c1:c2].copy()
        else:
            self._crop_data = self._data.copy()

        cfg = self.param_panel.get_config()
        self._worker = PipelineWorker(self._crop_data, cfg)
        self._worker.finished.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_result(self, result: dict):
        self._result = result
        self._progress.hide()
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶  运行断层追踪流水线")

        display_data = getattr(self, '_crop_data', self._data)

        self._canvases['raw'].imshow(display_data, "原始数据", cmap='viridis')
        self._canvases['smoothed'].imshow(
            result['data_smoothed'], "平滑/滤波后数据")
        self._canvases['binary_morph'].imshow(
            result['binary_before_morph'], "二值化（形态学处理前）", cmap='gray')
        self._canvases['binary'].imshow(
            result['binary'], "二值化（形态学处理后 + 断层连接）", cmap='gray')
        self._canvases['skeleton'].imshow(
            result['skeleton'], f"骨架 — {len(result['junctions'])} 个交叉点", cmap='gray')
        self._canvases['polygons'].plot_polygons(
            display_data,
            result['filtered'],
            f"断层多边形 — {result['count']} 条, 耗时 {result['elapsed']:.3f}s")

        # 控制台打印每步耗时
        st = result.get('step_times', {})
        if st:
            print(f"\n=== 流水线耗时分解 ===")
            print(f"  预处理:      {st.get('preprocess', 0):.3f}s")
            print(f"  二值化:      {st.get('binarize', 0):.3f}s")
            print(f"  形态学:      {st.get('morph', 0):.3f}s")
            print(f"  断层追踪:    {st.get('track', 0):.3f}s")
            print(f"  轮廓提取:    {st.get('contour_extract', 0):.3f}s")
            print(f"  矢量化:      {st.get('vectorize', 0):.3f}s")
            if st.get('multiscale', 0) > 0:
                print(f"  多尺度融合:  {st.get('multiscale', 0):.3f}s")
            print(f"  总耗时:      {result['elapsed']:.3f}s")
            print("=" * 30)

        self._status_label.setText(
            f"完成  |  "
            f"断层 {result['count']} 条  |  "
            f"耗时 {result['elapsed']:.3f}s  |  "
            f"面积: {result['min_area']:.0f}~{result['max_area']:.0f} 像素²  |  "
            f"总长度(估): {result['total_length']:.0f} 像素  |  "
            f"可导出 GeoJSON / CSV / TXT")

    def _on_error(self, msg: str):
        self._progress.hide()
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶  运行断层追踪流水线")
        QMessageBox.critical(self, "处理错误", f"流水线运行失败：\n{msg}")
        self._status_label.setText("处理出错，请检查参数设置或数据格式")

    def _on_export(self, fmt: str):
        if self._result is None or not self._result['filtered']:
            QMessageBox.information(self, "提示", "没有可导出的多边形，请先运行流水线。")
            return

        polygons = self._result['filtered']
        areas = self._result['areas']

        if fmt == 'geojson':
            path, _ = QFileDialog.getSaveFileName(
                self, "导出断层多边形 GeoJSON", "fault_polygons.geojson",
                "GeoJSON (*.geojson);;所有文件 (*)")
            if not path:
                return
            from src.vectorize import export_geojson
            export_geojson(polygons, path, areas)
            self._status_label.setText(f"已导出 {len(polygons)} 条多边形 → {os.path.basename(path)}")

        elif fmt == 'csv':
            path, _ = QFileDialog.getSaveFileName(
                self, "导出统计表 CSV", "fault_statistics.csv",
                "CSV (*.csv);;所有文件 (*)")
            if not path:
                return
            from src.vectorize import export_csv
            export_csv(polygons, path, areas)
            self._status_label.setText(f"已导出统计表 → {os.path.basename(path)}")

        elif fmt == 'txt':
            path, _ = QFileDialog.getSaveFileName(
                self, "导出断层多边形 TXT", "fault_polygons.txt",
                "文本 (*.txt);;所有文件 (*)")
            if not path:
                return
            from src.vectorize import export_polygons_txt
            export_polygons_txt(polygons, path)
            self._status_label.setText(f"已导出 {len(polygons)} 条多边形 → {os.path.basename(path)}")

        elif fmt == 'shp':
            path, _ = QFileDialog.getSaveFileName(
                self, "导出断层多边形 Shapefile", "fault_polygons.shp",
                "Shapefile (*.shp);;所有文件 (*)")
            if not path:
                return
            try:
                from src.vectorize import export_shapefile
                export_shapefile(polygons, path, areas)
                self._status_label.setText(f"已导出 Shapefile → {os.path.basename(path)}")
            except ImportError as e:
                QMessageBox.warning(self, "缺少依赖", str(e))
            except Exception as e:
                QMessageBox.warning(self, "导出失败", str(e))

    def _on_load_unet(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "加载UNet模型文件", "",
            "PyTorch模型 (*.pt *.pth);;所有文件 (*)")
        if not path:
            return
        cfg = self.param_panel.get_config()
        cfg.unet_model_path = path
        cfg.use_unet = True
        self.param_panel._widgets['use_unet'].setChecked(True)
        self._status_label.setText(f"UNet模型已加载: {os.path.basename(path)}")

    def _on_ai_assistant(self):
        cfg = self.param_panel.get_config()
        dlg = AiDialog(self, cfg, self._result)
        dlg.exec_()

    def _on_about(self):
        QMessageBox.about(self, "关于 — 断层多边形自动追踪系统",
            "<h3>断层多边形自动追踪系统 v2.1</h3>"
            "<p>基于传统图像处理与几何分析的断层多边形自动识别与追踪工具。</p>"
            "<p><b>核心功能：</b></p>"
            "<ul>"
            "<li>多格式数据读取（.dat / .npy / .png / .tiff / .txt）</li>"
            "<li>四种二值化模式：Otsu / 固定 / 自适应 / 双阈值滞后分割</li>"
            "<li>可选 Gabor 方向性滤波增强弱断层</li>"
            "<li>三种形态学核形状：圆形 / 椭圆 / 十字</li>"
            "<li>方向引导的断层片段智能连接</li>"
            "<li>断层多边形提取 + DP简化(支持周长比例) + 统计</li>"
            "<li>GeoJSON / CSV / TXT / Shapefile 多格式导出</li>"
            "<li>可选 UNet 深度学习分割</li>"
            "</ul>"
            "<p><b>技术栈：</b>NumPy + SciPy + scikit-image + PyQt5 + Matplotlib</p>"
            "<p><b>适用平台：</b>Windows / Linux (GeoEast 虚拟机)</p>"
        )


# --- 入口 ---

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont('Microsoft YaHei', 9))

    # 全局样式
    app.setStyleSheet("""
        QMainWindow { background-color: #f5f5f5; }
        QGroupBox { font-weight: bold; border: 1px solid #ddd; border-radius: 4px; margin-top: 8px; padding-top: 12px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        QTabWidget::pane { border: 1px solid #ccc; }
        QPushButton { padding: 4px 12px; }
        QStatusBar { background-color: #e8e8e8; }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
