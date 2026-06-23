# 断层多边形自动追踪 — GeoEast 集成模块

基于官方 QProcess_Run_Python 框架的 **前后端分离** 标准模块。

## 架构

```
geoeast_module/
├── pyalgo_wzy/                    # Python 后端（算法核心）
│   ├── entry.py                   # 标准化入口 run() 函数
│   └── __init__.py
├── QProcess_Run_Python/           # C++ 前端（Qt 主窗口）
│   ├── main.cpp                   # 程序入口
│   ├── Application.{h,cpp}        # GeoEast 框架通讯
│   ├── DataManager.{h,cpp}        # BO 数据库管理
│   ├── LoginDialog.{h,cpp,ui}     # 数据库登录
│   ├── MainWindow.{h,cpp,ui}      # 主界面 + QProcess 调用
│   └── QProcess_Run_Python.pro    # Qt 工程文件
└── README.md
```

## 数据流

```
C++ 前端 (MainWindow)                Python 后端 (entry.py)
─────────────────────                ─────────────────────
UI 参数输入                           │
  │                                   │
  ├─ QProcess::start(python, args) ─→│ 接收参数
  │                                   ├─ PyBO 读取 GeoEast 网格
  │                                   ├─ 运行断层追踪算法
  │                                   ├─ PyBO 写回断层多边形
  │                                   ├─ print() 输出进度
  ├─ slot_readyReadStdOutput()  ←────┤
  │                                   │
  ├─ UI 显示结果                      │
  │                                   │
  ├─ write("exit\n") ───────────────→│ input_listener 接收退出
```

## Python 后端接口

```python
from entry import run

run(
    project_name="DFB_S7_WZY",    # GeoEast 项目名
    survey_name="survey1",         # 工区名
    grid_name="fault_attr",        # 输入沿层属性网格
    output_name="fault_polygons",  # 输出断层多边形对象
    threshold_mode="otsu",         # 二值化模式
    otsu_scale=0.8,               # Otsu 缩放
    gaussian_sigma=1.5,           # 高斯滤波 σ
    closing_radius=2,             # 闭运算半径
    opening_radius=1,             # 开运算半径
    min_polygon_area=15.0,        # 最小面积
    dp_epsilon=1.0,               # DP 简化容差
    polygon_mode="skeleton",      # 多边形模式
)
```

## 编译 C++ 前端（需 Windows 环境）

```bash
# 1. 设置环境
set NGP=D:\GeoEastRC\GeoEast-RC-V2.2
set GEOEAST=D:\GeoEastRC\iEcoV2.1

# 2. 用 Qt Creator 或命令行编译
cd geoeast_module\QProcess_Run_Python
qmake QProcess_Run_Python.pro
nmake   # MSVC
# 或
mingw32-make  # MinGW
```

## 部署

将编译后的 `FaultPolygonTracker.exe` 和 `pyalgo_wzy/` 目录
放置到 GeoEast 模块目录下，由 GeoEast 主控加载。
