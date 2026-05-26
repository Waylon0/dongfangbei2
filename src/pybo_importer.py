"""PyBO 导入器 — 通过 NDP 框架连接 GeoEast 数据库。

前置条件:
    - Python 3.9 (使用 D:\GeoEastRC\support\miniconda3\envs\nv\python.exe)
    - PostgreSQL 数据库运行在 localhost:5555
    - 系统 PATH 包含 GeoEast DLL 目录 (libso/ndp, libso/ndp/plugins 等)

环境变量:
    NGP      → GeoEast-RC-V2.2 安装目录
    GEOEAST  → iEco V2.1 安装目录
    PG_HOME  → network/hostname.pg 所在目录
"""

import os
import sys
import ctypes


def _setup():
    ngp = os.environ.get('NGP', r'D:\GeoEastRC\GeoEast-RC-V2.2')
    geoeast = os.environ.get('GEOEAST', r'D:\GeoEastRC\iEcoV2.1')

    if 'PG_HOME' not in os.environ:
        os.environ['PG_HOME'] = ngp

    # 必须在 import libPyBO39 之前设置 DLL 搜索路径
    _dirs = [
        os.path.join(ngp, 'libso', 'ndp'),
        os.path.join(ngp, 'libso', 'ndp', 'plugins'),
        os.path.join(ngp, 'libso', 'ndp', 'plugins', 'drivers'),
        os.path.join(ngp, 'libso', 'common'),
        os.path.join(ngp, 'libso', 'iecopy'),
        os.path.join(ngp, 'bin'),
        os.path.join(geoeast, 'bin'),
        os.path.join(geoeast, 'support', 'pg', 'bin'),
        os.path.join(geoeast, 'support', 'pg', 'lib'),
    ]

    if sys.platform.startswith('win'):
        # PATH 追加
        for d in _dirs:
            if os.path.isdir(d):
                entries = os.environ.get('PATH', '').split(os.pathsep)
                if d not in entries:
                    os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')

        # os.add_dll_directory (Python 3.8+ 受限 DLL 加载)
        for d in _dirs:
            if os.path.isdir(d):
                try:
                    os.add_dll_directory(d)
                except OSError:
                    pass

    # 预加载 NDP 框架 DLL — 顺序不能变
    # dpbase  → 基础平台
    # dpconf  → 配置加载 (读取 configs/ndp/*.conf)
    # dpplug  → 插件加载 (加载 ngp.dll, seismicnas.dll 等)
    _ndp_lib = os.path.join(ngp, 'libso', 'ndp')
    for _dll in ['dpbase.dll', 'dpconf.dll', 'dpplug.dll']:
        ctypes.CDLL(os.path.join(_ndp_lib, _dll))

    # libPyBO39.pyd 所在目录
    _iecopy = os.path.join(ngp, 'libso', 'iecopy')
    if _iecopy not in sys.path:
        sys.path.insert(0, _iecopy)


_setup()

import libPyBO39 as pybo


def get_root(dpname='ndp', user='admin1', password='admin1'):
    """获取 PyBOSystemRoot 实例并设置鉴权。"""
    root = pybo.PyBOSystemRoot.instance(dpname)
    root.setAuth(user, password)
    return root


def get_project(name, dpname='ndp', user='admin1', password='admin1'):
    """获取或创建项目。"""
    root = get_root(dpname, user, password)
    if not root.hasProject(name):
        root.createProject(name)
    return root.getProject(name)
