"""GeoEast PyBO 数据读写封装

GeoEast 环境就绪后，替换 TODO 处的示例代码即可。
PyBO API 签名以实际 libPyBO39.pyd 暴露的为准。
"""

import numpy as np


def read_grid_data(project, grid_name: str) -> np.ndarray:
    """从 GeoEast 项目读取 BOMapGrid 沿层属性网格。

    Args:
        project: BOProject 对象（已打开的项目）
        grid_name: 网格数据名称

    Returns:
        2D numpy 数组 (rows, cols)，无效值替换为 0
    """
    # TODO: PyBO 实现
    # grid_obj = project.GetObject("BOMapGrid", grid_name)
    # if grid_obj is None:
    #     raise ValueError(f"网格数据不存在: {grid_name}")
    # nx, ny = grid_obj.GetSize()
    # buf = grid_obj.ReadData()
    # data = np.array(buf).reshape(ny, nx).astype(np.float64)
    # data[data > 1e30] = 0.0  # 无效值置零
    # return data
    raise NotImplementedError("PyBO 环境未就绪，请安装 GeoEast 后替换此函数")


def write_fault_polygons(project, name: str,
                         polygons: list,
                         areas: list = None) -> int:
    """将断层多边形写入 BOFaultPolygon 对象。

    Args:
        project: BOProject 对象
        name: 输出对象名称
        polygons: 多边形列表，每个元素为 (N, 2) numpy 数组 [[x, y], ...]
        areas: 面积列表（可选，用于排序）

    Returns:
        创建的 BOFaultPolygon 对象 ID
    """
    # TODO: PyBO 实现
    # obj = project.CreateObject("BOFaultPolygon", name)
    # for poly in polygons:
    #     seg = obj.AddSegment()
    #     for x, y in poly:
    #         seg.AddPoint(x, y, 0, 0)
    # obj.Save()
    # return obj.GetID()
    raise NotImplementedError("PyBO 环境未就绪，请安装 GeoEast 后替换此函数")


def list_grids(project) -> list:
    """列出项目下所有 BOMapGrid 网格数据名称。"""
    # TODO: PyBO 实现
    # return project.ListObjects("BOMapGrid")
    raise NotImplementedError("PyBO 环境未就绪，请安装 GeoEast 后替换此函数")


def remove_fault_polygon(project, name: str) -> bool:
    """删除指定名称的断层多边形（覆盖写入前清理）。"""
    # TODO: PyBO 实现
    # return project.DeleteObject("BOFaultPolygon", name)
    raise NotImplementedError("PyBO 环境未就绪，请安装 GeoEast 后替换此函数")
