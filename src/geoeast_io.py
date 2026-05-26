"""GeoEast PyBO 数据读写封装 — 断层多边形 I/O。

依赖:
    pybo_importer.py  — 提供 get_root() / get_project()
    libPyBO39.pyd     — GeoEast PyBO C++ 扩展 (Python 3.9)
"""

import numpy as np


def read_fault_polygons(project, name: str) -> list:
    """从 GeoEast 项目读取断层多边形数据。

    Args:
        project: PyBOProject 对象（已打开的项目）
        name: 断层多边形对象名称

    Returns:
        多边形列表，每个元素包含:
          - points: (N, 2) numpy 数组 [[x, y], ...]
          - close_flag: 是否闭合
    """
    if not project.hasFaultPolygon(name):
        raise ValueError(f"断层多边形不存在: {name}")

    fp = project.getFaultPolygon(name)
    fp.readData()
    fp.readDataHead()
    data = fp.getData()
    head = fp.getDataHead()

    polygons = []
    for seg in data.faultPolygonList:
        pts = seg.faultPolygonData
        if pts:
            coords = np.array([[p.x, p.y] for p in pts], dtype=np.float64)
        else:
            coords = np.empty((0, 2), dtype=np.float64)
        polygons.append({
            'points': coords,
            'close_flag': bool(seg.closeFlag),
        })

    return polygons


def write_fault_polygons(project, name: str,
                         polygons: list,
                         areas: list = None) -> bool:
    """将断层多边形写入 GeoEast 项目。

    Args:
        project: PyBOProject 对象
        name: 输出对象名称
        polygons: 多边形列表，每个元素为 (N, 2) numpy 数组 [[x, y], ...]
        areas: 面积列表（可选）

    Returns:
        True 表示写入成功
    """
    import libPyBO39 as pybo

    # 覆盖写入：先删再建
    if project.hasFaultPolygon(name):
        project.eraseFaultPolygon(name)
    fp = project.createFaultPolygon(name)

    # 构建线段数据
    segments = []
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for poly in polygons:
        seg = pybo.BAFaultPolygonSegment()
        seg.closeFlag = 1  # 断层多边形默认闭合

        pts = []
        for x, y in poly:
            pt = pybo.BAFaultPolygonPoint()
            pt.x = float(x)
            pt.y = float(y)
            pt.z = 0.0
            pts.append(pt)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        seg.faultPolygonData = pts  # 必须直接赋值，不能 append
        segments.append(seg)

    # 构建头部
    head = pybo.BAFaultPolygonHead()
    head.fauNum = len(segments)
    head.minX = min_x if min_x != float('inf') else 0.0
    head.maxX = max_x if max_x != float('-inf') else 0.0
    head.minY = min_y if min_y != float('inf') else 0.0
    head.maxY = max_y if max_y != float('-inf') else 0.0

    # 构建数据容器
    data = pybo.BAFaultPolygonData()
    data.faultPolygonList = segments  # 必须直接赋值

    fp.saveData(head, data)  # head 在前，data 在后；saveData 已包含持久化，不要再调 save()
    return True


def read_grid_geometry(project, grid_name: str) -> dict:
    """读取网格几何参数（不读全量数据，但 PyBO 不支持仅读头，会读数据）。"""
    if not project.hasMapGrid(grid_name):
        raise ValueError(f"网格数据不存在: {grid_name}")
    grid = project.getMapGrid(grid_name)
    grid.readData()
    head = grid.getDataHead()
    return {
        'nx': head.nx, 'ny': head.ny,
        'dx': head.dx, 'dy': head.dy,
        'sx': head.sx, 'sy': head.sy,
    }


def pixels_to_physical(polygons: list, geometry: dict) -> list:
    """将像素坐标多边形转为物理坐标。

    Args:
        polygons: [(N,2) array in (col,row) pixel coords, ...]
        geometry: read_grid_geometry 返回的几何参数字典

    Returns:
        [(N,2) array in (x,y) physical coords, ...]
    """
    import numpy as np
    sx, sy = geometry['sx'], geometry['sy']
    dx, dy = geometry['dx'], geometry['dy']
    result = []
    for poly in polygons:
        phys = poly.copy().astype(np.float64)
        phys[:, 0] = sx + poly[:, 0] * dx   # col → X
        phys[:, 1] = sy + poly[:, 1] * dy   # row → Y
        result.append(phys)
    return result


def list_grids(project) -> list:
    """列出项目下所有网格数据名称。"""
    grids = []
    for g in project.listMapGrid():
        grids.append(g.getName())
    return grids


def remove_fault_polygon(project, name: str) -> bool:
    """删除指定名称的断层多边形（覆盖写入前清理）。"""
    if project.hasFaultPolygon(name):
        project.eraseFaultPolygon(name)
        return True
    return False


def read_grid_data(project, grid_name: str) -> np.ndarray:
    """从 GeoEast 项目读取 BOMapGrid 沿层属性网格。

    Args:
        project: PyBOProject 对象
        grid_name: 网格数据名称

    Returns:
        2D numpy 数组 (rows, cols)
    """
    if not project.hasMapGrid(grid_name):
        raise ValueError(f"网格数据不存在: {grid_name}")

    grid = project.getMapGrid(grid_name)
    grid.readData()
    grid.readDataHead()
    data = grid.getData()
    head = grid.getDataHead()

    # BAMapGridData -> numpy
    if hasattr(data, 'data'):
        arr = np.array(data.data, dtype=np.float64).reshape(head.ny, head.nx)
    elif hasattr(data, 'getData'):
        arr = np.array(data.getData(), dtype=np.float64).reshape(head.ny, head.nx)
    else:
        raise RuntimeError(f"无法解析网格数据格式: {type(data)}")

    arr[arr > 1e30] = 0.0  # 无效值置零
    return arr
