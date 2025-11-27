"""基于 Shapely 的等值线采样与交点工具集。"""
#TODO 给整个模块改名

from __future__ import annotations

from enum import Enum
from typing import Sequence

import numpy as np
import numpy.typing as npt
from matplotlib.contour import ContourSet
from shapely import line_merge, shortest_line
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point

# 表示 (N, 2) 形状的二维坐标点数组，每行为 [x, y]。
CoordArray = npt.NDArray[np.float64]


# ============= 核心功能 =============

def get_contour_at_level(contour_set: ContourSet, level: float) -> MultiLineString:
    """从 ContourSet 中提取指定 level 的等值线。
    
    Args:
        contour_set: matplotlib 的 ContourSet 对象（必须来自 contour() 而非 contourf()）。
        level: 目标等值线的值。
    
    Returns:
        该 level 的等值线（可能包含多条分离的曲线）。
    
    Raises:
        ValueError: 如果 ContourSet 来自 contourf()，或 level 不存在，或该 level 没有合法路径。
    """
    if getattr(contour_set, "filled", False):
        msg = "ContourSet 来自 contourf(); 仅支持 contour() 生成的线 contours"
        raise ValueError(msg)

    levels_list = list(contour_set.levels)
    
    try:
        level_idx = levels_list.index(level)
    except ValueError as exc:
        msg = f"Level {level} 不存在; 可用 levels: {levels_list}"
        raise ValueError(msg) from exc

    # 这里假定了 ContourSet 是 contour lines 而非 filled contour regions，
    # 所以 .levels 和 .allsegs 是等长的，应该不会越界
    segments = contour_set.allsegs[level_idx]

    lines = [
        LineString(seg)
        for seg in segments
        if seg.ndim == 2 and seg.shape[0] >= 2
    ]

    if not lines:
        msg = f"等值线 {level} 没有合法路径 (至少 2 个顶点)"
        raise ValueError(msg)

    return MultiLineString(lines)


class LineConnectStrategy(Enum):
    """多条等值线路径的连接策略。"""

    LONGEST_ONLY = "longest"          # 只取最长路径
    DIRECT_CONCAT = "direct"          # 直接按顺序连接
    GREEDY = "greedy"                 # 从指定起点贪心连接最近邻
    SHORTEST_GREEDY = "shortest_greedy"  # 枚举所有起点的贪心结果，取最短者（不一定是所有连线方案中最短的）


def _extract_endpoints(line: LineString) -> MultiPoint:
    """提取 LineString 的首尾端点，作为 MultiPoint 返回。
    这里不使用 line.boundary，因为对于闭合线段，boundary 是空的。而这个函数想确保总是返回两个点组成的 MultiPoint
    """
    return MultiPoint([line.coords[0], line.coords[-1]])


def _greedy_connect(lines: Sequence[LineString], start_index: int = 0) -> LineString:
    """从指定线段开始，贪心地连接最近的下一条线段。

    Args:
        lines: 待连接的线段列表。
        start_index: 起始线段的索引。

    Returns:
        连接后的 LineString。
    """
    remaining = list(lines)
    connected = remaining.pop(start_index)

    # 每次选择 remaining 中与 connected 端点距离最近的线段，用 line_merge 连接。
    while remaining:
        connected_endpoints: MultiPoint = _extract_endpoints(connected)  # connected 的首尾点。不应该用 boundary 因为有可能是闭合的
        bridge_list: list[LineString] = [shortest_line(connected_endpoints, _extract_endpoints(line)) for line in remaining]
        shortest_bridge_idx = np.argmin([bridge.length for bridge in bridge_list])
        bridge: LineString = bridge_list[shortest_bridge_idx]
        next_line: LineString = remaining.pop(shortest_bridge_idx)
        # line_merge 如果有的 LineString 无共享端点，会返回 MultiLineString。那么下一次循环就会出错（没有 .coords 属性）。但这里的算法应该保证了总是有共享端点。
        connected = line_merge(MultiLineString([connected, bridge, next_line])) 

    return connected


def connect_lines(
    lines: MultiLineString | Sequence[LineString],
    strategy: LineConnectStrategy | str,
    *,
    start_index: int = 0,
) -> LineString:
    """根据策略将多条 LineString 连接为一条 LineString。

    Args:
        lines: 待连接的折线，可以是 MultiLineString 或 LineString 序列。
        strategy: 连接策略枚举或其字符串值。
        start_index: 仅用于 GREEDY 策略，指定从哪条线开始（默认为 0）。
    """
    # 统一转为 Sequence[LineString]
    if isinstance(lines, MultiLineString):
        lines = list(lines.geoms)

    if isinstance(strategy, str):
        strategy = LineConnectStrategy(strategy)

    if not lines:
        msg = "lines 不能为空"
        raise ValueError(msg)

    if len(lines) == 1:
        return lines[0]

    if strategy is LineConnectStrategy.LONGEST_ONLY:
        return max(lines, key=lambda line: line.length)

    elif strategy is LineConnectStrategy.DIRECT_CONCAT:
        coords = np.vstack([np.array(line.coords) for line in lines])
        return LineString(coords)

    elif strategy is LineConnectStrategy.GREEDY:
        return _greedy_connect(lines, start_index=start_index)

    elif strategy is LineConnectStrategy.SHORTEST_GREEDY:
        # 枚举所有起点的贪心结果，取最短者。注意：这不保证全局最优。
        greedy_connect_results = (_greedy_connect(lines, start_index=idx) for idx in range(len(lines)))
        return min(greedy_connect_results, key=lambda line: line.length)

    else:
        msg = f"未知的连接策略: {strategy}"
        raise ValueError(msg)


def sample_along_line(
    line: LineString | MultiLineString,
    num_samples: int,
) -> CoordArray:
    """在 LineString 或 MultiLineString 上进行等距采样。
    
    对于 MultiLineString，采样按总长度进行（各段连续计算）。
    
    Args:
        line: 待采样的线段或多线段。
        num_samples: 采样点数量，必须为正整数。
    
    Returns:
        shape 为 (num_samples, 2) 的坐标数组。
    
    Raises:
        ValueError: 如果 num_samples <= 0。
    """
    if num_samples <= 0:
        msg = f"num_samples 必须为正整数，得到 {num_samples}"
        raise ValueError(msg)
    fractions = np.linspace(0.0, 1.0, num_samples)
    samples = line.interpolate(fractions, normalized=True)
    return np.array([[point.x, point.y] for point in samples])


def sample_points_on_contour(
    contour_set: ContourSet,
    *,
    target_level: float,
    num_samples: int,
    connect_strategy: LineConnectStrategy | str | None = None,
) -> CoordArray:
    """
    高层封装：返回指定等值线上的采样点坐标数组。
    
    1. 提取指定 level 的所有 segments
    2. 按策略连接为一条 LineString，如果策略为 None 则不连接
    3. 等距采样
    
    Args:
        connect_strategy: 连接策略。若为 None，则不连接，直接对所有线段组成的 MultiLineString 按总长度进行采样。
    """
    contour = get_contour_at_level(contour_set, target_level)
    
    if connect_strategy is None:
        # 不合并，直接对 MultiLineString 采样（按总长度）
        target_line = contour
    else:
        target_line = connect_lines(contour, strategy=connect_strategy)
    
    return sample_along_line(target_line, num_samples)


def get_contour_line_intersections(
    contour_set: ContourSet,
    guide_line: LineString,
) -> dict[float, tuple[float, float]]:
    """
    计算 ContourSet 中所有 levels 与 guide_line 的交点。
    用于确定 clabel 的放置位置。
    
    Args:
        contour_set: matplotlib ContourSet 对象
        guide_line: 用于定位的引导线 (Shapely LineString)
        
    Returns:
        dict: {level_value: (x, y)}
    """

    intersection_points: dict[float, tuple[float, float]] = {}
    
    for level in contour_set.levels:
        try:
            contour = get_contour_at_level(contour_set, level)
        except ValueError:
            # 该 level 可能没有路径
            continue
        
        intersection = contour.intersection(guide_line)
        
        if intersection.is_empty:
            continue

        if isinstance(intersection, Point):
            points = [intersection]
        elif isinstance(intersection, MultiPoint):
            points = list(intersection.geoms)
        else:
            msg = f"交点类型 {intersection.geom_type} 不受支持，仅允许 Point 或 MultiPoint"
            raise TypeError(msg)

        if not points:
            continue

        pt = points[0]
        intersection_points[level] = (pt.x, pt.y)
            
    return intersection_points


__all__ = [
    "get_contour_at_level",
    "connect_lines",
    "sample_along_line",
    "sample_points_on_contour",
    "get_contour_line_intersections",
    "LineConnectStrategy",
    "CoordArray",
]