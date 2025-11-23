"""基于 Shapely 的等值线采样与交点工具集。"""

#TODO 重构：改用 shapely 之后，很多功能完全冗余了。这个模块目前只想做两件事：放置 clabel、在 contour line 上采样。

from __future__ import annotations

from enum import Enum
from typing import Sequence

import numpy as np
import numpy.typing as npt
from matplotlib.contour import ContourSet
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
)

Array2D = npt.NDArray[np.float64]


# ============= 通用工具 =============

def _ensure_linestring(vertices: Array2D) -> LineString:
    """将 (N,2) 顶点数组转为 LineString，并在入口处做完整校验。"""

    arr = np.asarray(vertices, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        msg = "曲线顶点必须是 (N, 2) 结构"
        raise ValueError(msg)
    if arr.shape[0] < 2:
        msg = "曲线至少需要 2 个顶点"
        raise ValueError(msg)
    return LineString(arr)


def _line_to_array(line: LineString) -> Array2D:
    """将 LineString 转回 numpy 数组，方便排序或拼接。"""

    return np.asarray(line.coords, dtype=float)


def _first_point(line: LineString) -> Array2D:
    """取首个顶点坐标。"""

    return np.array(line.coords[0], dtype=float)


def _last_point(line: LineString) -> Array2D:
    """取最后一个顶点坐标。"""

    return np.array(line.coords[-1], dtype=float)


def _stack_vertices(lines: Sequence[LineString]) -> Array2D:
    """按顺序堆叠多条线的顶点。"""

    if not lines:
        return np.empty((0, 2), dtype=float)
    arrays = [_line_to_array(line) for line in lines]
    return np.vstack(arrays)


def _line_lengths(lines: Sequence[LineString]) -> list[float]:
    """返回每条折线的长度数组。"""

    return [line.length for line in lines]


def _geometry_to_points(geometry) -> Array2D:
    """把 Shapely 交点对象统一展开为 (N,2) 数组。"""

    if geometry.is_empty:
        return np.empty((0, 2), dtype=float)

    if isinstance(geometry, Point):
        return np.array([[geometry.x, geometry.y]], dtype=float)

    if isinstance(geometry, MultiPoint):
        return np.array([[pt.x, pt.y] for pt in geometry.geoms], dtype=float)

    if isinstance(geometry, LineString):
        return _line_to_array(geometry)

    if isinstance(geometry, MultiLineString):
        segments = [_line_to_array(line) for line in geometry.geoms]
        return np.vstack(segments)

    if isinstance(geometry, GeometryCollection):
        pieces: list[Array2D] = []
        for geom in geometry.geoms:
            piece = _geometry_to_points(geom)
            if piece.size:
                pieces.append(piece)
        return np.vstack(pieces) if pieces else np.empty((0, 2), dtype=float)

    msg = f"暂不支持的交点类型: {geometry.geom_type}"
    raise TypeError(msg)


# ============= 步骤 1: 提取等值线路径 =============

def extract_contour_lines(contour_set: ContourSet, target_level: float) -> list[LineString]:
    """从 `ContourSet` 中抽取目标等值线对应的所有折线。"""

    if not hasattr(contour_set, "levels") or not hasattr(contour_set, "allsegs"):
        msg = "ContourSet 需要同时包含 'levels' 与 'allsegs' 属性"
        raise AttributeError(msg)

    levels_list = list(contour_set.levels)
    
    #* 这里假定了 ContourSet 是 contour lines 而非 filled contour regions，所以 .levels 和 .allsegs 是等长的
    try:
        level_idx = levels_list.index(target_level)
    except ValueError as exc:  # pragma: no cover - defensive branch
        msg = f"Level {target_level} is not present; available levels: {levels_list}"
        raise ValueError(msg) from exc

    try:
        segments = contour_set.allsegs[level_idx]
    except IndexError as exc:  # pragma: no cover - defensive branch
        msg = f"Index {level_idx} is out of bounds for allsegs"
        raise IndexError(msg) from exc

    lines = [
        LineString(np.asarray(seg, dtype=float))
        for seg in segments
        if isinstance(seg, np.ndarray) and seg.ndim == 2 and seg.shape[0] >= 2
    ]

    if not lines:
        msg = f"等值线 {target_level} 没有合法路径 (至少 2 个顶点)"
        raise ValueError(msg)

    return lines


# ============= 步骤 3: 路径合并策略 =============
    
class LineMergeStrategy(Enum):
    """Strategies for combining multiple contour lines."""

    LONGEST_ONLY = "longest"          # 只取最长路径
    DIRECT_CONCAT = "direct"          # 直接按顺序连接
    SORT_BY_START_X = "sort_x"        # 按起点 x 坐标排序后连接
    SORT_BY_START_Y = "sort_y"        # 按起点 y 坐标排序后连接
    SORT_BY_LENGTH = "sort_length"    # 按长度降序排列后连接
    NEAREST_NEIGHBOR = "nearest"      # 最近邻连接（贪心）


def merge_lines(lines: Sequence[LineString], strategy: LineMergeStrategy | str) -> LineString:
    """根据策略返回一条新的合并折线。"""

    if isinstance(strategy, str):
        strategy = LineMergeStrategy(strategy)

    if not lines:
        msg = "至少需要一条路径才能合并"
        raise ValueError(msg)

    if len(lines) == 1:
        return LineString(lines[0])

    if strategy is LineMergeStrategy.LONGEST_ONLY:
        lengths = _line_lengths(lines)
        return LineString(lines[int(np.argmax(lengths))])

    if strategy is LineMergeStrategy.DIRECT_CONCAT:
        return LineString(_stack_vertices(lines))

    if strategy is LineMergeStrategy.SORT_BY_START_X:
        ordered = sorted(lines, key=lambda line: _first_point(line)[0])
        return LineString(_stack_vertices(ordered))

    if strategy is LineMergeStrategy.SORT_BY_START_Y:
        ordered = sorted(lines, key=lambda line: _first_point(line)[1])
        return LineString(_stack_vertices(ordered))

    if strategy is LineMergeStrategy.SORT_BY_LENGTH:
        lengths = _line_lengths(lines)
        ordered = [line for _, line in sorted(zip(lengths, lines), reverse=True)]
        return LineString(_stack_vertices(ordered))

    if strategy is LineMergeStrategy.NEAREST_NEIGHBOR:
        remaining = list(lines)
        merged_order = [remaining.pop(0)]
        while remaining:
            tail = _last_point(merged_order[-1])
            distances = [np.linalg.norm(_first_point(line) - tail) for line in remaining]
            merged_order.append(remaining.pop(int(np.argmin(distances))))
        return LineString(_stack_vertices(merged_order))

    msg = f"未知的合并策略: {strategy}"
    raise ValueError(msg)


# ============= 步骤 4: 单条路径上的均匀采样 =============

def sample_along_single_line(line: LineString, num_samples: int) -> Array2D:
    """利用 `LineString.interpolate` 等距采样。"""

    if num_samples <= 0:
        return np.empty((0, 2), dtype=float)

    length = line.length
    if length == 0:
        first = _line_to_array(line)[:1]
        return np.repeat(first, num_samples, axis=0)

    distances = np.linspace(0.0, length, num_samples)
    samples = [line.interpolate(dist).coords[0] for dist in distances]
    return np.asarray(samples, dtype=float)


# ============= 步骤 5: 多路径独立采样策略 =============

def sample_multiple_lines_separately(
    lines: Sequence[LineString],
    num_samples: int,
    *,
    by_length_proportion: bool = True,
) -> Array2D:
    """对每条路径独立采样，再拼接输出。"""

    if not lines:
        return np.empty((0, 2), dtype=float)

    if by_length_proportion:
        lengths = _line_lengths(lines)
        total = sum(lengths)
        if total <= 0:
            per_line = [max(1, num_samples // len(lines))] * len(lines)
        else:
            per_line = [max(1, int(num_samples * length / total)) for length in lengths]
    else:
        base = max(1, num_samples // len(lines))
        per_line = [base] * len(lines)

    samples = [
        sample_along_single_line(line, count)
        for line, count in zip(lines, per_line, strict=False)
        if count > 0
    ]
    return np.vstack(samples) if samples else np.empty((0, 2), dtype=float)


# ============= 步骤 6: 高层组装函数 =============

def sample_points_on_contour(
    contour_set: ContourSet,
    *,
    target_level: float,
    num_samples: int,
    merge_strategy: LineMergeStrategy | str,
    separate_sampling: bool = False,
    by_length_proportion: bool = True,
) -> Array2D:
    """高层封装：返回指定等值线上的采样点坐标数组。"""

    lines = extract_contour_lines(contour_set, target_level)

    if separate_sampling:
        return sample_multiple_lines_separately(lines, num_samples, by_length_proportion=by_length_proportion)

    merged = merge_lines(lines, strategy=merge_strategy)
    return sample_along_single_line(merged, num_samples)


def inspect_contour_lines(contour_set: ContourSet, target_level: float) -> dict[str, object]:
    """输出等值线的数量、长度和顶点数信息，便于调试布局策略。"""

    lines = extract_contour_lines(contour_set, target_level)
    lengths = _line_lengths(lines)
    return {
        "num_lines": len(lines),
        "line_lengths": lengths,
        "num_vertices": [len(line.coords) for line in lines],
        "total_length": float(sum(lengths)),
        "longest_idx": int(np.argmax(lengths)),
        "shortest_idx": int(np.argmin(lengths)),
    }


# ============= 交点：全程依赖 Shapely =============

def intersect_line_with_polyline(line: LineString, curve_vertices: Array2D) -> Array2D:
    """求 line 与另一条折线的所有交点（Point/Line 均兼容）。"""

    other = _ensure_linestring(curve_vertices)
    intersection = line.intersection(other)
    return _geometry_to_points(intersection)

#TODO 写注释、简化、取代 intersect_line_with_polyline 函数、更新文档
def get_contour_line_intersections(
    contour_set: ContourSet,
    clabel_pos_line: LineString,
):

    intersection_points: dict[float, tuple[float, float]] = {}
    for level in contour_set.levels:
        try:
            contour_line_segments = extract_contour_lines(contour_set, level)
        except ValueError as e:  # contour line 为空
            continue
        contour_multiline = MultiLineString(contour_line_segments)
    
        point = contour_multiline.intersection(clabel_pos_line)
        assert not point.is_empty, "No intersection found"
        assert type(point) is Point, f"Expected Point, got {type(point)}"
        intersection_points[level] = point.coords[0]
    
    return intersection_points

__all__ = [
    "Array2D",
    "LineMergeStrategy",
    "extract_contour_lines",
    "inspect_contour_lines",
    "intersect_line_with_polyline",
    "merge_lines",
    "sample_along_single_line",
    "sample_multiple_lines_separately",
    "sample_points_on_contour",
    "get_contour_line_intersections",
]