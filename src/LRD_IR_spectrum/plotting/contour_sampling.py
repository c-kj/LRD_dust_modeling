"""基于 Shapely 的等值线采样与交点工具集。"""

from __future__ import annotations

from enum import Enum
from typing import Sequence

import numpy as np
import numpy.typing as npt
from matplotlib.contour import ContourSet
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point

Array2D = npt.NDArray[np.float64]


# ============= 核心功能 =============

def extract_contour_lines(contour_set: ContourSet, target_level: float) -> list[LineString]:
    """从 `ContourSet` 中抽取目标等值线对应的所有折线。"""

    if getattr(contour_set, "filled", False):
        msg = "ContourSet 来自 contourf(); 仅支持 contour() 生成的线 contours"
        raise ValueError(msg)

    levels_list = list(contour_set.levels)
    
    try:
        level_idx = levels_list.index(target_level)
    except ValueError as exc:
        msg = f"Level {target_level} 不存在; 可用 levels: {levels_list}"
        raise ValueError(msg) from exc

    # 这里假定了 ContourSet 是 contour lines 而非 filled contour regions，所以 .levels 和 .allsegs 是等长的，应该不会越界
    segments = contour_set.allsegs[level_idx]

    lines = [
        LineString(seg)
        for seg in segments
        if seg.ndim == 2 and seg.shape[0] >= 2
    ]

    # 对于某个 level 上没有路径的情况，lines 是空的
    if not lines:
        msg = f"等值线 {target_level} 没有合法路径 (至少 2 个顶点)"
        raise ValueError(msg)

    return lines


class LineMergeStrategy(Enum):
    """多条等值线路径的合并策略。"""

    LONGEST_ONLY = "longest"          # 只取最长路径
    DIRECT_CONCAT = "direct"          # 直接按顺序连接
    NEAREST_NEIGHBOR = "nearest"      # 最近邻连接（贪心）


def merge_lines(
    lines: Sequence[LineString],
    strategy: LineMergeStrategy | str,
    *,
    allow_reverse: bool = True,
) -> LineString:
    """根据策略将多条 LineString 合并为一条 LineString。

    Args:
        lines: 待合并的折线序列。
        strategy: 合并策略枚举或其字符串值。
        allow_reverse: 在最近邻策略中，是否允许为减少跳跃而反转候选折线。
    """

    if isinstance(strategy, str):
        strategy = LineMergeStrategy(strategy)

    if not lines:
        msg = "至少需要一条路径才能合并"
        raise ValueError(msg)

    if len(lines) == 1:
        return lines[0]

    if strategy is LineMergeStrategy.LONGEST_ONLY:
        return max(lines, key=lambda line: line.length)

    if strategy is LineMergeStrategy.DIRECT_CONCAT:
        # 直接拼接坐标数组
        coords = np.vstack([np.array(line.coords) for line in lines])
        return LineString(coords)

    #BUG 这里的算法好像有问题，不如预期，有待检查
    if strategy is LineMergeStrategy.NEAREST_NEIGHBOR:
        # 贪心策略：从最长的路径开始，每次拼接最近的下一条
        remaining = sorted(list(lines), key=lambda line: line.length, reverse=True)
        merged_coords = list(remaining.pop(0).coords)

        while remaining:
            tail_point = Point(merged_coords[-1])
            best_idx = -1
            min_dist = float("inf")
            best_should_reverse = False

            for idx, candidate in enumerate(remaining):
                head_point = Point(candidate.coords[0])
                tail_candidate_point = Point(candidate.coords[-1])

                head_dist = tail_point.distance(head_point)
                candidate_dist = head_dist
                should_reverse = False

                if allow_reverse:
                    tail_dist = tail_point.distance(tail_candidate_point)
                    if tail_dist < candidate_dist:
                        candidate_dist = tail_dist
                        should_reverse = True

                if candidate_dist < min_dist:
                    min_dist = candidate_dist
                    best_idx = idx
                    best_should_reverse = should_reverse

            next_line = remaining.pop(best_idx)
            coords = list(next_line.coords)
            if allow_reverse and best_should_reverse:
                coords = list(reversed(coords))
            merged_coords.extend(coords)

        return LineString(merged_coords)

    msg = f"未知的合并策略: {strategy}"
    raise ValueError(msg)


#TODO: 支持 LineString | MultiLineString 作为输入。增加相应测试。考虑改名。
def sample_along_single_line(line: LineString, num_samples: int) -> Array2D:
    """在 LineString 上进行等距采样。"""
    fractions = np.linspace(0.0, 1.0, num_samples)
    samples = line.interpolate(fractions, normalized=True)
    return np.array([[point.x, point.y] for point in samples])


def sample_points_on_contour(
    contour_set: ContourSet,
    *,
    target_level: float,
    num_samples: int,
    merge_strategy: LineMergeStrategy | str = LineMergeStrategy.LONGEST_ONLY,
    allow_reverse: bool = True,
) -> Array2D:
    """
    高层封装：返回指定等值线上的采样点坐标数组。
    
    1. 提取指定 level 的所有 segments
    2. 按策略合并为一条 LineString（可选是否允许反向连接）
    3. 等距采样
    """

    lines = extract_contour_lines(contour_set, target_level)
    merged = merge_lines(lines, strategy=merge_strategy, allow_reverse=allow_reverse)
    return sample_along_single_line(merged, num_samples)


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
            lines = extract_contour_lines(contour_set, level)
        except ValueError:
            # 该 level 可能没有路径
            continue
            
        # 使用 MultiLineString 表示该 level 的所有等值线
        contour_multiline = MultiLineString(lines)
        
        intersection = contour_multiline.intersection(guide_line)
        
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
    "extract_contour_lines",
    "merge_lines",
    "sample_along_single_line",
    "sample_points_on_contour",
    "get_contour_line_intersections",
    "LineMergeStrategy",
]