"""Utilities for sampling points along contour lines.

These helpers were extracted from exploratory notebook work
and are now available as part of the public plotting toolkit.
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence, TypeAlias

import numpy as np
import numpy.typing as npt
from matplotlib.contour import ContourSet

PathArray: TypeAlias = npt.NDArray[np.floating]
"""Alias for arrays representing an N*2 sequence of contour vertices."""

# ============= 步骤 1: 提取等值线路径 =============

def extract_contour_paths(contour_set: ContourSet, target_level: float) -> list[PathArray]:
    """Return all contour paths at *target_level* as arrays of vertices."""
    if not hasattr(contour_set, "levels") or not hasattr(contour_set, "allsegs"):
        msg = "ContourSet requires both 'levels' and 'allsegs' attributes"
        raise AttributeError(msg)

    levels_list = list(contour_set.levels)
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

    valid_paths: list[PathArray] = []
    for seg in segments:
        if isinstance(seg, np.ndarray) and seg.ndim == 2 and seg.shape[0] >= 2:
            valid_paths.append(seg.astype(float, copy=False))

    if not valid_paths:
        msg = f"Level {target_level} has no valid paths with at least two vertices"
        raise ValueError(msg)

    return valid_paths


# ============= 步骤 2: 计算路径长度 =============

def calc_path_length(path: PathArray) -> float:
    """Compute the total arc length of a single path."""
    if path.size == 0 or path.shape[0] < 2:
        return 0.0

    segment_vectors = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    return float(np.sum(segment_lengths))


def calc_all_path_lengths(paths: Sequence[PathArray]) -> list[float]:
    """Return the lengths for every path in *paths*."""
    return [calc_path_length(path) for path in paths]


# ============= 步骤 3: 路径合并策略 =============
    
class PathMergeStrategy(Enum):
    """Strategies for combining multiple contour paths."""
    LONGEST_ONLY = "longest"          # 只取最长路径
    DIRECT_CONCAT = "direct"          # 直接按顺序连接
    SORT_BY_START_X = "sort_x"        # 按起点 x 坐标排序后连接
    SORT_BY_START_Y = "sort_y"        # 按起点 y 坐标排序后连接
    SORT_BY_LENGTH = "sort_length"    # 按长度降序排列后连接
    NEAREST_NEIGHBOR = "nearest"      # 最近邻连接（贪心）


def merge_paths(paths: Sequence[PathArray], strategy: PathMergeStrategy | str) -> PathArray:
    """Merge *paths* according to *strategy* and return a single vertex array."""
    if isinstance(strategy, str):
        strategy = PathMergeStrategy(strategy)

    if not paths:
        return np.empty((0, 2), dtype=float)

    if len(paths) == 1:
        return np.asarray(paths[0], dtype=float)

    if strategy is PathMergeStrategy.LONGEST_ONLY:
        lengths = calc_all_path_lengths(paths)
        longest_idx = int(np.argmax(lengths))
        return np.asarray(paths[longest_idx], dtype=float)

    if strategy is PathMergeStrategy.DIRECT_CONCAT:
        return np.vstack([np.asarray(path, dtype=float) for path in paths])

    if strategy is PathMergeStrategy.SORT_BY_START_X:
        sorted_paths = [np.asarray(path, dtype=float) for path in sorted(paths, key=lambda p: p[0, 0])]
        return np.vstack(sorted_paths)

    if strategy is PathMergeStrategy.SORT_BY_START_Y:
        sorted_paths = [np.asarray(path, dtype=float) for path in sorted(paths, key=lambda p: p[0, 1])]
        return np.vstack(sorted_paths)

    if strategy is PathMergeStrategy.SORT_BY_LENGTH:
        lengths = calc_all_path_lengths(paths)
        sorted_paths = [np.asarray(path, dtype=float) for _, path in sorted(zip(lengths, paths), reverse=True)]
        return np.vstack(sorted_paths)

    if strategy is PathMergeStrategy.NEAREST_NEIGHBOR:
        remaining = list(paths)
        result = [remaining.pop(0)]
        while remaining:
            last_point = result[-1][-1]
            distances = [np.linalg.norm(path[0] - last_point) for path in remaining]
            nearest_idx = int(np.argmin(distances))
            result.append(remaining.pop(nearest_idx))
        stacked = [np.asarray(path, dtype=float) for path in result]
        return np.vstack(stacked)

    msg = f"Unsupported strategy: {strategy}"
    raise ValueError(msg)

# ============= 步骤 4: 单条路径上的均匀采样 =============

def sample_along_single_path(path: PathArray, num_samples: int) -> PathArray:
    """Uniformly sample *num_samples* points along a single path."""
    if path.size == 0:
        return np.empty((0, 2), dtype=float)

    if path.shape[0] == 1:
        return np.repeat(path[:1], num_samples, axis=0)

    segment_vectors = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    total_length = float(np.sum(segment_lengths))

    if total_length <= 0:
        return np.repeat(path[:1], num_samples, axis=0)

    cumulative_distances = np.zeros(path.shape[0], dtype=float)
    cumulative_distances[1:] = np.cumsum(segment_lengths)

    sample_distances = np.linspace(0, total_length, num_samples)
    sampled_x = np.interp(sample_distances, cumulative_distances, path[:, 0])
    sampled_y = np.interp(sample_distances, cumulative_distances, path[:, 1])

    return np.column_stack((sampled_x, sampled_y))

# ============= 步骤 5: 多路径独立采样策略 =============

def sample_multiple_paths_separately(
    paths: Sequence[PathArray],
    num_samples: int,
    *,
    by_length_proportion: bool = True,
) -> PathArray:
    """Sample each path independently and return the stacked samples."""
    if not paths:
        return np.empty((0, 2), dtype=float)

    if by_length_proportion:
        lengths = calc_all_path_lengths(paths)
        total_length = sum(lengths)
        if total_length <= 0:
            base = max(1, num_samples // len(paths))
            samples_per_path = [base] * len(paths)
        else:
            samples_per_path = [max(1, int(num_samples * length / total_length)) for length in lengths]
    else:
        base = max(1, num_samples // len(paths))
        samples_per_path = [base] * len(paths)

    samples = [
        sample_along_single_path(path, count)
        for path, count in zip(paths, samples_per_path, strict=False)
        if count > 0
    ]

    return np.vstack(samples) if samples else np.empty((0, 2), dtype=float)


# ============= 步骤 6: 高层组装函数 =============

def sample_points_on_contour(
    contour_set: ContourSet,
    *,
    target_level: float,
    num_samples: int,
    merge_strategy: PathMergeStrategy | str,
    separate_sampling: bool = False,
    by_length_proportion: bool = True,
) -> PathArray:
    """High-level orchestration for sampling points on a contour level."""
    paths = extract_contour_paths(contour_set, target_level)

    if separate_sampling:
        return sample_multiple_paths_separately(paths, num_samples, by_length_proportion=by_length_proportion)

    merged_path = merge_paths(paths, strategy=merge_strategy)
    return sample_along_single_path(merged_path, num_samples)


def inspect_contour_paths(contour_set: ContourSet, target_level: float) -> dict[str, object]:
    """Return summary statistics describing contour paths at *target_level*."""
    paths = extract_contour_paths(contour_set, target_level)
    lengths = calc_all_path_lengths(paths)
    return {
        "num_paths": len(paths),
        "path_shapes": [path.shape for path in paths],
        "path_lengths": lengths,
        "total_length": float(sum(lengths)),
        "longest_idx": int(np.argmax(lengths)),
        "shortest_idx": int(np.argmin(lengths)),
    }



__all__ = [
    "PathArray",
    "PathMergeStrategy",
    "calc_all_path_lengths",
    "calc_path_length",
    "extract_contour_paths",
    "inspect_contour_paths",
    "merge_paths",
    "sample_along_single_path",
    "sample_multiple_paths_separately",
    "sample_points_on_contour",
]