"""contour_sampling 模块的单元测试 (pytest)。"""

import sys
from pathlib import Path
import numpy as np
import pytest
from shapely.geometry import LineString
from unittest.mock import MagicMock

# Ensure src is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LRD_IR_spectrum.plotting.contour_sampling import (
    LineMergeStrategy,
    extract_contour_lines,
    merge_lines,
    sample_along_single_line,
    sample_points_on_contour,
    get_contour_line_intersections,
)

# ============= Fixtures =============

@pytest.fixture
def mock_contour_set():
    """创建一个模拟的 ContourSet 对象。"""
    cs = MagicMock()
    cs.filled = False
    cs.levels = [10.0, 20.0, 30.0]
    # allsegs 结构: [level_index][segment_index][point_index][x, y]
    cs.allsegs = [
        [ # Level 10: 2 segments
            np.array([[0, 0], [1, 1], [2, 2]]),
            np.array([[3, 3], [4, 4]])
        ],
        [ # Level 20: 1 segment
            np.array([[0, 10], [10, 10]])
        ],
        [ # Level 30: Empty (no segments)
            np.empty((0, 2), dtype=float)
        ]
    ]
    return cs

# ============= Tests =============

def test_extract_contour_lines(mock_contour_set):
    # Test valid extraction
    lines = extract_contour_lines(mock_contour_set, 10.0)
    assert len(lines) == 2
    assert isinstance(lines[0], LineString)
    assert lines[0].coords[:] == [(0, 0), (1, 1), (2, 2)]

    # Test level not found
    with pytest.raises(ValueError, match="Level 99.0 不存在"):
        extract_contour_lines(mock_contour_set, 99.0)

    # Test empty level
    with pytest.raises(ValueError, match="没有合法路径"):
        extract_contour_lines(mock_contour_set, 30.0)

def test_extract_contour_lines_rejects_filled(mock_contour_set):
    mock_contour_set.filled = True
    with pytest.raises(ValueError, match=r"ContourSet 来自 contourf\(\)"):
        extract_contour_lines(mock_contour_set, 10.0)

#TODO 为什么用 .coords 来检验？shapely 的对象支持直接比较 == ，是不是用 == 更好？
def test_merge_lines_longest():
    l1 = LineString([(0, 0), (1, 0)]) # Length 1
    l2 = LineString([(0, 0), (0, 10)]) # Length 10
    
    merged = merge_lines([l1, l2], LineMergeStrategy.LONGEST_ONLY)
    assert merged.length == 10.0
    assert merged.coords[:] == [(0, 0), (0, 10)]

def test_merge_lines_direct():
    l1 = LineString([(0, 0), (1, 1)])
    l2 = LineString([(2, 2), (3, 3)])
    
    merged = merge_lines([l1, l2], LineMergeStrategy.DIRECT_CONCAT)
    # Should be (0,0)->(1,1)->(2,2)->(3,3)
    expected = [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert merged.coords[:] == expected

def test_merge_lines_nearest():
    # l1: (0,0) -> (1,0)
    # l2: (2,0) -> (3,0)
    # l3: (10,0) -> (11,0)
    # Start with l1. Nearest to (1,0) is l2's (2,0) [dist=1]. l3 is far [dist=9].
    l1 = LineString([(0, 0), (1, 0)])
    l2 = LineString([(2, 0), (3, 0)])
    l3 = LineString([(10, 0), (11, 0)])
    
    merged = merge_lines([l1, l3, l2], LineMergeStrategy.NEAREST_NEIGHBOR)
    # Expected order: l1 -> l2 -> l3
    expected = [(0, 0), (1, 0), (2, 0), (3, 0), (10, 0), (11, 0)]
    assert merged.coords[:] == expected

def test_merge_lines_nearest_allow_reverse_toggle():
    l1 = LineString([(0, 0), (1, 0)])
    l2 = LineString([(2, 0), (1, 0)])  # 方向与期望相反

    merged_allow = merge_lines(
        [l1, l2],
        LineMergeStrategy.NEAREST_NEIGHBOR,
        allow_reverse=True,
    )
    assert merged_allow.coords[:] == [(0, 0), (1, 0), (1, 0), (2, 0)]

    merged_disallow = merge_lines(
        [l1, l2],
        LineMergeStrategy.NEAREST_NEIGHBOR,
        allow_reverse=False,
    )
    assert merged_disallow.coords[:] == [(0, 0), (1, 0), (2, 0), (1, 0)]

def test_sample_along_single_line():
    line = LineString([(0, 0), (10, 0)])
    samples = sample_along_single_line(line, 11)
    assert samples.shape == (11, 2)
    assert samples[0].tolist() == [0.0, 0.0]
    assert samples[-1].tolist() == [10.0, 0.0]
    assert samples[5].tolist() == [5.0, 0.0] # Midpoint

def test_sample_along_single_line_zero_length():
    line = LineString([(1, 1), (1, 1)])
    samples = sample_along_single_line(line, 3)
    assert np.all(samples == np.array([[1.0, 1.0]] * 3))

def test_sample_along_single_line_zero_samples():
    line = LineString([(0, 0), (1, 1)])
    samples = sample_along_single_line(line, 0)
    assert isinstance(samples, np.ndarray)

def test_sample_points_on_contour(mock_contour_set):
    # Level 20 has one line: (0, 10) -> (10, 10). Length 10.
    samples = sample_points_on_contour(
        mock_contour_set, 
        target_level=20.0, 
        num_samples=3,
        merge_strategy='longest'
    )
    assert samples.shape == (3, 2)
    expected = np.array([[0, 10], [5, 10], [10, 10]])
    np.testing.assert_allclose(samples, expected)

def test_get_contour_line_intersections(mock_contour_set):
    # Guide line: Vertical line at x=0.5
    # Level 10: (0,0)->(2,2) intersects at (0.5, 0.5)
    # Level 20: (0,10)->(10,10) intersects at (0.5, 10)
    guide = LineString([(0.5, -10), (0.5, 20)])
    
    intersections = get_contour_line_intersections(mock_contour_set, guide)
    
    assert 10.0 in intersections
    assert 20.0 in intersections
    
    np.testing.assert_allclose(intersections[10.0], (0.5, 0.5))
    np.testing.assert_allclose(intersections[20.0], (0.5, 10.0))

def test_get_contour_line_intersections_supports_multipoint():
    cs = MagicMock()
    cs.filled = False
    cs.levels = [5.0]
    cs.allsegs = [[
        np.array([[0, 0], [2, 0]]),
        np.array([[0, 1], [2, 1]]),
    ]]

    guide = LineString([(1.0, -1.0), (1.0, 2.0)])
    intersections = get_contour_line_intersections(cs, guide)

    assert 5.0 in intersections
    assert intersections[5.0] in {(1.0, 0.0), (1.0, 1.0)}

def test_get_contour_line_intersections_raises_on_linestring_contact():
    cs = MagicMock()
    cs.filled = False
    cs.levels = [5.0]
    cs.allsegs = [[
        np.array([[0, 0], [2, 0]]),
    ]]

    guide = LineString([(0, 0), (2, 0)])

    with pytest.raises(TypeError, match="交点类型 LineString 不受支持"):
        get_contour_line_intersections(cs, guide)
