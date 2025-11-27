"""contour_sampling 模块的单元测试 (pytest)。"""

import sys
from pathlib import Path
import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString
from unittest.mock import MagicMock

#TODO 是否有更好的方式处理 src 路径问题？
# Ensure src is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LRD_IR_spectrum.plotting.contour_utils import (
    LineConnectStrategy,
    get_contour_at_level,
    get_contour_line_intersections,
    connect_lines,
    sample_along_line,
    sample_points_on_contour,
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

def test_get_contour_at_level(mock_contour_set):
    # Test valid extraction
    contour = get_contour_at_level(mock_contour_set, 10.0)
    assert isinstance(contour, MultiLineString)
    assert len(contour.geoms) == 2
    assert contour.geoms[0] == LineString([(0, 0), (1, 1), (2, 2)])

    # Test level not found
    with pytest.raises(ValueError, match="Level 99.0 不存在"):
        get_contour_at_level(mock_contour_set, 99.0)

    # Test empty level
    with pytest.raises(ValueError, match="没有合法路径"):
        get_contour_at_level(mock_contour_set, 30.0)

def test_get_contour_at_level_rejects_filled(mock_contour_set):
    mock_contour_set.filled = True
    with pytest.raises(ValueError, match=r"ContourSet 来自 contourf\(\)"):
        get_contour_at_level(mock_contour_set, 10.0)

def test_connect_lines_longest():
    l1 = LineString([(0, 0), (1, 0)]) # Length 1
    l2 = LineString([(0, 0), (0, 10)]) # Length 10
    
    connected = connect_lines([l1, l2], LineConnectStrategy.LONGEST_ONLY)
    assert connected.length == 10.0
    assert connected == LineString([(0, 0), (0, 10)])

def test_connect_lines_accepts_multilinestring():
    """connect_lines 应该同时支持 MultiLineString 和 Sequence[LineString]。"""
    mls = MultiLineString([
        [(0, 0), (1, 0)],   # Length 1
        [(0, 0), (0, 10)],  # Length 10
    ])
    connected = connect_lines(mls, LineConnectStrategy.LONGEST_ONLY)
    assert connected.length == 10.0
    assert connected == LineString([(0, 0), (0, 10)])

def test_connect_lines_direct():
    l1 = LineString([(0, 0), (1, 1)])
    l2 = LineString([(2, 2), (3, 3)])
    
    connected = connect_lines([l1, l2], LineConnectStrategy.DIRECT_CONCAT)
    # Should be (0,0)->(1,1)->(2,2)->(3,3)
    assert connected == LineString([(0, 0), (1, 1), (2, 2), (3, 3)])

def test_connect_lines_greedy():
    """GREEDY 策略从 lines[0] 开始，贪心选择最近的下一条。"""
    l1 = LineString([(0, 0), (1, 0)])
    l2 = LineString([(2, 0), (3, 0)])
    l3 = LineString([(10, 0), (11, 0)])
    
    # 从 l1 开始，最近的是 l2，然后是 l3
    connected = connect_lines([l1, l3, l2], LineConnectStrategy.GREEDY)
    assert connected == LineString([(0, 0), (1, 0), (2, 0), (3, 0), (10, 0), (11, 0)])

def test_connect_lines_greedy_with_start_index():
    """GREEDY 策略可通过 start_index 指定起点。"""
    l1 = LineString([(0, 0), (1, 0)])
    l2 = LineString([(2, 0), (3, 0)])
    
    # 从 l2 (index=1) 开始，结果包含所有点
    connected = connect_lines([l1, l2], LineConnectStrategy.GREEDY, start_index=1)
    # line_merge 会自动处理方向，结果方向不确定，但应包含所有坐标
    assert set(connected.coords) == {(0, 0), (1, 0), (2, 0), (3, 0)}
    assert connected.length == 3.0  # 总长度 = 1 + 1 + 1

def test_connect_lines_greedy_auto_reverse():
    """GREEDY 策略自动处理方向并合并共享端点。"""
    l1 = LineString([(0, 0), (1, 0)])
    l2 = LineString([(2, 0), (1, 0)])  # 尾端与 l1 尾端相邻

    connected = connect_lines([l1, l2], LineConnectStrategy.GREEDY)
    # line_merge 会自动反转 l2 并合并共享端点 (1,0)
    assert connected == LineString([(0, 0), (1, 0), (2, 0)])

def test_connect_lines_shortest_greedy():
    """SHORTEST_GREEDY 策略应比单一起点 GREEDY 找到更短的总长度。
    
    注意：这是多起点贪心，不保证全局最优。
    
    构造一个 GREEDY 会"走错路"的例子：
    
        l3 (远)
        |
        l1 -- l2 (近但方向错)
    
    l1: (0,0) -> (1,0)
    l2: (0.9,0) -> (0.9,-1)   # 头端离 l1 尾端近 (0.1)，但会把路径带向 -y 方向
    l3: (2,0) -> (3,0)        # 离 l1 尾端远 (1.0)，但在 +x 方向
    
    GREEDY 从 l1: 选 l2 (距离 0.1) -> 然后到 l3 需要跳很远
    更优: l1 -> l3 -> l2
    """
    l1 = LineString([(0, 0), (1, 0)])
    l2 = LineString([(0.9, 0), (0.9, -1)])   # 近但会带偏
    l3 = LineString([(2, 0), (3, 0)])

    greedy_result = connect_lines([l1, l2, l3], LineConnectStrategy.GREEDY)
    shortest_result = connect_lines([l1, l2, l3], LineConnectStrategy.SHORTEST_GREEDY)
    
    # SHORTEST_GREEDY 应该找到更短（或相等）的结果
    assert shortest_result.length <= greedy_result.length
    
    # 验证确实包含所有原始线段的点
    all_original_coords = {(0, 0), (1, 0), (0.9, 0), (0.9, -1), (2, 0), (3, 0)}
    assert set(shortest_result.coords).issuperset(all_original_coords)

def test_sample_along_line():
    line = LineString([(0, 0), (10, 0)])
    samples = sample_along_line(line, 11)
    assert samples.shape == (11, 2)
    assert samples[0].tolist() == [0.0, 0.0]
    assert samples[-1].tolist() == [10.0, 0.0]
    assert samples[5].tolist() == [5.0, 0.0] # Midpoint

def test_sample_along_line_zero_length():
    line = LineString([(1, 1), (1, 1)])
    samples = sample_along_line(line, 3)
    assert samples.shape == (3, 2)
    assert np.all(samples == np.array([[1.0, 1.0]] * 3))

def test_sample_along_line_zero_samples():
    line = LineString([(0, 0), (1, 1)])
    with pytest.raises(ValueError, match="num_samples 必须为正整数"):
        sample_along_line(line, 0)

def test_sample_along_line_multilinestring():
    """MultiLineString 采样按总长度进行。"""
    # 两条线段，各长 5，总长 10
    mls = MultiLineString([
        [(0, 0), (5, 0)],   # 长度 5
        [(10, 0), (15, 0)], # 长度 5
    ])
    samples = sample_along_line(mls, 3)  # 0%, 50%, 100%
    assert samples.shape == (3, 2)
    # 0% -> (0, 0), 50% -> (5, 0) 第一段末端, 100% -> (15, 0)
    np.testing.assert_allclose(samples[0], [0.0, 0.0])
    np.testing.assert_allclose(samples[1], [5.0, 0.0])
    np.testing.assert_allclose(samples[2], [15.0, 0.0])

def test_sample_points_on_contour(mock_contour_set):
    # Level 20 has one line: (0, 10) -> (10, 10). Length 10.
    samples = sample_points_on_contour(
        mock_contour_set, 
        target_level=20.0, 
        num_samples=3,
        connect_strategy='longest'
    )
    assert samples.shape == (3, 2)
    expected = np.array([[0, 10], [5, 10], [10, 10]])
    np.testing.assert_allclose(samples, expected)

def test_sample_points_on_contour_no_connect(mock_contour_set):
    """connect_strategy=None 时不连接，直接对 MultiLineString 采样。"""
    # Level 10 has 2 segments: (0,0)->(2,2) and (3,3)->(4,4)
    # 总长度: sqrt(8) + sqrt(2) ≈ 2.83 + 1.41 = 4.24
    samples = sample_points_on_contour(
        mock_contour_set,
        target_level=10.0,
        num_samples=3,
        connect_strategy=None,
    )
    assert samples.shape == (3, 2)
    # 第一个点应该在第一条线的起点
    np.testing.assert_allclose(samples[0], [0.0, 0.0])
    # 最后一个点应该在第二条线的终点
    np.testing.assert_allclose(samples[-1], [4.0, 4.0])

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
