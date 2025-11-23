"""contour_sampling 模块的基础单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
from shapely.geometry import LineString

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LRD_IR_spectrum.plotting.contour_sampling import (  # noqa: E402
    LineMergeStrategy,
    intersect_line_with_polyline,
    merge_lines,
    sample_along_single_line,
    sample_multiple_lines_separately,
)


class ContourSamplingTests(unittest.TestCase):
    def test_arc_length_and_sampling(self) -> None:
        path = LineString([[0.0, 0.0], [3.0, 4.0], [6.0, 4.0]])
        self.assertAlmostEqual(path.length, 8.0, places=6)

        sampled = sample_along_single_line(path, 4)
        self.assertEqual(sampled.shape, (4, 2))
        np.testing.assert_allclose(sampled[0], path.coords[0])
        np.testing.assert_allclose(sampled[-1], path.coords[-1])

    def test_merge_paths_nearest_neighbor(self) -> None:
        path_a = LineString([[0.0, 0.0], [1.0, 0.0]])
        path_b = LineString([[1.0, 0.0], [1.0, 1.0]])
        merged = merge_lines([path_a, path_b], LineMergeStrategy.NEAREST_NEIGHBOR)
        merged_array = np.asarray(merged.coords, dtype=float)
        self.assertEqual(merged_array.shape, (4, 2))
        np.testing.assert_allclose(merged_array[0], path_a.coords[0])
        np.testing.assert_allclose(merged_array[-1], path_b.coords[-1])

    def test_intersect_line_with_polyline(self) -> None:
        diagonal = LineString([[0.0, 0.0], [1.0, 1.0]])
        vertical = np.array([[0.5, 0.0], [0.5, 1.0]], dtype=float)
        points = intersect_line_with_polyline(diagonal, vertical)
        self.assertEqual(points.shape[1], 2)
        self.assertGreaterEqual(points.shape[0], 1)
        np.testing.assert_allclose(points[0], np.array([0.5, 0.5]), atol=1e-8)

    def test_separate_sampling_distribution(self) -> None:
        short = LineString([[0.0, 0.0], [1.0, 0.0]])
        long = LineString([[0.0, 0.0], [0.0, 3.0]])
        samples = sample_multiple_lines_separately([short, long], 10)
        self.assertEqual(samples.shape[1], 2)
        lengths = np.asarray([line.length for line in (short, long)], dtype=float)
        total_length = float(np.sum(lengths))
        expected_total = sum(max(1, int(10 * length / total_length)) for length in lengths)
        self.assertEqual(samples.shape[0], expected_total)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
