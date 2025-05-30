"""
Unit tests for Voronoi cell analysis functions in `voronoi_analysis.py`.

This module tests:
- `compute_voronoi_density_weights`: Calculation of density weights, including
  handling of bounds, normalization, and edge cases for point counts.
- `compute_cell_centroid`: Centroid calculation for Voronoi cells.
- `get_cell_neighbors`: Identification of neighboring cells based on shared vertices.
- `compute_cell_perimeter_2d`: Perimeter of 2D Voronoi cells.
- `compute_cell_surface_area_3d`: Surface area of 3D Voronoi cells.
- `compute_circularity_2d`: Circularity shape factor for 2D cells.
- `compute_sphericity_3d`: Sphericity shape factor for 3D cells.
"""
import torch
import unittest
import math # For math.pi, math.sqrt in tests
from collections import defaultdict # For context, not directly used in all tests here

from ..voronoi_analysis import (
    compute_voronoi_density_weights,
    compute_cell_centroid,
    get_cell_neighbors,
    compute_cell_perimeter_2d,
    compute_cell_surface_area_3d,
    compute_circularity_2d,
    compute_sphericity_3d
)
from ..geometry_core import EPSILON, ConvexHull # ConvexHull for creating test inputs if needed

class TestVoronoiDensityWeights(unittest.TestCase):
    """
    Tests for the `compute_voronoi_density_weights` function.
    Focuses on weight calculation, normalization, and handling of various
    input scenarios like bounds and insufficient points.
    """
    def test_simple_2d_square(self):
        """Tests density weights for 4 points forming a square in 2D."""
        points = torch.tensor([
            [0.,0.], [1.,0.], [1.,1.], [0.,1.]
        ], dtype=torch.float32)
        weights = compute_voronoi_density_weights(points)
        self.assertEqual(weights.shape, (points.shape[0],), "Weights tensor shape mismatch.")
        self.assertTrue(torch.allclose(torch.sum(weights), torch.tensor(1.0)), 
                        f"Weights do not sum to 1.0. Sum: {torch.sum(weights)}")
        # For a square, current fallback for unclipped cells with <3 Voronoi vertices per cell
        # (which is the case here as only 1 Voronoi vertex is generated at the center)
        # leads to equal weights after normalization.
        self.assertTrue(torch.allclose(weights, torch.tensor([0.25, 0.25, 0.25, 0.25])),
                        f"Weights for a square are not equal or not 1/N: {weights}")

    def test_simple_3d_tetrahedron_with_center(self):
        """Tests density weights for 5 points (tetrahedron + center) in 3D."""
        p0 = torch.tensor([1., 1., 1.]); p1 = torch.tensor([1., -1., -1.])
        p2 = torch.tensor([-1., 1., -1.]); p3 = torch.tensor([-1., -1., 1.])
        p4_center = (p0 + p1 + p2 + p3) / 4.0 
        points = torch.stack([p0, p1, p2, p3, p4_center], dim=0).float()
        
        weights = compute_voronoi_density_weights(points, normalize_weights_flag=True) # Explicitly normalize
        
        self.assertEqual(weights.shape, (points.shape[0],), "Weights tensor shape mismatch.")
        self.assertTrue(torch.allclose(torch.sum(weights), torch.tensor(1.0), atol=EPSILON),
                        f"Weights do not sum to 1.0. Sum: {torch.sum(weights)}")
        
        # Outer points (0-3) are on the convex hull, their Voronoi cells are unbounded.
        # The center point's (4) Voronoi cell is bounded.
        # Unbounded cells get a very large measure -> small raw weight.
        # Bounded cell gets a finite measure -> larger raw weight.
        # After normalization, the center point should have a significantly larger weight.
        weight_outer_avg = torch.mean(weights[:4])
        weight_center = weights[4]
        self.assertTrue(weight_center > weight_outer_avg * 2, # Heuristic: center weight should be much larger
                        f"Center point weight {weight_center.item()} not substantially larger than outer avg {weight_outer_avg.item()}")
        # Symmetric outer points should have similar weights to each other.
        self.assertTrue(torch.allclose(weights[0], weights[1], atol=1e-5))
        self.assertTrue(torch.allclose(weights[1], weights[2], atol=1e-5))
        self.assertTrue(torch.allclose(weights[2], weights[3], atol=1e-5))


    def test_normalization_flag_false_2d(self):
        """Tests `normalize_weights_flag=False` for a 2D case."""
        points = torch.tensor([[0.,0.], [2.,0.], [1.,1.5]], dtype=torch.float32) # A simple triangle
        
        # With normalize_weights_flag=False, weights are 1/area.
        # These are convex hull points, so without bounds, area is set to open_cell_measure_without_bounds.
        # open_cell_measure_without_bounds = 1.0 / (EPSILON**2)
        # So, raw weights should be EPSILON**2.
        weights_raw = compute_voronoi_density_weights(points, normalize_weights_flag=False)
        expected_raw_weight = EPSILON**2
        
        self.assertTrue(torch.allclose(weights_raw, torch.tensor([expected_raw_weight]*3, dtype=torch.float32)),
                        f"Raw weights incorrect: {weights_raw}, expected approx {expected_raw_weight}")
        sum_raw = torch.sum(weights_raw)
        self.assertTrue(sum_raw > 0, "Sum of raw weights should be positive.")
        if abs(sum_raw.item() - 1.0) > EPSILON:
             self.assertNotAlmostEqual(sum_raw.item(), 1.0, msg="Sum of raw weights for this case should not be 1.0.")
        self.assertTrue(torch.all(weights_raw > 0), "All raw weights should be positive.")

    def test_edge_case_too_few_points_2d(self):
        """Tests density weights with 1 or 2 points in 2D (expects uniform weights)."""
        points_1 = torch.tensor([[0.,0.]], dtype=torch.float32)
        weights_1 = compute_voronoi_density_weights(points_1)
        self.assertTrue(torch.allclose(weights_1, torch.tensor([1.0])))
        
        points_2 = torch.tensor([[0.,0.], [1.,1.]], dtype=torch.float32)
        weights_2 = compute_voronoi_density_weights(points_2) # Delaunay is empty, fallback to uniform
        self.assertTrue(torch.allclose(weights_2, torch.tensor([0.5, 0.5])))

    def test_edge_case_too_few_points_3d(self):
        """Tests density weights with 1, 2 or 3 points in 3D (expects uniform weights)."""
        points_1 = torch.tensor([[0.,0.,0.]], dtype=torch.float32)
        weights_1 = compute_voronoi_density_weights(points_1)
        self.assertTrue(torch.allclose(weights_1, torch.tensor([1.0])))

        points_2 = torch.tensor([[0.,0.,0.], [1.,1.,1.]], dtype=torch.float32)
        weights_2 = compute_voronoi_density_weights(points_2)
        self.assertTrue(torch.allclose(weights_2, torch.tensor([0.5, 0.5])))
        
        points_3 = torch.tensor([[0.,0.,0.], [1.,1.,1.], [1.,0.,0.]], dtype=torch.float32)
        weights_3 = compute_voronoi_density_weights(points_3)
        self.assertTrue(torch.allclose(weights_3, torch.tensor([1/3., 1/3., 1/3.]), atol=1e-6))

class TestComputeCellCentroid(unittest.TestCase):
    """Tests for `compute_cell_centroid` function."""
    def test_centroid_empty_list(self):
        """Tests centroid calculation with an empty list of vertices."""
        self.assertIsNone(compute_cell_centroid([]))
    def test_centroid_single_point_2d(self):
        """Tests centroid of a single 2D point (should be the point itself)."""
        p1 = torch.tensor([1.0, 2.0]); centroid = compute_cell_centroid([p1])
        self.assertTrue(torch.allclose(centroid, p1))
    def test_centroid_triangle_2d(self):
        """Tests centroid of a 2D triangle."""
        p1=torch.tensor([0.,0.]);p2=torch.tensor([3.,0.]);p3=torch.tensor([0.,3.])
        centroid = compute_cell_centroid([p1,p2,p3])
        self.assertTrue(torch.allclose(centroid, torch.tensor([1.,1.])))

class TestGetCellNeighbors(unittest.TestCase):
    """Tests for `get_cell_neighbors` function (primarily 2D focus)."""
    def _create_square_cell_list_of_tensors(self, x_offset, y_offset, size=1.0):
        """Helper to create a list of vertex coordinate tensors for a square cell."""
        return [
            torch.tensor([x_offset, y_offset]), torch.tensor([x_offset + size, y_offset]),
            torch.tensor([x_offset + size, y_offset + size]), torch.tensor([x_offset, y_offset + size])
        ]
    def test_simple_grid_neighbors(self):
        """Tests neighbor finding for a simple 3x1 grid of square cells."""
        all_cells = [ self._create_square_cell_list_of_tensors(float(i),0.0) for i in range(3) ]
        self.assertEqual(sorted(get_cell_neighbors(0, all_cells)), [1])
        self.assertEqual(sorted(get_cell_neighbors(1, all_cells)), [0,2])
        self.assertEqual(sorted(get_cell_neighbors(2, all_cells)), [1])

class TestVoronoiCellAnalysis(unittest.TestCase):
    """Tests for new cell property computation functions (perimeter, area, shape factors)."""

    def test_compute_cell_perimeter_2d(self):
        """Tests perimeter calculation for various 2D cells."""
        square_verts = torch.tensor([[0,0],[1,0],[1,1],[0,1]], dtype=torch.float32)
        self.assertAlmostEqual(compute_cell_perimeter_2d(square_verts).item(), 4.0, msg="Perimeter of unit square.")
        
        triangle_verts = torch.tensor([[0,0],[3,0],[0,4]], dtype=torch.float32) # 3-4-5 triangle
        self.assertAlmostEqual(compute_cell_perimeter_2d(triangle_verts).item(), 12.0, msg="Perimeter of 3-4-5 triangle.")

        line_verts = torch.tensor([[0,0],[1,0]], dtype=torch.float32) # A line segment
        self.assertAlmostEqual(compute_cell_perimeter_2d(line_verts).item(), 2.0, msg="Perimeter of a line (length * 2).")

        single_point = torch.tensor([[0,0]], dtype=torch.float32)
        self.assertAlmostEqual(compute_cell_perimeter_2d(single_point).item(), 0.0, msg="Perimeter of single point.")
        
        empty_verts = torch.empty((0,2), dtype=torch.float32)
        self.assertAlmostEqual(compute_cell_perimeter_2d(empty_verts).item(), 0.0, msg="Perimeter of empty vertex set.")

    def test_compute_cell_surface_area_3d(self):
        """Tests 3D surface area calculation using ConvexHull."""
        cube_verts = torch.tensor([
            [0,0,0],[1,0,0],[1,1,0],[0,1,0],
            [0,0,1],[1,0,1],[1,1,1],[0,1,1]
        ], dtype=torch.float32)
        self.assertAlmostEqual(compute_cell_surface_area_3d(cube_verts).item(), 6.0, places=5, msg="Surface area of unit cube.")

        p0=torch.tensor([1.,1.,1.]); p1=torch.tensor([1.,-1.,-1.]); 
        p2=torch.tensor([-1.,1.,-1.]); p3=torch.tensor([-1.,-1.,1.])
        tet_verts = torch.stack([p0,p1,p2,p3]).float()
        expected_surf_area = 8 * math.sqrt(3) # SA of regular tetrahedron with side length sqrt(8)
        self.assertAlmostEqual(compute_cell_surface_area_3d(tet_verts).item(), expected_surf_area, places=5, msg="Surface area of regular tetrahedron.")

        planar_verts = torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=torch.float32) 
        # ConvexHull.area for 3D points that are coplanar returns 2 * area of the 2D polygon.
        self.assertAlmostEqual(compute_cell_surface_area_3d(planar_verts).item(), 2.0, places=5, 
                                msg="Surface area of coplanar points (flat 3D hull) should be 2 * 2D area.")
        
        less_than_3_verts = torch.tensor([[0,0,0],[1,0,0]], dtype=torch.float32)
        self.assertAlmostEqual(compute_cell_surface_area_3d(less_than_3_verts).item(), 0.0, msg="SA for <3 points.")
        
        empty_verts = torch.empty((0,3), dtype=torch.float32)
        self.assertAlmostEqual(compute_cell_surface_area_3d(empty_verts).item(), 0.0, msg="SA for empty vertex set.")


    def test_compute_circularity_2d(self):
        """Tests 2D circularity calculation."""
        circle_area = torch.tensor(math.pi); circle_perimeter = torch.tensor(2 * math.pi) # r=1
        self.assertAlmostEqual(compute_circularity_2d(circle_area, circle_perimeter).item(), 1.0, msg="Circularity of perfect circle.")

        square_area = torch.tensor(1.0); square_perimeter = torch.tensor(4.0) # s=1
        self.assertAlmostEqual(compute_circularity_2d(square_area, square_perimeter).item(), math.pi / 4.0, msg="Circularity of square.")

        self.assertAlmostEqual(compute_circularity_2d(torch.tensor(1.0), torch.tensor(0.0)).item(), 0.0, msg="Circularity with zero perimeter.")
        self.assertAlmostEqual(compute_circularity_2d(torch.tensor(0.0), torch.tensor(0.0)).item(), 0.0, msg="Circularity with zero area and perimeter.")

    def test_compute_sphericity_3d(self):
        """Tests 3D sphericity calculation."""
        sphere_volume = torch.tensor((4/3) * math.pi); sphere_surface_area = torch.tensor(4 * math.pi) # r=1
        self.assertAlmostEqual(compute_sphericity_3d(sphere_volume, sphere_surface_area).item(), 1.0, places=5, msg="Sphericity of perfect sphere.")

        cube_volume = torch.tensor(1.0); cube_surface_area = torch.tensor(6.0) # s=1
        expected_sphericity_cube = (math.pi / 6.0)**(1/3)
        self.assertAlmostEqual(compute_sphericity_3d(cube_volume, cube_surface_area).item(), expected_sphericity_cube, places=5, msg="Sphericity of cube.")

        self.assertAlmostEqual(compute_sphericity_3d(torch.tensor(1.0), torch.tensor(0.0)).item(), 0.0, msg="Sphericity with zero surface area.")
        self.assertAlmostEqual(compute_sphericity_3d(torch.tensor(0.0), torch.tensor(0.0)).item(), 0.0, msg="Sphericity with zero volume and surface area.")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
