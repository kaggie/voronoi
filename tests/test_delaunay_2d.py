"""
Unit tests for 2D Delaunay triangulation and its helper functions.

This module tests the functionality provided in `delaunay_2d.py`, including:
- Calculation of triangle circumcircle details.
- Predicate for checking if a point is inside a triangle's circumcircle.
- The main 2D Delaunay triangulation algorithm (`delaunay_triangulation_2d`).
Tests cover various point configurations, including degenerate cases, regular shapes,
and random point sets, to ensure robustness and correctness.
"""
import torch
import unittest
import math # For math.sqrt in random point test verification (not directly used yet)
# Assuming delaunay_2d is in the parent directory or accessible via PYTHONPATH
from ..delaunay_2d import delaunay_triangulation_2d, get_triangle_circumcircle_details_2d, is_point_in_circumcircle
from ..geometry_core import EPSILON # For numerical comparisons in tests

class TestDelaunay2DHelpers(unittest.TestCase):
    """
    Tests for helper functions used in 2D Delaunay triangulation,
    such as circumcircle calculations and point-in-circumcircle tests.
    """
    def test_circumcircle_right_angle(self):
        """Tests circumcircle calculation for a simple right-angled triangle."""
        p1 = torch.tensor([0.,0.])
        p2 = torch.tensor([2.,0.])
        p3 = torch.tensor([0.,2.])
        center, sq_radius = get_triangle_circumcircle_details_2d(p1,p2,p3)
        self.assertIsNotNone(center, "Circumcenter should be found for a valid triangle.")
        self.assertIsNotNone(sq_radius, "Squared circumradius should be found.")
        if center is not None and sq_radius is not None:
            self.assertTrue(torch.allclose(center, torch.tensor([1.,1.])))
            self.assertAlmostEqual(sq_radius.item(), 2.0)

    def test_circumcircle_equilateral(self):
        """Tests circumcircle calculation for an equilateral triangle."""
        p1 = torch.tensor([0.,0.])
        p2 = torch.tensor([2.,0.])
        p3 = torch.tensor([1., torch.sqrt(torch.tensor(3.0))])
        center, sq_radius = get_triangle_circumcircle_details_2d(p1,p2,p3)
        self.assertIsNotNone(center)
        self.assertIsNotNone(sq_radius)
        if center is not None and sq_radius is not None:
            # Expected center: (1, sqrt(3)/3)
            # Expected radius R = a/sqrt(3) = 2/sqrt(3). R^2 = 4/3
            self.assertTrue(torch.allclose(center, torch.tensor([1.0, 1.0/torch.sqrt(torch.tensor(3.0))])))
            self.assertAlmostEqual(sq_radius.item(), 4.0/3.0, places=5)
        
    def test_circumcircle_collinear(self):
        """Tests circumcircle calculation for collinear points (should return None)."""
        p1 = torch.tensor([0.,0.])
        p2 = torch.tensor([1.,1.])
        p3 = torch.tensor([2.,2.])
        center, sq_radius = get_triangle_circumcircle_details_2d(p1,p2,p3)
        self.assertIsNone(center, "Circumcenter should be None for collinear points.")
        self.assertIsNone(sq_radius, "Squared radius should be None for collinear points.")

    def test_is_point_in_circumcircle_inside(self):
        """Tests `is_point_in_circumcircle` for a point clearly inside."""
        p1, p2, p3 = torch.tensor([0.,0.]), torch.tensor([2.,0.]), torch.tensor([1.,1.]) # Triangle
        point_inside = torch.tensor([1.0, 0.1]) # Point to check
        self.assertTrue(is_point_in_circumcircle(point_inside, p1, p2, p3))

    def test_is_point_in_circumcircle_outside(self):
        """Tests `is_point_in_circumcircle` for a point clearly outside."""
        p1, p2, p3 = torch.tensor([0.,0.]), torch.tensor([2.,0.]), torch.tensor([1.,1.])
        point_outside = torch.tensor([1.0, 2.0]) 
        self.assertFalse(is_point_in_circumcircle(point_outside, p1, p2, p3))

    def test_is_point_in_circumcircle_on_circle(self):
        """Tests `is_point_in_circumcircle` for a point on the circle (should be False for *strictly* inside)."""
        p1, p2, p3 = torch.tensor([0.,0.]), torch.tensor([2.,0.]), torch.tensor([0.,2.])
        point_on_circle = torch.tensor([2.0,2.0]) 
        self.assertFalse(is_point_in_circumcircle(point_on_circle, p1, p2, p3))
        
    def test_is_point_in_circumcircle_collinear_triangle(self):
        """Tests `is_point_in_circumcircle` with collinear triangle vertices (should be False)."""
        p1, p2, p3 = torch.tensor([0.,0.]), torch.tensor([1.,1.]), torch.tensor([2.,2.])
        point = torch.tensor([0.5, 0.5])
        self.assertFalse(is_point_in_circumcircle(point, p1, p2, p3))

class TestDelaunayTriangulation2D(unittest.TestCase):
    """
    Tests for the main 2D Delaunay triangulation function `delaunay_triangulation_2d`.
    Covers various scenarios including edge cases, simple geometric shapes, and random distributions.
    """
    def _check_triangles_validity(self, points, triangles):
        """Helper to check basic validity of returned triangles."""
        n_points = points.shape[0]
        if triangles.numel() == 0:
            self.assertTrue(triangles.shape[1] == 3 if triangles.ndim == 2 else True, "Empty triangles should have 3 columns or be 1D empty.")
            return True 
        
        self.assertTrue(torch.all(triangles >= 0) and torch.all(triangles < n_points), 
                        f"Triangle indices out of bounds. Points: {n_points}, Min idx: {torch.min(triangles)}, Max idx: {torch.max(triangles)}")
        for i, tri in enumerate(triangles):
            self.assertEqual(len(set(tri.tolist())), 3, f"Triangle {i} ({tri.tolist()}) has duplicate vertices.")
        return True

    def test_dt_empty_input(self):
        """Tests Delaunay triangulation with no input points."""
        points = torch.empty((0,2), dtype=torch.float32)
        triangles = delaunay_triangulation_2d(points)
        self.assertEqual(triangles.shape[0], 0)

    def test_dt_less_than_3_points(self):
        """Tests Delaunay triangulation with fewer than 3 points (should produce no triangles)."""
        points_2 = torch.tensor([[0.,0.],[1.,1.]])
        triangles_2 = delaunay_triangulation_2d(points_2)
        self.assertEqual(triangles_2.shape[0], 0)
        
        points_1 = torch.tensor([[0.,0.]])
        triangles_1 = delaunay_triangulation_2d(points_1)
        self.assertEqual(triangles_1.shape[0], 0)

    def test_dt_3_points_collinear(self):
        """Tests Delaunay triangulation with 3 collinear points (should produce no triangles)."""
        points = torch.tensor([[0.,0.],[1.,1.],[2.,2.]])
        triangles = delaunay_triangulation_2d(points)
        self.assertEqual(triangles.shape[0], 0, "Collinear points should not form triangles.")

    def test_dt_nearly_collinear(self):
        """Tests nearly collinear points; should form a valid (skinny) triangle."""
        points = torch.tensor([[0.,0.],[1., EPSILON/10],[2., EPSILON/5]]) # Slightly perturbed from collinear
        triangles = delaunay_triangulation_2d(points)
        self.assertEqual(triangles.shape[0], 1, "Nearly collinear points should form one triangle.")
        self._check_triangles_validity(points, triangles)

    def test_dt_3_points_triangle(self):
        """Tests Delaunay triangulation for a single valid triangle."""
        points = torch.tensor([[0.,0.],[1.,0.],[0.,1.]]) 
        triangles = delaunay_triangulation_2d(points)
        self.assertEqual(triangles.shape[0], 1, "Three non-collinear points should form one triangle.") 
        self._check_triangles_validity(points, triangles)
        if triangles.shape[0] == 1:
             # Check if the triangle uses the 3 input points (indices 0,1,2)
             self.assertTrue(all(i in triangles[0].tolist() for i in [0,1,2]))

    def test_dt_duplicate_points(self):
        """Tests Delaunay triangulation with duplicate input points."""
        # Case 1: Duplicates lead to a collinear unique set
        points_collinear_unique = torch.tensor([[0.,0.],[1.,1.],[0.,0.],[2.,2.],[1.,1.]])
        triangles_collinear = delaunay_triangulation_2d(points_collinear_unique)
        self.assertEqual(triangles_collinear.shape[0], 0, "Duplicates resulting in collinear unique set should be empty.")

        # Case 2: Duplicates where unique points form a triangle
        points_triangle_unique = torch.tensor([[0.,0.],[1.,0.],[0.,1.],[0.,0.],[1.,0.]])
        triangles_triangle = delaunay_triangulation_2d(points_triangle_unique)
        self.assertEqual(triangles_triangle.shape[0], 1, "Duplicates resulting in triangular unique set should form one triangle.")
        self._check_triangles_validity(points_triangle_unique, triangles_triangle)


    def test_dt_4_points_square(self):
        """Tests Delaunay triangulation for 4 points forming a square (expects 2 triangles)."""
        points = torch.tensor([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
        triangles = delaunay_triangulation_2d(points)
        self.assertEqual(triangles.shape[0], 2, "Square should form 2 Delaunay triangles.") 
        self._check_triangles_validity(points, triangles)
        unique_triangle_vertices = torch.unique(triangles.flatten())
        self.assertEqual(len(unique_triangle_vertices), 4, "All 4 points of the square should be part of the triangulation.")


    def test_dt_regular_grid(self):
        """Tests Delaunay triangulation for points on a regular 3x3 grid."""
        points = torch.tensor([[float(i),float(j)] for i in range(3) for j in range(3)], dtype=torch.float32) # 9 points
        triangles = delaunay_triangulation_2d(points)
        # Expected triangles for a 3x3 grid: 2 * (rows-1) * (cols-1) = 2 * 2 * 2 = 8
        self.assertEqual(triangles.shape[0], 8, "3x3 grid should form 8 Delaunay triangles.")
        self._check_triangles_validity(points, triangles)

    def test_dt_points_very_close(self):
        """Tests Delaunay triangulation with some points very close to each other."""
        points = torch.tensor([[0.,0.],[EPSILON/10, EPSILON/10],[1.,0.],[0.,1.]])
        triangles = delaunay_triangulation_2d(points)
        # Expect 2 triangles generally, as (0,0) and (eps,eps) are distinct enough.
        # One triangle from ((0,0), (1,0), (0,1))
        # Another involving (eps,eps) and two of the other three points.
        self.assertTrue(triangles.shape[0] >= 1, "Points very close should still form valid triangulation.") 
        self.assertTrue(triangles.shape[0] <= 2, "Expected 1 or 2 triangles for this close point configuration.")
        self._check_triangles_validity(points, triangles)


    def test_dt_random_points(self):
        """Tests Delaunay triangulation with randomly generated point sets."""
        for num_pts in [10, 30]: # Reduced from 50 for speed if necessary
            points = torch.rand((num_pts, 2), dtype=torch.float32) * 100 # Scale points
            triangles = delaunay_triangulation_2d(points)
            
            self.assertTrue(triangles.shape[0] >= num_pts - 2 if num_pts >=3 else 0, 
                            f"Num triangles ({triangles.shape[0]}) for {num_pts} random points is less than lower bound (N-2).")
            # Max triangles for N points is 2N-5 (for N>=3) if no 3 points are collinear and no 4 are cocircular.
            # For random points, it's unlikely to hit exact maximums or minimums consistently.
            # A more relaxed upper bound might be 2N-2 (includes exterior face edges).
            # The current Bowyer-Watson returns only interior triangles. So 2N-5 is a better theoretical max.
            self.assertTrue(triangles.shape[0] <= 2 * num_pts - 5 if num_pts >=3 else 0,
                             f"Num triangles ({triangles.shape[0]}) for {num_pts} random points exceeds upper bound (2N-5).")
            self._check_triangles_validity(points, triangles)
            
            # Optional: Empty circumcircle property check for a few random triangles
            if triangles.shape[0] > 0 and num_pts > 3:
                num_checks = min(3, triangles.shape[0]) # Check a few to save time
                for i in range(num_checks):
                    tri_idx = torch.randint(0, triangles.shape[0], (1,)).item()
                    tri = triangles[tri_idx]
                    p1, p2, p3 = points[tri[0]], points[tri[1]], points[tri[2]]
                    
                    is_valid_delaunay_tri = True
                    for pt_idx in range(num_pts):
                        if pt_idx not in tri: # Check all other points
                            if is_point_in_circumcircle(points[pt_idx], p1, p2, p3):
                                is_valid_delaunay_tri = False
                                # print(f"Delaunay check failed: Point {pt_idx} ({points[pt_idx]}) is in circumcircle of triangle {tri.tolist()} {p1}, {p2}, {p3}")
                                break
                    self.assertTrue(is_valid_delaunay_tri, 
                                    f"Delaunay property violated for triangle {tri.tolist()} with points {p1}, {p2}, {p3}.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
