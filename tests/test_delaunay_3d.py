"""
Unit tests for 3D Delaunay tetrahedralization and its helper functions.

This module tests functionality related to `delaunay_3d.py`. This includes:
- Geometric predicates `_orientation3d_pytorch` and `_in_circumsphere3d_pytorch`
  (which are defined in `geometry_core.py` but crucial for 3D Delaunay).
- The main 3D Delaunay tetrahedralization algorithm `delaunay_triangulation_3d`.
Tests cover various point configurations: degenerate cases (coplanar, collinear),
regular shapes (cube, grid), random point sets, and numerically sensitive scenarios,
aiming to ensure the robustness and correctness of the 3D triangulation.
"""
import torch
import unittest
import math # For math.sqrt, math.pi, etc.
# Updated imports based on file restructuring
from ..delaunay_3d import delaunay_triangulation_3d 
from ..geometry_core import ConvexHull, _orientation3d_pytorch, _in_circumsphere3d_pytorch, EPSILON

class TestDelaunay3DHelpers(unittest.TestCase):
    """
    Tests for helper functions crucial to 3D Delaunay triangulation,
    specifically geometric predicates like orientation and in-circumsphere tests.
    These predicates are located in `geometry_core.py`.
    """
    def test_orientation3d(self):
        """Tests the _orientation3d_pytorch predicate for various point configurations."""
        p1 = torch.tensor([0.,0.,0.])
        p2 = torch.tensor([1.,0.,0.])
        p3 = torch.tensor([0.,1.,0.])
        p4_pos = torch.tensor([0.,0.,1.]) # Positive orientation (p4 above plane p1-p2-p3)
        p4_neg = torch.tensor([0.,0.,-1.])# Negative orientation (p4 below plane)
        p4_coplanar = torch.tensor([0.5,0.5,0.]) # Coplanar with p1-p2-p3

        self.assertEqual(_orientation3d_pytorch(p1,p2,p3,p4_pos, EPSILON), 1, "Positive orientation failed.")
        self.assertEqual(_orientation3d_pytorch(p1,p2,p3,p4_neg, EPSILON), -1, "Negative orientation failed.")
        self.assertEqual(_orientation3d_pytorch(p1,p2,p3,p4_coplanar, EPSILON), 0, "Coplanar orientation failed.")

    def test_in_circumsphere_tetrahedron(self):
        """Tests the _in_circumsphere3d_pytorch predicate."""
        # Vertices for a reference tetrahedron (example)
        t1 = torch.tensor([1.,0.,-1./math.sqrt(2.0)])
        t2 = torch.tensor([-1.,0.,-1./math.sqrt(2.0)])
        t3 = torch.tensor([0.,1.,1./math.sqrt(2.0)])
        t4 = torch.tensor([0.,-1.,1./math.sqrt(2.0)])
        # This tetrahedron is roughly centered around the origin.
        
        p_inside = torch.tensor([0.1, 0.1, 0.1])   # Expected to be inside
        p_outside = torch.tensor([2., 2., 2.])    # Expected to be outside
        p_on_vertex = t1.clone()                  # A vertex of the tetrahedron itself
        
        # Ensure the test tetrahedron has a non-zero volume (non-degenerate) for the predicate to be meaningful
        # The _in_circumsphere3d_pytorch predicate handles orientation internally.
        self.assertTrue(_orientation3d_pytorch(t1,t2,t3,t4, EPSILON) != 0, "Test tetrahedron is degenerate.")

        self.assertTrue(_in_circumsphere3d_pytorch(p_inside, t1,t2,t3,t4, EPSILON), "Point expected inside circumsphere.")
        self.assertFalse(_in_circumsphere3d_pytorch(p_outside, t1,t2,t3,t4, EPSILON), "Point expected outside circumsphere.")
        # A point on a vertex of the tetrahedron is on the boundary of the circumsphere, not strictly inside.
        self.assertFalse(_in_circumsphere3d_pytorch(p_on_vertex, t1,t2,t3,t4, EPSILON), "Point on vertex should not be strictly inside.") 


class TestDelaunayTriangulation3D(unittest.TestCase):
    """
    Tests for the main 3D Delaunay tetrahedralization function `delaunay_triangulation_3d`.
    """
    def _check_tetrahedra_validity(self, points, tetrahedra, check_orientation=True, expected_orientation=None):
        """Helper to check basic validity of returned tetrahedra and their orientation."""
        n_points = points.shape[0]
        if tetrahedra.numel() == 0:
            self.assertTrue(tetrahedra.shape[1] == 4 if tetrahedra.ndim == 2 else True, "Empty tetrahedra should have 4 columns or be 1D empty.")
            return True
        
        self.assertTrue(torch.all(tetrahedra >= 0) and torch.all(tetrahedra < n_points), 
                        f"Tetrahedra indices out of bounds. Points: {n_points}, Min idx: {torch.min(tetrahedra)}, Max idx: {torch.max(tetrahedra)}")
        
        orientations = []
        for i, tet_indices in enumerate(tetrahedra):
            self.assertEqual(len(set(tet_indices.tolist())), 4, f"Tetrahedron {i} ({tet_indices.tolist()}) has duplicate vertices.")
            if check_orientation:
                p0, p1, p2, p3 = points[tet_indices[0]], points[tet_indices[1]], points[tet_indices[2]], points[tet_indices[3]]
                # Standard orientation check: det([p1-p0, p2-p0, p3-p0])
                # The _orientation3d_pytorch(p0,p1,p2,p3) checks orientation of p3 relative to plane (p0,p1,p2)
                # For a tetrahedron p0,p1,p2,p3, if p0,p1,p2 is CCW from p3, then orientation(p0,p1,p2,p3) > 0.
                # The delaunay_triangulation_3d algorithm aims to make all tetrahedra have positive orientation
                # by swapping vertices of the base face if needed when forming new tetrahedra.
                current_tet_orientation = _orientation3d_pytorch(p0, p1, p2, p3, EPSILON)
                
                if abs(current_tet_orientation) < EPSILON: # Effectively zero volume
                    print(f"Warning: Degenerate tetrahedron {tet_indices.tolist()} found with volume ~0.")
                orientations.append(current_tet_orientation)

        if check_orientation and orientations:
            valid_orientations = [o for o in orientations if abs(o) > EPSILON] # Filter out zero-volume tets for orientation check
            if valid_orientations:
                if expected_orientation is not None:
                     self.assertTrue(all(o == expected_orientation for o in valid_orientations),
                                f"Tetrahedra have inconsistent or unexpected orientations: {orientations}. Expected all {expected_orientation}.")
                else: # Just check for consistency (all same sign)
                     self.assertTrue(all(o > 0 for o in valid_orientations) or all(o < 0 for o in valid_orientations),
                                f"Tetrahedra have mixed orientations: {orientations}.")
        return True

    def test_dt3d_empty_input(self):
        """Tests Delaunay 3D with no input points."""
        points = torch.empty((0,3), dtype=torch.float32)
        tetrahedra = delaunay_triangulation_3d(points)
        self.assertEqual(tetrahedra.shape[0], 0)

    def test_dt3d_less_than_4_points(self):
        """Tests Delaunay 3D with fewer than 4 points."""
        points = torch.tensor([[0.,0.,0.],[1.,1.,1.],[2.,0.,0.]])
        tetrahedra = delaunay_triangulation_3d(points)
        self.assertEqual(tetrahedra.shape[0], 0)
        points_one = torch.tensor([[0.,0.,0.]])
        tetrahedra_one = delaunay_triangulation_3d(points_one)
        self.assertEqual(tetrahedra_one.shape[0], 0)

    def test_dt3d_4_points_coplanar(self):
        """Tests Delaunay 3D with 4 coplanar points (should yield 0 tetrahedra)."""
        points = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
        tetrahedra = delaunay_triangulation_3d(points)
        self.assertEqual(tetrahedra.shape[0], 0, "Coplanar points should not form 3D tetrahedra.")

    def test_dt3d_collinear_points(self):
        """Tests Delaunay 3D with 4 collinear points (should yield 0 tetrahedra)."""
        points = torch.tensor([[0.,0.,0.],[1.,1.,1.],[2.,2.,2.],[3.,3.,3.]])
        tetrahedra = delaunay_triangulation_3d(points)
        self.assertEqual(tetrahedra.shape[0], 0, "Collinear points should not form 3D tetrahedra.")

    def test_dt3d_duplicate_points(self):
        """Tests Delaunay 3D with duplicate input points."""
        points = torch.tensor([
            [0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.], # Forms a tetrahedron
            [0.,0.,0.],[1.,0.,0.] # Duplicates of first two points
        ], dtype=torch.float32)
        tetrahedra = delaunay_triangulation_3d(points)
        # Effective unique points are the first 4.
        self.assertEqual(tetrahedra.shape[0], 1, "Should form one tetrahedron from unique points.")
        self._check_tetrahedra_validity(points, tetrahedra, expected_orientation=1) # Expect positive orientation

    def test_dt3d_single_tetrahedron(self):
        """Tests Delaunay 3D for a single, simple tetrahedron configuration."""
        points = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]], dtype=torch.float32)
        tetrahedra = delaunay_triangulation_3d(points)
        self.assertEqual(tetrahedra.shape[0], 1, "Should form one tetrahedron.") 
        self._check_tetrahedra_validity(points, tetrahedra, expected_orientation=1)
        if tetrahedra.shape[0] == 1:
            self.assertTrue(all(i in tetrahedra[0].tolist() for i in [0,1,2,3]), "Tetrahedron doesn't use all input points.")

    def test_dt3d_cube_points(self):
        """Tests Delaunay 3D for the 8 vertices of a unit cube."""
        points = torch.tensor([
            [0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.],
            [1.,1.,0.], [1.,0.,1.], [0.,1.,1.], [1.,1.,1.]
        ], dtype=torch.float32)
        tetrahedra = delaunay_triangulation_3d(points)
        # Standard triangulations of a cube result in 5 or 6 tetrahedra.
        self.assertTrue(tetrahedra.shape[0] >= 5, f"Expected >= 5 tetrahedra for a cube, got {tetrahedra.shape[0]}.") 
        self._check_tetrahedra_validity(points, tetrahedra, expected_orientation=1) # Assuming positive orientation output
        
        # Verify total volume of tetrahedra equals volume of the cube's convex hull (which is 1.0)
        total_volume_delaunay = 0.0
        if tetrahedra.numel() > 0:
            for tet_indices in tetrahedra:
                tet_points = points[tet_indices]
                # Volume = 1/6 * |det(p1-p0, p2-p0, p3-p0)|
                vol_mat = torch.stack([tet_points[1]-tet_points[0], 
                                       tet_points[2]-tet_points[0], 
                                       tet_points[3]-tet_points[0]], dim=0).to(torch.float64)
                total_volume_delaunay += torch.abs(torch.det(vol_mat)) / 6.0
        
        # ConvexHull class from geometry_core
        cube_hull = ConvexHull(points, tol=EPSILON) 
        self.assertAlmostEqual(total_volume_delaunay, cube_hull.volume.item(), places=5, 
                               msg="Sum of Delaunay tetrahedra volumes should equal cube's convex hull volume.")

    def test_dt3d_regular_grid(self):
        """Tests Delaunay 3D for points on a regular 2x2x2 grid (equivalent to a cube)."""
        points = torch.tensor([[float(i),float(j),float(k)] for i in range(2) for j in range(2) for k in range(2)], dtype=torch.float32)
        tetrahedra = delaunay_triangulation_3d(points)
        self.assertTrue(tetrahedra.shape[0] >= 5, "2x2x2 grid should form at least 5 tetrahedra.") 
        self._check_tetrahedra_validity(points, tetrahedra, expected_orientation=1)

    def test_dt3d_points_very_close(self):
        """Tests Delaunay 3D with some points very close to each other."""
        points = torch.tensor([
            [0.,0.,0.], [EPSILON/10, EPSILON/10, EPSILON/10], # Point very close to origin
            [1.,0.,0.],[0.,1.,0.],[0.,0.,1.]
        ], dtype=torch.float32)
        tetrahedra = delaunay_triangulation_3d(points)
        # Should form at least one tetrahedron (from the 4 distinct points).
        # The close point might or might not alter the triangulation significantly depending on `tol`.
        self.assertTrue(tetrahedra.shape[0] >= 1, "Points very close should still form a valid triangulation.")
        self._check_tetrahedra_validity(points, tetrahedra, expected_orientation=1)

    def test_dt3d_nearly_coplanar_but_3d(self):
        """Tests nearly coplanar points that still form a 3D shape."""
        points = torch.tensor([
            [0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.], # Base plane
            [0.5, 0.5, EPSILON * 10] # Slightly above plane, making it 3D
        ], dtype=torch.float32)
        tetrahedra = delaunay_triangulation_3d(points)
        # This configuration (a flat base quadrilateral and a point slightly above its center)
        # should typically result in 2 or 4 tetrahedra.
        # E.g., if base is split into (0,1,3) and (0,2,3), then (0,1,3,4) and (0,2,3,4).
        self.assertTrue(tetrahedra.shape[0] >= 2, f"Expected >=2 tetrahedra for nearly coplanar points, got {tetrahedra.shape[0]}.")
        self._check_tetrahedra_validity(points, tetrahedra, expected_orientation=1)

    def test_dt3d_random_points(self):
        """Tests Delaunay 3D with randomly generated point sets."""
        for num_pts in [10, 20]: # Reduced from 30 for CI speed
            points = torch.rand((num_pts, 3), dtype=torch.float32) * 100
            tetrahedra = delaunay_triangulation_3d(points)
            self.assertTrue(tetrahedra.shape[0] >= num_pts - 3 if num_pts >=4 else 0,
                            f"Num tetra ({tetrahedra.shape[0]}) for {num_pts} random points is less than lower bound (N-3).")
            self._check_tetrahedra_validity(points, tetrahedra, expected_orientation=1)

            # Optional: Empty circumsphere property check for a few random tetrahedra
            if tetrahedra.shape[0] > 0 and num_pts > 4:
                num_checks = min(2, tetrahedra.shape[0]) # Check a few to save time
                for i in range(num_checks):
                    tet_idx = torch.randint(0, tetrahedra.shape[0], (1,)).item()
                    tet = tetrahedra[tet_idx]
                    p0,p1,p2,p3 = points[tet[0]], points[tet[1]], points[tet[2]], points[tet[3]]
                    
                    is_valid_delaunay_tet = True
                    for pt_idx in range(num_pts):
                        if pt_idx not in tet: # Check all other points
                            if _in_circumsphere3d_pytorch(points[pt_idx], p0, p1, p2, p3, EPSILON):
                                is_valid_delaunay_tet = False
                                # print(f"Delaunay 3D check failed: Point {pt_idx} ({points[pt_idx]}) is in circumsphere of tet {tet.tolist()}")
                                break
                    self.assertTrue(is_valid_delaunay_tet, 
                                    f"Delaunay 3D property violated for tet {tet.tolist()}.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
