import torch
import unittest
# Assuming voronoi_utils is the package name or accessible in PYTHONPATH
from ..geometry_core import (
    ConvexHull, clip_polygon_2d, compute_polygon_area, 
    compute_convex_hull_volume, normalize_weights, EPSILON,
    monotone_chain_2d, monotone_chain_convex_hull_3d,
    clip_polyhedron_3d 
    # _orientation3d_pytorch, _in_circumsphere3d_pytorch # Not tested here directly, but by delaunay_3d tests
)

"""
Unit tests for the `geometry_core.py` module.

This test suite covers:
- 2D Monotone Chain convex hull algorithm (`monotone_chain_2d`).
- The `ConvexHull` class for both 2D and 3D inputs.
- 2D polygon clipping (`clip_polygon_2d`).
- Standalone area and volume computation functions.
- Weight normalization (`normalize_weights`).
- 3D polyhedron clipping (`clip_polyhedron_3d`).
"""

class TestMonotoneChain2D(unittest.TestCase):
    """Tests for the 2D Monotone Chain convex hull algorithm."""

    def test_simple_square_and_internal_point(self):
        """Tests a square with an internal point; hull should be the square."""
        points = torch.tensor([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[0.5,0.5]]) 
        hull_indices, simplices = monotone_chain_2d(points, tol=EPSILON) # Use updated EPSILON
        self.assertEqual(hull_indices.shape[0], 4, "Hull should have 4 vertices for a square.") 
        self.assertEqual(simplices.shape[0], 4, "Square hull should have 4 edges.")
        hull_coords = points[hull_indices]
        expected_corners = torch.tensor([[0.,0.],[1.,0.],[0.,1.],[1.,1.]]) # Order might vary after sort in monotone_chain
        
        # Check that all expected corners are present in the hull_coords
        # Convert to list of tuples for easier comparison if order is not guaranteed
        hull_coords_set = set(tuple(p.tolist()) for p in hull_coords)
        expected_coords_set = set(tuple(p.tolist()) for p in expected_corners)
        self.assertEqual(hull_coords_set, expected_coords_set, "Hull vertices do not match expected square corners.")

    def test_collinear_points(self):
        """Tests collinear points; hull should be the two extreme points."""
        points = torch.tensor([[0.,0.],[1.,1.],[2.,2.],[3.,3.]])
        hull_indices, simplices = monotone_chain_2d(points, tol=EPSILON)
        self.assertEqual(hull_indices.shape[0], 2, "Hull of collinear points should have 2 vertices.") 
        self.assertEqual(simplices.shape[0], 1, "Hull of collinear points should have 1 edge.") 
        # Check if the extreme points (original indices 0 and 3) are part of the hull
        self.assertIn(points[0].tolist(), points[hull_indices].tolist(), "First point missing from collinear hull.")
        self.assertIn(points[3].tolist(), points[hull_indices].tolist(), "Last point missing from collinear hull.")


    def test_single_point(self):
        """Tests hull of a single point."""
        points = torch.tensor([[1.,1.]])
        hull_indices, simplices = monotone_chain_2d(points, tol=EPSILON)
        self.assertEqual(hull_indices.shape[0], 1, "Hull of single point is the point itself.")
        self.assertEqual(simplices.shape[0], 0, "Hull of single point has no edges.")

    def test_two_points(self):
        """Tests hull of two distinct points."""
        points = torch.tensor([[0.,0.], [1.,1.]])
        hull_indices, simplices = monotone_chain_2d(points, tol=EPSILON)
        self.assertEqual(hull_indices.shape[0], 2, "Hull of two points are the points themselves.")
        self.assertEqual(simplices.shape[0], 1, "Hull of two points has one edge.")

class TestConvexHullClass(unittest.TestCase):
    """Tests for the ConvexHull class (2D and 3D)."""

    def test_ch_2d_square_with_internal(self):
        """Tests ConvexHull with a 2D square and an internal point."""
        points = torch.tensor([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[0.5,0.5]])
        hull = ConvexHull(points, tol=EPSILON)
        self.assertEqual(hull.vertices.shape[0], 4, "Square hull should have 4 vertices.")
        self.assertAlmostEqual(hull.area.item(), 1.0, places=6) # Area of unit square
        self.assertEqual(hull.simplices.shape[0], 4, "Square hull should have 4 edges.")

    def test_ch_2d_collinear(self):
        """Tests ConvexHull with 2D collinear points."""
        points = torch.tensor([[0.,0.],[1.,1.],[2.,2.],[3.,3.]])
        hull = ConvexHull(points, tol=EPSILON) 
        self.assertEqual(hull.vertices.shape[0], 2, "Collinear hull should have 2 vertices.") 
        self.assertAlmostEqual(hull.area.item(), 0.0, places=6, msg="Area of collinear points hull should be 0.")
        self.assertEqual(hull.simplices.shape[0],1, "Collinear hull should have 1 edge.")

    def test_ch_3d_tetrahedron(self):
        """Tests ConvexHull with a simple 3D tetrahedron."""
        points = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        hull = ConvexHull(points, tol=EPSILON)
        self.assertEqual(hull.vertices.shape[0], 4, "Tetrahedron hull has 4 vertices.")
        self.assertTrue(hull.simplices is not None and hull.simplices.shape[0] == 4, 
                        f"Tetrahedron hull should have 4 faces, got {hull.simplices.shape[0] if hull.simplices is not None else 'None'}")
        self.assertAlmostEqual(hull.volume.item(), 1/6, places=6, msg="Volume of standard tetrahedron.")
        # Surface area: 3 faces are right triangles area 0.5. One equilateral triangle sqrt(3)/4 * (sqrt(2))^2 = sqrt(3)/2
        # Total SA = 3 * 0.5 + sqrt(3)/2 = 1.5 + sqrt(3)/2 approx 1.5 + 0.866 = 2.366
        expected_sa = 1.5 + (math.sqrt(3.0) / 2.0) 
        self.assertAlmostEqual(hull.area.item(), expected_sa, places=5, msg="Surface area of standard tetrahedron.")

    def test_ch_3d_cube(self):
        """Tests ConvexHull with a 3D unit cube."""
        points = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,0.],[1.,0.,1.],[0.,1.,1.],[1.,1.,1.]])
        hull = ConvexHull(points, tol=EPSILON)
        self.assertEqual(hull.vertices.shape[0], 8, "Cube hull should have 8 vertices.")
        # A cube has 6 faces, each triangulated into 2 triangles by monotone_chain_3d usually, so 12 faces.
        self.assertTrue(hull.simplices is not None and hull.simplices.shape[0] == 12, 
                        f"Cube hull should have 12 triangular faces, got {hull.simplices.shape[0] if hull.simplices is not None else 'None'}") 
        self.assertAlmostEqual(hull.volume.item(), 1.0, places=6, msg="Volume of unit cube.")
        self.assertAlmostEqual(hull.area.item(), 6.0, places=6, msg="Surface area of unit cube.")

    def test_ch_3d_planar_points(self):
        """Tests ConvexHull with 3D points that are coplanar."""
        points = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]]) # Square on XY plane
        hull = ConvexHull(points, tol=EPSILON) 
        # The 3D hull of coplanar points is essentially a 2D polygon in 3D space.
        # The number of vertices should be 4.
        self.assertEqual(hull.vertices.shape[0], 4, "Coplanar 3D hull should identify the 4 vertices.")
        self.assertAlmostEqual(hull.volume.item(), 0.0, places=6, msg="Volume of coplanar points hull should be 0.")
        # Surface area is area of polygon * 2 (top and bottom view of the flat shape)
        # Area of this square is 1.0. So surface area should be 2.0.
        self.assertAlmostEqual(hull.area.item(), 2.0, places=6, msg="Surface area of coplanar square (viewed as flat 3D object).") 
        # Faces: monotone_chain_3d might return 2 faces for a planar shape.
        self.assertTrue(hull.simplices is not None and hull.simplices.shape[0] == 2, 
                        f"Coplanar hull should have 2 faces, got {hull.simplices.shape[0] if hull.simplices is not None else 'None'}")


class TestClipping2D(unittest.TestCase):
    """Tests for 2D polygon clipping (`clip_polygon_2d`)."""

    def test_clip_polygon_2d_fully_inside(self):
        """Tests clipping a polygon that is already fully inside the bounds."""
        polygon = torch.tensor([[0.25,0.25],[0.75,0.25],[0.75,0.75],[0.25,0.75]])
        bounds = torch.tensor([[0.,0.],[1.,1.]])
        clipped = clip_polygon_2d(polygon, bounds)
        self.assertEqual(clipped.shape[0], 4, "Polygon fully inside should remain unchanged.")
        self.assertAlmostEqual(compute_polygon_area(clipped), 0.25, places=5)

    def test_clip_polygon_2d_partial_overlap(self):
        """Tests clipping a polygon that partially overlaps the bounds."""
        polygon = torch.tensor([[0.5,0.5],[1.5,0.5],[1.5,1.5],[0.5,1.5]]) # Square, 1x1, starting at (0.5,0.5)
        bounds = torch.tensor([[0.,0.],[1.,1.]]) # Unit square bounds
        clipped = clip_polygon_2d(polygon, bounds) # Should clip to (0.5,0.5)-(1,0.5)-(1,1)-(0.5,1)
        self.assertTrue(clipped.shape[0]==4, f"Partially overlapping clip resulted in {clipped.shape[0]} vertices, expected 4.")
        self.assertAlmostEqual(compute_polygon_area(clipped), 0.25, places=5, msg="Area of clipped square incorrect.")

    def test_clip_polygon_fully_outside(self):
        """Tests clipping a polygon that is fully outside the bounds."""
        polygon = torch.tensor([[10.,10.],[12.,10.],[12.,12.],[10.,12.]])
        bounds = torch.tensor([[0.,0.],[1.,1.]])
        clipped = clip_polygon_2d(polygon, bounds)
        self.assertEqual(clipped.shape[0], 0, "Polygon fully outside should result in 0 vertices.")

    def test_clip_triangle(self):
        """Tests clipping a triangle against bounds."""
        polygon = torch.tensor([[0.0,0.0],[2.0,0.0],[1.0,2.0]]) # Triangle with area 0.5*2*2 = 2
        bounds = torch.tensor([[0.0,0.0],[1.0,1.0]]) # Clip to unit square
        clipped = clip_polygon_2d(polygon, bounds)
        # Expected clipped shape: (0,0)-(1,0)-(0.5,1)-(0,1) -> quadrilateral
        # Area: (0,0)-(1,0)-(0.5,1) area 0.5. (0,0)-(0.5,1)-(0,1) area 0.25. Total 0.75
        self.assertEqual(clipped.shape[0],4, f"Triangle clip resulted in {clipped.shape[0]} vertices, expected 4.")
        self.assertAlmostEqual(compute_polygon_area(clipped),0.75,places=5, msg="Area of clipped triangle incorrect.")

class TestAreaVolumeFunctions(unittest.TestCase):
    """Tests for standalone area and volume computation functions."""
    def test_cpa_square(self): self.assertAlmostEqual(compute_polygon_area(torch.tensor([[0.,0.],[2.,0.],[2.,2.],[0.,2.]])),4.0,places=6)
    def test_cpa_degenerate_line(self): self.assertAlmostEqual(compute_polygon_area(torch.tensor([[0.,0.],[1.,1.]])),0.0,places=6)
    def test_cpa_collinear_gt_3pts(self): self.assertAlmostEqual(compute_polygon_area(torch.tensor([[0.,0.],[1.,1.],[2.,2.]])),0.0,places=6)
    def test_cchv_cube(self): self.assertAlmostEqual(compute_convex_hull_volume(torch.tensor([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,1.],[1.,1.,1.],[0.,1.,1.]])),1.0,places=6)
    def test_cchv_degenerate_plane(self): self.assertAlmostEqual(compute_convex_hull_volume(torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])),0.0,places=6)
    def test_cchv_coplanar_gt_4pts(self): self.assertAlmostEqual(compute_convex_hull_volume(torch.tensor([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])),0.0,places=6)

class TestNormalizeWeightsFunc(unittest.TestCase):
    """Tests for the normalize_weights function."""
    def test_nw_simple(self):
        w=torch.tensor([1.,2.,3.,4.]); n=normalize_weights(w, tol=EPSILON)
        self.assertTrue(torch.allclose(n,torch.tensor([0.1,0.2,0.3,0.4]))); self.assertAlmostEqual(torch.sum(n).item(),1.0,places=6)
    def test_nw_target_sum(self): self.assertTrue(torch.allclose(normalize_weights(torch.tensor([1.,1.,1.,1.]),target_sum=4.0, tol=EPSILON),torch.tensor([1.,1.,1.,1.])))
    def test_nw_with_zeros(self): self.assertTrue(torch.allclose(normalize_weights(torch.tensor([0.,1.,0.,3.]), tol=EPSILON),torch.tensor([0.,0.25,0.,0.75])))
    def test_nw_all_zeros_returns_uniform(self): # Updated expectation based on new logic
        weights = torch.tensor([0.,0.,0.])
        normalized = normalize_weights(weights, tol=EPSILON)
        self.assertTrue(torch.allclose(normalized, torch.tensor([1/3., 1/3., 1/3.])))
    def test_nw_small_sum_returns_uniform(self): # Updated expectation
        weights = torch.tensor([EPSILON/10,EPSILON/10])
        normalized = normalize_weights(weights,tol=EPSILON)
        self.assertTrue(torch.allclose(normalized, torch.tensor([0.5, 0.5])))
    def test_nw_negative_weights_error(self):
        with self.assertRaisesRegex(ValueError,"Weights must be non-negative"): normalize_weights(torch.tensor([1.,-1.,2.]), tol=EPSILON)
    def test_nw_negative_weights_near_zero_ok(self):
        # Test that small negative values (within tolerance) are clamped and normalized.
        n=normalize_weights(torch.tensor([1.,-EPSILON/2,2.-EPSILON/2]),tol=EPSILON)
        # Expected: [1,0,2] -> [1/3, 0, 2/3]
        self.assertTrue(torch.allclose(n,torch.tensor([1/3.,0.,2/3.]),atol=1e-7))
    def test_nw_empty_tensor(self):
        norm_empty=normalize_weights(torch.empty(0,dtype=torch.float32), tol=EPSILON)
        self.assertEqual(norm_empty.numel(),0); self.assertEqual(norm_empty.dtype,torch.float32)

class TestClipPolyhedron3D(unittest.TestCase):
    """Tests for 3D polyhedron clipping (`clip_polyhedron_3d`)."""
    def test_clip_cube_fully_inside_itself(self):
        """Tests clipping a unit cube against its own bounds."""
        cube_verts = torch.tensor([
            [0.,0.,0.], [1.,0.,0.], [1.,1.,0.], [0.,1.,0.],
            [0.,0.,1.], [1.,0.,1.], [1.,1.,1.], [0.,1.,1.]
        ], dtype=torch.float32)
        bounds = torch.tensor([[0.,0.,0.],[1.,1.,1.]], dtype=torch.float32)
        clipped_verts = clip_polyhedron_3d(cube_verts, bounds, tol=EPSILON)
        
        self.assertIsNotNone(clipped_verts)
        if clipped_verts is not None:
             # The exact number of vertices after clipping can vary if vertices lie on boundary.
             # The key is that the volume should be correct.
            self.assertTrue(clipped_verts.shape[0] >= 4, "Clipped cube should have at least 4 vertices.")
            if clipped_verts.shape[0] >= 4:
                 hull_of_clipped = ConvexHull(clipped_verts, tol=EPSILON)
                 self.assertAlmostEqual(hull_of_clipped.volume.item(), 1.0, places=5, msg="Volume of cube clipped by itself.")

    def test_clip_cube_half_way(self):
        """Tests clipping a unit cube to half its extent along one axis."""
        cube_verts = torch.tensor([
            [0.,0.,0.], [1.,0.,0.], [1.,1.,0.], [0.,1.,0.],
            [0.,0.,1.], [1.,0.,1.], [1.,1.,1.], [0.,1.,1.]
        ], dtype=torch.float32)
        bounds = torch.tensor([[0.,0.,0.],[0.5,1.,1.]], dtype=torch.float32) # Clip along x-axis at x=0.5
        clipped_verts = clip_polyhedron_3d(cube_verts, bounds, tol=EPSILON)
        
        self.assertIsNotNone(clipped_verts)
        if clipped_verts is not None and clipped_verts.shape[0] >=4:
            hull_of_clipped = ConvexHull(clipped_verts, tol=EPSILON)
            self.assertAlmostEqual(hull_of_clipped.volume.item(), 0.5, places=5, msg="Volume of half-clipped cube.")
            # A clean cut of a cube (8 vertices) by a plane results in 4 new vertices on the cut plane,
            # plus the 4 original vertices on the kept side. So, 8 vertices.
            self.assertEqual(hull_of_clipped.vertices.shape[0], 8, "Half-clipped cube should form an 8-vertex polyhedron.")


    def test_clip_poly_fully_outside_3d(self):
        """Tests clipping a polyhedron that is entirely outside the bounds."""
        poly_verts = torch.tensor([[2.,2.,2.], [3.,2.,2.], [3.,3.,2.], [2.,3.,2.],
                                   [2.,2.,3.], [3.,2.,3.], [3.,3.,3.], [2.,3.,3.]], dtype=torch.float32) # Cube from (2,2,2) to (3,3,3)
        bounds = torch.tensor([[0.,0.,0.],[1.,1.,1.]], dtype=torch.float32) # Unit cube bounds
        clipped_verts = clip_polyhedron_3d(poly_verts, bounds, tol=EPSILON)
        self.assertIsNotNone(clipped_verts)
        if clipped_verts is not None:
            # Expect 0 vertices, or if some complex intermediate points are generated but form no hull,
            # then ConvexHull might return <4 points. The key is volume should be 0.
            if clipped_verts.shape[0] >= 4:
                hull_of_clipped = ConvexHull(clipped_verts, tol=EPSILON)
                self.assertAlmostEqual(hull_of_clipped.volume.item(), 0.0, places=5)
            else:
                self.assertTrue(clipped_verts.shape[0] < 4) # Should be 0, or not enough for a volume

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) # Add for direct test running if needed
