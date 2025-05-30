"""
Computes circumcenters for 2D triangles and 3D tetrahedra.

This module provides functions to calculate the geometric circumcenter of a
set of points defining a simplex (triangle in 2D, tetrahedron in 3D).
The circumcenter is the center of the unique circle (2D) or sphere (3D)
that passes through all vertices of the simplex. These calculations are
fundamental for constructing Voronoi diagrams from Delaunay triangulations,
as Voronoi vertices are located at these circumcenters.

The functions rely on `EPSILON` from `geometry_core.py` (aliased locally as
`EPSILON_GEOMETRY`) for robust handling of floating-point comparisons,
especially in degenerate cases (collinear or coplanar points).
"""
import torch
import unittest

from .geometry_core import EPSILON as EPSILON_GEOMETRY

# --- Circumcenter Calculation Functions ---

def compute_triangle_circumcenter_2d(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor | None:
    """
    Computes the circumcenter of a 2D triangle defined by three points.

    The circumcenter is equidistant from all three vertices. This function uses
    standard algebraic formulas for circumcenter calculation. It checks for
    collinearity of points using `EPSILON_GEOMETRY` (from `geometry_core.EPSILON`);
    if points are collinear, no valid circumcenter exists, and `None` is returned.

    Args:
        p1 (torch.Tensor): Tensor of shape (2,) representing the first vertex.
        p2 (torch.Tensor): Tensor of shape (2,) representing the second vertex.
        p3 (torch.Tensor): Tensor of shape (2,) representing the third vertex.

    Returns:
        torch.Tensor | None: 
            Coordinates of the circumcenter as a tensor of shape (2,).
            Returns `None` if the points are collinear (degenerate triangle) within
            the tolerance `EPSILON_GEOMETRY`.
    """
    p1x, p1y = p1[0], p1[1]
    p2x, p2y = p2[0], p2[1]
    p3x, p3y = p3[0], p3[1]

    # For numerical stability, calculations can be done in float64
    # D_f64 = 2 * (p1x_f64 * (p2y_f64 - p3y_f64) + p2x_f64 * (p3y_f64 - p1y_f64) + p3x_f64 * (p1y_f64 - p2y_f64))
    # Using input dtype directly after ensuring they are float for the operations:
    points_float = torch.stack([p1, p2, p3]).to(torch.float64) # Promote for precision
    p1x_f, p1y_f = points_float[0,0], points_float[0,1]
    p2x_f, p2y_f = points_float[1,0], points_float[1,1]
    p3x_f, p3y_f = points_float[2,0], points_float[2,1]


    D_val = 2 * (p1x_f * (p2y_f - p3y_f) + 
                 p2x_f * (p3y_f - p1y_f) + 
                 p3x_f * (p1y_f - p2y_f))

    if torch.abs(D_val) < EPSILON_GEOMETRY: 
        return None 

    p1_sq = p1x_f**2 + p1y_f**2
    p2_sq = p2x_f**2 + p2y_f**2
    p3_sq = p3x_f**2 + p3y_f**2

    Ux = (p1_sq * (p2y_f - p3y_f) + p2_sq * (p3y_f - p1y_f) + p3_sq * (p1y_f - p2y_f)) / D_val
    Uy = (p1_sq * (p3x_f - p2x_f) + p2_sq * (p1x_f - p3x_f) + p3_sq * (p2x_f - p1x_f)) / D_val
    
    # Return circumcenter in the original input dtype
    return torch.tensor([Ux, Uy], dtype=p1.dtype, device=p1.device)


def compute_tetrahedron_circumcenter_3d(p1: torch.Tensor, p2: torch.Tensor, 
                                        p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor | None:
    """
    Computes the circumcenter of a 3D tetrahedron defined by four points.

    The circumcenter is equidistant from all four vertices. This function uses a
    system of linear equations derived from the property that the circumcenter
    lies on the perpendicular bisector planes of the tetrahedron's edges.
    It checks for coplanarity of points using a volume test with `EPSILON_GEOMETRY`
    (from `geometry_core.EPSILON`); if points are coplanar, no valid 3D circumcenter
    (for a non-degenerate tetrahedron) exists, and `None` is returned.

    Args:
        p1 (torch.Tensor): Tensor of shape (3,) representing the first vertex.
        p2 (torch.Tensor): Tensor of shape (3,) representing the second vertex.
        p3 (torch.Tensor): Tensor of shape (3,) representing the third vertex.
        p4 (torch.Tensor): Tensor of shape (3,) representing the fourth vertex.

    Returns:
        torch.Tensor | None: 
            Coordinates of the circumcenter as a tensor of shape (3,).
            Returns `None` if the points are coplanar (degenerate tetrahedron)
            within the tolerance `EPSILON_GEOMETRY`, or if the linear system
            for solving is singular.
    """
    points_f64 = torch.stack([p1, p2, p3, p4]).to(torch.float64) # Promote to float64 for precision

    # Check for coplanarity using signed volume of the tetrahedron.
    # Volume = 1/6 * det([p2-p1; p3-p1; p4-p1])
    # If volume is close to zero, points are coplanar.
    v10 = points_f64[1] - points_f64[0]
    v20 = points_f64[2] - points_f64[0]
    v30 = points_f64[3] - points_f64[0]
    
    volume_signed_x6 = torch.det(torch.stack([v10, v20, v30], dim=0)) 

    # Using a slightly larger tolerance for volume check as it involves products
    if torch.abs(volume_signed_x6) < EPSILON_GEOMETRY * 100: # (EPSILON is already small, scale factor can be debated)
        return None

    # Setup linear system: A * x = B, where x = [xc, yc, zc, k]^T
    # (x-x_i)^2 + (y-y_i)^2 + (z-z_i)^2 = R^2 for i=1,2,3,4
    # Subtracting equation for i=1 from i=2,3,4 gives 3 linear equations in xc,yc,zc.
    # A more common matrix form uses:
    # A_matrix (4x4) * [xc, yc, zc, c]^T = B_vector (4x1) where c is related to R^2 - xc^2 - yc^2 - zc^2
    # Row_i of A: [2*xi, 2*yi, 2*zi, 1]
    # Element_i of B: [xi^2 + yi^2 + zi^2]
    A_matrix = torch.empty((4, 4), dtype=torch.float64, device=p1.device)
    B_vector = torch.empty((4, 1), dtype=torch.float64, device=p1.device) # Make B_vector a column vector

    for i in range(4):
        pt = points_f64[i]
        A_matrix[i, 0] = 2 * pt[0]
        A_matrix[i, 1] = 2 * pt[1]
        A_matrix[i, 2] = 2 * pt[2]
        A_matrix[i, 3] = 1.0
        B_vector[i, 0] = pt[0]**2 + pt[1]**2 + pt[2]**2 # Assign to column vector
    
    try:
        # Check determinant of A_matrix before solving for robustness,
        # although the volume check should largely prevent singular A for non-degenerate tetrahedra.
        if torch.abs(torch.det(A_matrix)) < EPSILON_GEOMETRY: # Check for singularity
             return None
        solution = torch.linalg.solve(A_matrix, B_vector)
        circumcenter_f64 = solution[:3].squeeze() # First 3 elements are xc, yc, zc
        return circumcenter_f64.to(dtype=p1.dtype) # Cast back to original input dtype
    except Exception: # Catch potential errors from torch.linalg.solve (e.g., singular matrix)
        return None 

# --- Unit Tests (embedded in this file) ---
# Note: Typically, tests are in a separate tests/ directory. Keeping them here as per original structure.
class TestCircumcenterCalculations(unittest.TestCase):
    """Tests for circumcenter calculation functions."""
    def test_circumcenter_2d_right_triangle(self):
        """Tests 2D circumcenter for a right-angled triangle."""
        p1 = torch.tensor([0.,0.])
        p2 = torch.tensor([2.,0.])
        p3 = torch.tensor([0.,2.])
        center = compute_triangle_circumcenter_2d(p1,p2,p3)
        self.assertIsNotNone(center, "Center should not be None for a right triangle")
        if center is not None:
            self.assertTrue(torch.allclose(center, torch.tensor([1.,1.])))

    def test_circumcenter_2d_equilateral_triangle(self):
        """Tests 2D circumcenter for an equilateral triangle."""
        p1 = torch.tensor([0.,0.], dtype=torch.float32)
        p2 = torch.tensor([2.,0.], dtype=torch.float32)
        p3 = torch.tensor([1., torch.sqrt(torch.tensor(3.0))], dtype=torch.float32)
        center = compute_triangle_circumcenter_2d(p1,p2,p3)
        self.assertIsNotNone(center, "Center should not be None for an equilateral triangle")
        if center is not None:
            expected_center = torch.tensor([1.0, 1.0/torch.sqrt(torch.tensor(3.0))], dtype=torch.float32)
            self.assertTrue(torch.allclose(center, expected_center, atol=1e-6))

    def test_circumcenter_2d_collinear(self):
        """Tests 2D circumcenter for collinear points; expects None."""
        p1 = torch.tensor([0.,0.])
        p2 = torch.tensor([1.,1.])
        p3 = torch.tensor([2.,2.])
        center = compute_triangle_circumcenter_2d(p1,p2,p3)
        self.assertIsNone(center, "Center should be None for collinear points")

    def test_circumcenter_3d_simple_tetrahedron(self):
        """Tests 3D circumcenter for a simple tetrahedron at origin."""
        p1 = torch.tensor([0.,0.,0.])
        p2 = torch.tensor([1.,0.,0.])
        p3 = torch.tensor([0.,1.,0.])
        p4 = torch.tensor([0.,0.,1.])
        center = compute_tetrahedron_circumcenter_3d(p1,p2,p3,p4)
        self.assertIsNotNone(center, "Center should not be None for a simple tetrahedron")
        if center is not None:
            self.assertTrue(torch.allclose(center, torch.tensor([0.5,0.5,0.5])))
    
    def test_circumcenter_3d_regular_tetrahedron_origin_centered(self):
        """Tests 3D circumcenter for a regular tetrahedron expected to be centered at origin."""
        p1 = torch.tensor([1., 1., 1.]) 
        p2 = torch.tensor([1.,-1.,-1.])
        p3 = torch.tensor([-1.,1.,-1.])
        p4 = torch.tensor([-1.,-1.,1.])
        center = compute_tetrahedron_circumcenter_3d(p1,p2,p3,p4)
        self.assertIsNotNone(center, "Center should not be None for this regular tetrahedron")
        if center is not None:
             self.assertTrue(torch.allclose(center, torch.tensor([0.,0.,0.]), atol=1e-6))

    def test_circumcenter_3d_coplanar_points(self):
        """Tests 3D circumcenter for coplanar points; expects None."""
        p1 = torch.tensor([0.,0.,0.])
        p2 = torch.tensor([1.,0.,0.])
        p3 = torch.tensor([0.,1.,0.])
        p4 = torch.tensor([1.,1.,0.]) # All points on z=0 plane
        center = compute_tetrahedron_circumcenter_3d(p1,p2,p3,p4)
        self.assertIsNone(center, "Center should be None for coplanar points")
        
    def test_circumcenter_3d_another_coplanar_set(self):
        """Tests another set of coplanar points, including some collinear ones; expects None."""
        p1 = torch.tensor([0.0, 0.0, 0.0])
        p2 = torch.tensor([1.0, 0.0, 0.0])
        p3 = torch.tensor([2.0, 0.0, 0.0]) # p1,p2,p3 are collinear
        p4 = torch.tensor([0.0, 1.0, 0.0]) # p4 makes them coplanar but not all collinear
        center = compute_tetrahedron_circumcenter_3d(p1,p2,p3,p4)
        self.assertIsNone(center, "Center should be None for this coplanar/collinear set")

# if __name__ == '__main__':
#    # This allows running tests when the script is executed directly.
#    # For library use, tests are typically run by a test runner from the tests/ directory.
#    unittest.main(argv=['first-arg-is-ignored'], exit=False)
