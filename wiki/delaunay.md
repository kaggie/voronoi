# Delaunay Triangulation

This document outlines the Delaunay triangulation functionalities within the library, primarily handled by `delaunay_2d.py` and `delaunay_3d.py`.

## Core Concepts

Delaunay triangulation is a fundamental algorithm in computational geometry. For a given set P of discrete points in a plane, a Delaunay triangulation DT(P) is a triangulation such that no point in P is inside the circumcircle of any triangle in DT(P). This property maximizes the minimum angle of all the angles of the triangles in the triangulation, tending to avoid skinny triangles. The concept extends to 3D (tetrahedralization) where no point is inside the circumsphere of any tetrahedron.

Both 2D and 3D triangulation modules rely on geometric primitives and the `EPSILON` constant defined in `geometry_core.py` for robust calculations.

## 2D Delaunay Triangulation (`delaunay_2d.py`)

-   **Module:** `delaunay_2d.py`
-   **Main Function:** `delaunay_triangulation_2d(points: torch.Tensor) -> torch.Tensor`
-   **Description:** Computes the 2D Delaunay triangulation of a set of input points using the Bowyer-Watson algorithm.
-   **Input:**
    -   `points`: A PyTorch tensor of shape `(N, 2)` representing N points in 2D.
-   **Output:**
    -   A PyTorch tensor of shape `(M, 3)` representing M Delaunay triangles. Each row contains the original indices of the three points forming a triangle. Returns an empty tensor `(0,3)` if N < 3.
-   **Algorithm Notes:**
    -   Uses a super-triangle to initialize the triangulation.
    -   Incrementally inserts points. For each new point, it identifies "bad" triangles (those whose circumcircles contain the new point). These bad triangles are removed, forming a polygonal cavity.
    -   The cavity is then re-triangulated by connecting the new point to the edges of the polygonal cavity.
    -   Finally, triangles connected to the vertices of the super-triangle are removed from the triangulation.
    -   Helper functions like `get_triangle_circumcircle_details_2d` and `is_point_in_circumcircle` (which uses `EPSILON` from `geometry_core.py`) are used.
-   **Testing:** Unit tests in `tests/test_delaunay_2d.py` cover various scenarios, including:
    -   Basic configurations (e.g., single triangle, square).
    -   Degenerate cases (collinear points, duplicate points, too few points).
    -   Regular point sets (grids).
    -   Randomly generated point sets.
    -   Numerically sensitive scenarios (nearly collinear points, points very close together).

## 3D Delaunay Triangulation (`delaunay_3d.py`)

-   **Module:** `delaunay_3d.py`
-   **Main Function:** `delaunay_triangulation_3d(points: torch.Tensor, tol: float = EPSILON) -> torch.Tensor`
    (Note: `tol` defaults to `EPSILON` from `geometry_core.py`)
-   **Description:** Computes the 3D Delaunay tetrahedralization of a set of input points, also based on an incremental insertion algorithm similar to Bowyer-Watson.
-   **Input:**
    -   `points`: A PyTorch tensor of shape `(N, 3)` representing N points in 3D.
    -   `tol` (optional): Tolerance for geometric predicate calculations. Defaults to `EPSILON` from `geometry_core.py`.
-   **Output:**
    -   A PyTorch tensor of shape `(M, 4)` representing M Delaunay tetrahedra. Each row contains the original indices of the four points forming a tetrahedron. Returns an empty tensor `(0,4)` if N < 4.
-   **Algorithm Notes:**
    -   Initializes with a super-tetrahedron enclosing all input points.
    -   Points are added incrementally. For each point, tetrahedra whose circumspheres contain the point are identified and removed, forming a cavity.
    -   The cavity is re-triangulated by connecting the new point to the boundary faces of the cavity.
    -   Core geometric predicates `_orientation3d_pytorch` and `_in_circumsphere3d_pytorch` (now residing in and imported from `geometry_core.py`) are used with the specified tolerance.
    -   Super-tetrahedron elements are removed at the end to yield the final tetrahedralization of the input points.
-   **Testing:** Unit tests in `tests/test_delaunay_3d.py` cover a range of cases:
    -   Basic configurations (e.g., single tetrahedron, cube).
    -   Degenerate inputs (coplanar points, collinear points, duplicate points, too few points).
    -   Regular point sets (grids, cube).
    -   Randomly generated point sets.
    -   Numerically sensitive configurations.
    -   Checks for consistent orientation of resulting tetrahedra.
