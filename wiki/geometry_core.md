# `geometry_core.py`: Core Geometric Primitives

## Purpose

The `geometry_core.py` module is the foundation of the geometric computations within this library. It provides a centralized collection of robust, PyTorch-based functions for fundamental geometric operations and essential constants. By being PyTorch-centric, it aims to leverage GPU acceleration where possible and maintain tensor-based data flows.

This module is designed to be a low-level toolkit used by other modules that implement more complex geometric algorithms like Delaunay triangulation or Voronoi diagram construction.

## Key Functionalities and Primitives

The module includes, but is not limited to, the following key functionalities:

*   **`EPSILON` Constant:**
    *   A global constant (e.g., `1e-7`) used for floating-point comparisons and tolerance in various geometric tests to handle numerical inaccuracies.

*   **Convex Hull Computation:**
    *   **`ConvexHull` Class:** A class that wraps 2D and 3D convex hull computations. It provides properties like `vertices` (indices of points on the hull), `simplices` (edges in 2D, faces in 3D), `area` (polygon area in 2D, surface area in 3D), and `volume` (for 3D hulls).
    *   `monotone_chain_2d(points: torch.Tensor, tol: float)`: Computes the convex hull of 2D points using the Monotone Chain algorithm.
    *   `monotone_chain_convex_hull_3d(points: torch.Tensor, tol: float)`: Computes the convex hull of 3D points (e.g., using an incremental approach).

*   **Polygon and Polyhedron Clipping:**
    *   `clip_polygon_2d(polygon_vertices: torch.Tensor, clip_bounds: torch.Tensor)`: Clips a 2D polygon against an axis-aligned rectangular bounding box using the Sutherland-Hodgman algorithm.
    *   `clip_polyhedron_3d(input_poly_vertices_coords: torch.Tensor, bounding_box_minmax: torch.Tensor, tol: float)`: Clips a 3D convex polyhedron against an axis-aligned bounding box. It collects vertices inside the box and intersections of polyhedron edges with box planes, then computes the convex hull of these points.

*   **Area and Volume Calculations:**
    *   These are primarily accessed via the `ConvexHull` class properties (`hull.area`, `hull.volume`).
    *   `compute_polygon_area(points_coords: torch.Tensor)`: A standalone function (often using `ConvexHull` internally) to compute the area of a 2D polygon.
    *   `compute_convex_hull_volume(points_coords: torch.Tensor)`: A standalone function (often using `ConvexHull` internally) to compute the volume of the convex hull of 3D points.

*   **Geometric Predicates (Primarily for 3D Delaunay):**
    *   `_orientation3d_pytorch(p1, p2, p3, p4, tol)`: Determines the orientation of point `p4` relative to the plane defined by `p1, p2, p3` (e.g., coplanar, positive side, negative side). Crucial for constructing consistently oriented simplices.
    *   `_in_circumsphere3d_pytorch(p_check, t1, t2, t3, t4, tol)`: Checks if point `p_check` is strictly inside the circumsphere of the tetrahedron defined by `t1, t2, t3, t4`. This is a core predicate for the Bowyer-Watson algorithm for Delaunay triangulation.

*   **Other Utilities:**
    *   `normalize_weights(weights: torch.Tensor, ...)`: Normalizes a tensor of weights.
    *   Internal helper functions for clipping (e.g., `_sutherland_hodgman_is_inside`, `_sutherland_hodgman_intersect`, `_point_plane_signed_distance`, `_segment_plane_intersection`).

## PyTorch-Centric Nature

All functions are designed to work with `torch.Tensor` inputs and produce `torch.Tensor` outputs where appropriate. This allows for seamless integration with PyTorch-based workflows and potential acceleration on GPUs, although GPU support might vary depending on the specific operations used within each function (e.g., some complex loop-based logic might still run primarily on CPU unless explicitly parallelized).

## Simple Usage Example (Conceptual)

```python
import torch
from geometry_core import ConvexHull, EPSILON

# Create some 2D points
points_2d = torch.tensor([[0.,0.], [1.,0.], [1.,1.], [0.,1.], [0.5, 0.5]], dtype=torch.float32)

# Compute convex hull
hull_2d = ConvexHull(points_2d, tol=EPSILON)

print("2D Hull Vertices (indices):", hull_2d.vertices)
print("2D Hull Area:", hull_2d.area)

# Create some 3D points
points_3d = torch.tensor([
    [0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.], # A tetrahedron
    [0.5,0.5,0.5] # A point inside
], dtype=torch.float32)

# Compute 3D convex hull (will be the outer tetrahedron)
hull_3d = ConvexHull(points_3d, tol=EPSILON)
print("3D Hull Vertices (indices):", hull_3d.vertices) # Should be indices [0,1,2,3]
print("3D Hull Surface Area:", hull_3d.area)
print("3D Hull Volume:", hull_3d.volume)
```
This module serves as the building block for more advanced geometric algorithms in the library.
