# Voronoi Cell Analysis (`voronoi_analysis.py`)

The `voronoi_analysis.py` module provides functions to compute various geometric properties and relationships of Voronoi cells derived from a Voronoi diagram.

## Purpose

Once a Voronoi diagram is constructed (e.g., using `voronoi_from_delaunay.py`), this module allows for quantitative analysis of the individual Voronoi cells. This includes calculating density weights, geometric measures (centroids, perimeters, areas, volumes), shape descriptors (circularity, sphericity), and topological information (neighbors).

## Key Functions

### 1. `compute_voronoi_density_weights`

-   **Signature:** `compute_voronoi_density_weights(points: torch.Tensor, bounds: torch.Tensor | None = None, normalize_weights_flag: bool = True) -> torch.Tensor`
-   **Description:** Computes density compensation weights for a set of input seed points based on the measure (area in 2D, volume in 3D) of their corresponding Voronoi cells.
-   **Parameters:**
    -   `points`: The input seed points `(N, D)`.
    -   `bounds` (optional): A tensor `[[min_coords], [max_coords]]` defining an axis-aligned bounding box. If provided, Voronoi cells are clipped to these bounds before their measure is calculated. This is crucial for handling unbounded cells or focusing the analysis on a specific region.
    -   `normalize_weights_flag`: If `True` (default), the computed raw weights (typically `1/measure`) are normalized to sum to 1.0.
-   **Logic Highlights:**
    -   Performs Delaunay triangulation and Voronoi construction internally using modules from the library.
    -   **Clipping:** If `bounds` are provided, 2D cells are clipped using `clip_polygon_2d` and 3D cells using `clip_polyhedron_3d` (both from `geometry_core.py`).
    -   **Open Cell Handling (No Bounds):** If `bounds` are *not* provided, cells corresponding to points on the convex hull of the input set (which are likely unbounded) are assigned a very large measure, resulting in a small, stable weight. Interior cells are measured directly.
    -   **Measure Calculation:** Area (2D) or volume (3D) of the (potentially clipped) Voronoi cells is computed, typically using the `ConvexHull` class from `geometry_core.py`.
    -   If a cell's measure is zero or cannot be computed (e.g., degenerate after clipping), a very small positive measure is used as a fallback to ensure numerical stability for the weight calculation (`1/measure`).
-   **Returns:** A tensor of weights `(N,)`, one for each input point.

### 2. `compute_cell_centroid`

-   **Signature:** `compute_cell_centroid(cell_vertices_list: list[torch.Tensor]) -> torch.Tensor | None`
-   **Description:** Computes the geometric centroid (average of vertex coordinates) of a single Voronoi cell.
-   **Input:** `cell_vertices_list`: A list of PyTorch Tensors, where each tensor represents a vertex of the cell.
-   **Returns:** A tensor representing the centroid, or `None` if the input is invalid.

### 3. `get_cell_neighbors`

-   **Signature:** `get_cell_neighbors(target_cell_index: int, all_cells_vertices_list: list[list[torch.Tensor]], shared_vertices_threshold: int = 2, tolerance: float = 1e-5) -> list[int]`
-   **Description:** Identifies neighboring Voronoi cells for a given target cell. Primarily designed for 2D where neighbors share an edge (2 vertices).
-   **Logic:** Two cells are considered neighbors if they share at least `shared_vertices_threshold` vertices. Vertex comparison uses coordinate rounding based on the `tolerance` parameter to handle floating-point inaccuracies.
-   **Input:**
    -   `target_cell_index`: Index of the cell to find neighbors for.
    -   `all_cells_vertices_list`: A list where each element is a list of vertex coordinate Tensors for a cell (as returned by `construct_voronoi_polygons_2d`).
    -   `shared_vertices_threshold`: Typically 2 for 2D edge neighbors.
    -   `tolerance`: Used for robust vertex matching.
-   **Returns:** A list of indices of the neighboring cells.

### 4. `compute_cell_perimeter_2d`

-   **Signature:** `compute_cell_perimeter_2d(cell_vertices: torch.Tensor) -> torch.Tensor`
-   **Description:** Calculates the perimeter of a 2D Voronoi cell (polygon).
-   **Input:** `cell_vertices`: A tensor of shape `(N, 2)` containing the **ordered** vertices of the 2D polygon.
-   **Returns:** A scalar tensor representing the perimeter. Returns 0 if the cell has fewer than 2 vertices.

### 5. `compute_cell_surface_area_3d`

-   **Signature:** `compute_cell_surface_area_3d(cell_vertices: torch.Tensor) -> torch.Tensor`
-   **Description:** Calculates the surface area of a 3D Voronoi cell (polyhedron). It computes the surface area of the convex hull of the provided cell vertices.
-   **Input:** `cell_vertices`: A tensor of shape `(N, 3)` containing the vertices of the 3D polyhedron. The order does not strictly matter as `ConvexHull` is used.
-   **Returns:** A scalar tensor representing the surface area. Returns 0 if the cell has fewer than 3 vertices or if hull computation fails. It uses `ConvexHull(cell_vertices).area`.

### 6. `compute_circularity_2d`

-   **Signature:** `compute_circularity_2d(area: torch.Tensor, perimeter: torch.Tensor) -> torch.Tensor`
-   **Description:** Computes the circularity of a 2D shape, a measure of how close it is to a perfect circle (circularity = 1).
-   **Formula:** `(4 * pi * area) / (perimeter**2)`.
-   **Input:** Scalar tensors for `area` and `perimeter`.
-   **Returns:** A scalar tensor for circularity. Handles zero perimeter by returning 0.

### 7. `compute_sphericity_3d`

-   **Signature:** `compute_sphericity_3d(volume: torch.Tensor, surface_area: torch.Tensor) -> torch.Tensor`
-   **Description:** Computes the sphericity of a 3D shape, a measure of how close it is to a perfect sphere (sphericity = 1).
-   **Formula:** `(pi**(1/3) * (6 * volume)**(2/3)) / surface_area`.
-   **Input:** Scalar tensors for `volume` and `surface_area`.
-   **Returns:** A scalar tensor for sphericity. Handles zero surface area by returning 0.

## Usage Example (Conceptual)

```python
import torch
from voronoi_analysis import (compute_voronoi_density_weights, 
                              compute_cell_perimeter_2d, compute_cell_surface_area_3d,
                              compute_circularity_2d, compute_sphericity_3d)
# Assume 'points_2d', 'voronoi_cell_2d_vertices' (a list of ordered vertex Tensors for one cell),
# 'points_3d', 'voronoi_cell_3d_vertices' (a Tensor of vertices for one cell) are defined.
# Assume 'cell_area_2d', 'cell_volume_3d' are computed (e.g. via compute_voronoi_density_weights' internal logic or ConvexHull)

# Density Weights
weights_2d = compute_voronoi_density_weights(points_2d)
# For bounded analysis:
# bounds_2d = torch.tensor([[min_x, min_y], [max_x, max_y]])
# bounded_weights_2d = compute_voronoi_density_weights(points_2d, bounds=bounds_2d)

# 2D Cell Analysis
if len(voronoi_cell_2d_vertices) >= 3:
    cell_poly_tensor = torch.stack(voronoi_cell_2d_vertices)
    perimeter = compute_cell_perimeter_2d(cell_poly_tensor)
    # area = compute_polygon_area(cell_poly_tensor) # from geometry_core, or use ConvexHull
    # circularity = compute_circularity_2d(area, perimeter)
    # print(f"2D Cell Perimeter: {perimeter.item()}, Circularity: {circularity.item()}")

# 3D Cell Analysis
# cell_polyhedron_v_tensor = ... (vertices of a specific 3D Voronoi cell)
# if cell_polyhedron_v_tensor.shape[0] >= 4:
#     surface_area = compute_cell_surface_area_3d(cell_polyhedron_v_tensor)
#     # volume = ConvexHull(cell_polyhedron_v_tensor).volume
#     # sphericity = compute_sphericity_3d(volume, surface_area)
#     # print(f"3D Cell Surface Area: {surface_area.item()}, Sphericity: {sphericity.item()}")
```
This module provides the tools for deeper quantitative understanding of Voronoi diagrams.
