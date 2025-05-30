# Voronoi Diagram Plotting (`voronoi_plotting.py`)

The `voronoi_plotting.py` module offers functions to visualize 2D and 3D Voronoi diagrams using Matplotlib.

## Purpose

Visual inspection is often crucial for understanding and verifying the results of geometric algorithms. This module provides convenient utilities to plot the input seed points and the constructed Voronoi cells.

## Key Functions

### 1. `plot_voronoi_diagram_2d`

-   **Signature:** `plot_voronoi_diagram_2d(points: torch.Tensor, voronoi_cells_verts_list: list[list[torch.Tensor]] | None = None, unique_voronoi_vertices: torch.Tensor | None = None, bounds: torch.Tensor | None = None, show_voronoi_vertices: bool = True, ax=None, title: str = "2D Voronoi Diagram")`
-   **Description:** Plots a 2D Voronoi diagram.
-   **Parameters:**
    -   `points`: The input seed points `(N, 2)`.
    -   `voronoi_cells_verts_list` (optional): Precomputed list where each inner list contains **ordered coordinate tensors** for a Voronoi cell's vertices. If not provided, the function will compute it using `delaunay_triangulation_2d` and `construct_voronoi_polygons_2d`.
    -   `unique_voronoi_vertices` (optional): Precomputed tensor of unique Voronoi vertex coordinates. Used if `show_voronoi_vertices` is True.
    -   `bounds` (optional): A `(2,2)` tensor `[[min_x, min_y], [max_x, max_y]]` to draw a bounding box and adjust plot limits.
    -   `show_voronoi_vertices` (optional): Boolean to toggle plotting of unique Voronoi vertices.
    -   `ax` (optional): Existing Matplotlib axes to plot on.
    -   `title` (optional): Plot title.
-   **Visualization:**
    -   Input seed points are typically plotted as scatter markers.
    -   Voronoi cell edges are drawn using `matplotlib.patches.Polygon` (usually with `fill=False` for a wireframe appearance).
    -   Unique Voronoi vertices can be plotted as distinct markers.
    -   If `bounds` are given, a rectangle representing the bounds is drawn.

### 2. `plot_voronoi_wireframe_3d`

-   **Signature:** `plot_voronoi_wireframe_3d(points: torch.Tensor, voronoi_cells_faces_indices: list[list[list[int]]] | None = None, unique_voronoi_vertices: torch.Tensor | None = None, ax=None, title: str = "3D Voronoi Diagram (Wireframe)")`
-   **Description:** Plots a 3D Voronoi diagram, focusing on the wireframe structure of the polyhedral cells.
-   **Parameters:**
    -   `points`: The input seed points `(N, 3)`.
    -   `voronoi_cells_faces_indices` (optional): Precomputed list of cells. Each cell is a list of faces, and each face is a list of **ordered global indices** referring to `unique_voronoi_vertices`. If not provided, it's computed via `delaunay_triangulation_3d` and `construct_voronoi_polyhedra_3d`.
    -   `unique_voronoi_vertices` (optional): Precomputed `(V,3)` tensor of unique Voronoi vertex coordinates. Required if `voronoi_cells_faces_indices` is provided.
    -   `ax` (optional): Existing Matplotlib 3D axes to plot on.
    -   `title` (optional): Plot title.
-   **Visualization:**
    -   Input seed points are plotted as 3D scatter markers.
    -   Voronoi cell faces are drawn using `mpl_toolkits.mplot3d.art3d.Poly3DCollection`. Faces are typically rendered semi-transparently with visible edges to give a wireframe appearance. Different cells might be assigned different colors for clarity.

## Matplotlib Backend

These functions rely on Matplotlib for rendering. Ensure Matplotlib is installed and configured appropriately for your environment (e.g., interactive backends for display, or non-interactive backends like 'Agg' for saving to files).

## Example Image Placeholders (Illustrative)

If images were generated, they could be embedded like this:

```markdown
![Example of a 2D Voronoi Diagram](path/to/example_2d_voronoi.png)

![Example of a 3D Voronoi Wireframe](path/to/example_3d_voronoi.png)
```
*(Note: Actual image paths would need to be provided if images are generated and stored).*
