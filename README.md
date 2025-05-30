# PyTorch Voronoi & Delaunay Toolkit

## Project Overview

This library provides a collection of PyTorch-based tools for computing and analyzing Voronoi diagrams and Delaunay triangulations in 2D and 3D. It offers functionalities ranging from core geometric primitives to higher-level analysis and plotting utilities. The emphasis on PyTorch allows for potential GPU acceleration for various computations.

## Installation

1.  **Prerequisites:**
    *   Python 3.8 or higher.
    *   PyTorch: Installation instructions can be found at [pytorch.org](https://pytorch.org/). Ensure you install a version compatible with your CUDA toolkit if GPU support is desired.

2.  **Install Dependencies:**
    Clone the repository and install the necessary packages using `requirements.txt`:
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd <repository_name>   # Replace <repository_name> with the actual directory name
    pip install -r requirements.txt
    ```

## Key Features

*   **Core Geometric Primitives (`geometry_core.py`):**
    *   Robust, PyTorch-based functions for fundamental geometric operations.
    *   Common `EPSILON` constant for numerical stability (currently `1e-7`).
    *   Convex hull computation in 2D and 3D (`ConvexHull` class).
    *   Polygon (2D) and polyhedron (3D) clipping against bounding boxes.
    *   Area and volume calculations for convex shapes.
    *   Essential geometric predicates (e.g., 3D orientation tests, in-circumsphere tests).

*   **Delaunay Triangulation (`delaunay_2d.py`, `delaunay_3d.py`):**
    *   2D Delaunay triangulation using a Bowyer-Watson style algorithm.
    *   3D Delaunay tetrahedralization using an incremental insertion algorithm.

*   **Voronoi Diagram Construction (`voronoi_from_delaunay.py`):**
    *   Construction of Voronoi diagrams from their dual Delaunay triangulations.
    *   For 2D cells, provides ordered lists of vertex coordinates.
    *   For 3D cells, provides face definitions with ordered vertex indices (referencing a list of unique Voronoi vertices). Includes logic for ordering vertices within each 3D Voronoi face.

*   **Voronoi Cell Analysis (`voronoi_analysis.py`):**
    *   `compute_voronoi_density_weights`: Calculates density compensation weights based on cell area/volume. Includes options for clipping to bounds and handling of open/unbounded cells.
    *   `compute_cell_centroid`: Determines the geometric centroid of cell vertices.
    *   `get_cell_neighbors`: Finds adjacent cells in a 2D Voronoi diagram.
    *   Geometric property calculations:
        *   `compute_cell_perimeter_2d`
        *   `compute_cell_surface_area_3d`
        *   `compute_circularity_2d` (shape factor based on area and perimeter).
        *   `compute_sphericity_3d` (shape factor based on volume and surface area).

*   **Plotting Utilities (`voronoi_plotting.py`):**
    *   Matplotlib-based functions for visualizing Voronoi diagrams.
    *   `plot_voronoi_diagram_2d`: Plots 2D Voronoi cells, seed points, and optional Voronoi vertices. Can include a bounding box.
    *   `plot_voronoi_wireframe_3d`: Plots 3D Voronoi cells as wireframes with semi-transparent faces, along with seed points.

*   **Raster-based Voronoi Tessellation (`voronoi_tessellation.py`):**
    *   `compute_voronoi_tessellation`: Generates a grid where each cell is assigned the index of the nearest seed point. Supports masking and custom voxel sizes. Relies on `find_nearest_seed.py`.

## Basic Usage (2D Example)

Here's a quick example of how to compute and plot a 2D Voronoi diagram:

```python
import torch
import matplotlib.pyplot as plt

# Assuming the library modules are in the PYTHONPATH or structured as a package
# Example: from package_name.delaunay_2d import delaunay_triangulation_2d
# For direct script use, ensure modules are in the same directory or adjust sys.path
from delaunay_2d import delaunay_triangulation_2d
from voronoi_from_delaunay import construct_voronoi_polygons_2d
from voronoi_analysis import compute_voronoi_density_weights
from voronoi_plotting import plot_voronoi_diagram_2d

# 1. Define some 2D seed points
seed_points_2d = torch.rand((20, 2), dtype=torch.float32) * 100

# 2. Compute Delaunay triangulation
delaunay_tris = delaunay_triangulation_2d(seed_points_2d)

# 3. Construct Voronoi polygons
# Returns: list of (list of vertex coordinate Tensors per cell), tensor of unique Voronoi vertex coordinates
voronoi_cells_verts_list, unique_voronoi_vertices = construct_voronoi_polygons_2d(seed_points_2d, delaunay_tris)

# 4. Compute density weights (optional)
weights = compute_voronoi_density_weights(seed_points_2d, normalize_weights_flag=True)
print("Computed Density Weights (summing to 1):")
# for i, w in enumerate(weights): # Commented out for brevity in README example
#     print(f"Point {i}: {w.item():.4f}")

# 5. Plot the Voronoi diagram
fig, ax = plt.subplots(figsize=(8,8))
plot_voronoi_diagram_2d(
    seed_points_2d, 
    voronoi_cells_verts_list=voronoi_cells_verts_list, 
    unique_voronoi_vertices=unique_voronoi_vertices,
    ax=ax,
    title="2D Voronoi Diagram Example"
)
# Define bounds for visualization (optional, but good for unbounded cells)
plot_bounds_min = torch.min(seed_points_2d, dim=0).values - 10
plot_bounds_max = torch.max(seed_points_2d, dim=0).values + 10
plot_bounds = torch.stack([plot_bounds_min, plot_bounds_max])

# ax.clear() # Clear previous plot if re-plotting on same axes
# plot_voronoi_diagram_2d(seed_points_2d, bounds=plot_bounds, ax=ax, title="2D Voronoi Diagram with Bounds")

plt.show()
```

## Documentation

Detailed documentation for each module and its functions can be found in the **[wiki](./wiki/README.md)**.
This includes information on architecture, specific algorithms, function signatures, and usage notes.

---
*This library is currently under development.*
