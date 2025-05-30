# Library Architecture

This document outlines the high-level architecture of the Voronoi and Delaunay computation library. The library is designed with a modular approach, separating concerns into distinct Python modules.

## Key Modules and Their Roles

The core components of the library are:

*   **`geometry_core.py`**: This is the foundational module providing essential PyTorch-based geometric primitives. It includes functions for convex hull computation, polygon and polyhedron clipping, area and volume calculations, fundamental geometric predicates (like orientation tests and in-circumsphere tests), and a common `EPSILON` constant for numerical stability.

*   **`delaunay_2d.py`**: Contains the logic for performing 2D Delaunay triangulation of a given set of input points. It utilizes primitives from `geometry_core.py`.

*   **`delaunay_3d.py`**: Contains the logic for performing 3D Delaunay triangulation. Similar to its 2D counterpart, it relies on `geometry_core.py` for fundamental geometric tests.

*   **`circumcenter_calculations.py`**: Provides functions to calculate the circumcenters of 2D triangles and 3D tetrahedra, which are crucial for Voronoi diagram construction. It uses primitives from `geometry_core.py`.

*   **`voronoi_from_delaunay.py`**: This module takes the output of Delaunay triangulation (simplices) and constructs the Voronoi diagram. 
    *   For 2D, it generates ordered vertices for each Voronoi polygon.
    *   For 3D, it identifies Voronoi polyhedra, their faces, and the ordered vertices for each face. It uses circumcenter calculations and geometric ordering logic.

*   **`voronoi_analysis.py`**: Offers a suite of functions to analyze Voronoi diagrams and their constituent cells. This includes:
    *   `compute_voronoi_density_weights`: Calculates density compensation weights.
    *   `compute_cell_centroid`: Finds the geometric center of a Voronoi cell.
    *   `get_cell_neighbors`: Identifies adjacent Voronoi cells.
    *   Functions for geometric properties like perimeter (2D), surface area (3D), circularity (2D), and sphericity (3D).

*   **`voronoi_plotting.py`**: Provides utilities for visualizing Voronoi diagrams in 2D and 3D using Matplotlib. It can plot input points, Voronoi cell edges, and optionally Voronoi vertices.

*   **`voronoi_tessellation.py`**: Implements functionality to create a rasterized (grid-based) Voronoi tessellation, assigning each voxel in a grid to its nearest seed point. It uses `find_nearest_seed.py`.

*   **`find_nearest_seed.py`**: A utility module to efficiently find the nearest seed point for a given set of query points.

## Typical Workflow

A common workflow using this library would be:

1.  **Input**: Start with a set of 2D or 3D seed points (`torch.Tensor`).
2.  **Delaunay Triangulation**:
    *   Use `delaunay_2d.py` or `delaunay_3d.py` to compute the Delaunay triangulation/tetrahedralization of the input points.
3.  **Voronoi Diagram Construction**:
    *   Pass the input points and the Delaunay simplices to `voronoi_from_delaunay.py` to construct the Voronoi cells (polygons in 2D, polyhedra with face definitions in 3D) and identify the unique Voronoi vertices.
4.  **Analysis (Optional)**:
    *   Use functions from `voronoi_analysis.py` to compute properties like density weights, cell centroids, neighbors, perimeters, areas, volumes, circularity, or sphericity.
5.  **Plotting (Optional)**:
    *   Use `voronoi_plotting.py` to visualize the input points and the constructed Voronoi diagram (either 2D polygons or 3D wireframes/polyhedra).
6.  **Raster Tessellation (Alternative Workflow)**:
    *   If a grid-based tessellation is needed directly, `voronoi_tessellation.py` can be used with the input seed points to assign each grid voxel to the nearest seed.

This modular design allows for flexibility and reusability of components. Core geometric operations are centralized in `geometry_core.py`, ensuring consistency and making them available to all higher-level modules.
