# Project Wiki - PyTorch Voronoi & Delaunay Toolkit

Welcome to the main documentation hub for the PyTorch Voronoi & Delaunay Toolkit. This wiki provides detailed information about the library's architecture, core modules, functionalities, and usage.

## Overview

This library offers a suite of tools for generating, analyzing, and visualizing Voronoi diagrams and Delaunay triangulations using PyTorch. It is designed to be modular, allowing users to leverage individual components or follow a complete workflow from point sets to detailed geometric analysis.

## Key Documentation Pages

Please explore the following pages for in-depth information:

*   **[Overall Architecture](./architecture.md)**
    *   A high-level overview of the library's structure and how the different modules interact.

*   **[Core Geometric Primitives (`geometry_core.py`)](./geometry_core.md)**
    *   Details on the foundational geometric functions, constants (`EPSILON`), convex hull computations, clipping algorithms, and geometric predicates.

*   **[Delaunay Triangulation (`delaunay_2d.py`, `delaunay_3d.py`)](./delaunay.md)**
    *   Covers the 2D and 3D Delaunay triangulation algorithms, their inputs, outputs, and dependencies on `geometry_core.py`. This page also includes information on circumcenter calculations, which are integral to Voronoi diagram construction.

*   **[Voronoi Diagram Construction (`voronoi_from_delaunay.py`)](./voronoi_construction.md)**
    *   Explains how Voronoi diagrams are constructed from their dual Delaunay triangulations in both 2D (polygons with ordered vertices) and 3D (polyhedra with ordered face vertices).

*   **[Voronoi Cell Analysis (`voronoi_analysis.py`)](./voronoi_analysis.md)**
    *   Describes functions for analyzing Voronoi cells, including:
        *   Density weight calculation (with clipping and open cell handling).
        *   Centroid computation.
        *   Neighbor finding (2D).
        *   Perimeter (2D), surface area (3D), circularity (2D), and sphericity (3D) calculations.

*   **[Voronoi Diagram Plotting (`voronoi_plotting.py`)](./voronoi_plotting.md)**
    *   Information on using Matplotlib-based utilities to visualize 2D and 3D Voronoi diagrams.

*   **[Raster-based Voronoi Tessellation (`voronoi_tessellation.py`)](./tessellation.md)**
    *   Details on generating a grid-based Voronoi map where each voxel is assigned to its nearest seed. Mentions its use of `find_nearest_seed.py`.

*   **[Distance Utilities (`find_nearest_seed.py`)](./distance_utils.md)**
    *   Documentation for utility functions like `find_nearest_seed` used for efficient nearest neighbor searches.

---
*Navigate to the respective pages to learn more about each component.*
