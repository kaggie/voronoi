# Voronoi Diagram Construction

This document describes functions for constructing Voronoi diagrams from Delaunay triangulations, primarily focusing on the `voronoi_from_delaunay.py` module. The construction relies on the principle that the Voronoi diagram is the geometric dual of the Delaunay triangulation.

## Core Concepts

-   **Voronoi Vertices:** In a Delaunay triangulation, the circumcenter of each Delaunay simplex (triangle in 2D, tetrahedron in 3D) is a vertex of the Voronoi diagram.
-   **Voronoi Edges/Faces:** Voronoi edges (in 2D) or faces (in 3D) lie on the perpendicular bisectors of the Delaunay edges.
-   **Dependencies:** The functions in `voronoi_from_delaunay.py` utilize circumcenter calculations from `circumcenter_calculations.py` and fundamental geometric constants/primitives from `geometry_core.py`.

## 2D Voronoi Polygons

-   **Module:** `voronoi_from_delaunay.py`
-   **Function:** `construct_voronoi_polygons_2d(points: torch.Tensor, delaunay_triangles: torch.Tensor)`
-   **Description:** Constructs the Voronoi cells (polygons) for a set of 2D input points, given their Delaunay triangulation.
-   **Input:**
    -   `points`: A PyTorch tensor of shape `(N, 2)` containing the coordinates of the N input points.
    -   `delaunay_triangles`: A PyTorch tensor of shape `(M, 3)` representing M Delaunay triangles. Each row contains point indices referring to the `points` tensor.
-   **Output:**
    -   `voronoi_cells_vertices_list`: A list of lists (one for each input point). Each inner list `voronoi_cells_vertices_list[i]` contains PyTorch Tensors (each of shape `(2,)`) representing the **ordered coordinates** of the vertices of the Voronoi cell for `points[i]`.
    -   `unique_voronoi_vertices`: A PyTorch tensor of shape `(V, 2)` containing the coordinates of all unique Voronoi vertices identified (these are the circumcenters of the Delaunay triangles).
-   **Algorithm:**
    1.  Calculates the circumcenter for each valid Delaunay triangle; these form the set of potential Voronoi vertices.
    2.  Identifies unique Voronoi vertices from these circumcenters.
    3.  Maps each input point to the list of Delaunay triangles it is part of.
    4.  The Voronoi cell for an input point `P_i` is formed by the unique Voronoi vertices derived from the circumcenters of Delaunay triangles incident to `P_i`.
    5.  The vertices for each 2D Voronoi cell are ordered angularly around their centroid to form a well-defined polygon.
-   **Notes:**
    -   For input points on the convex hull of the point set, their Voronoi cells will be unbounded. The returned vertex list for such cells defines the finite part of the polygon. Full representation of unbounded cells (e.g., with rays) or clipping to a bounding box is handled by other modules (e.g., `voronoi_analysis.py` for weights, `voronoi_plotting.py` for visualization).

## 3D Voronoi Polyhedra

-   **Module:** `voronoi_from_delaunay.py`
-   **Function:** `construct_voronoi_polyhedra_3d(points: torch.Tensor, delaunay_tetrahedra: torch.Tensor)`
-   **Description:** Constructs Voronoi cells (polyhedra) for 3D input points from their Delaunay tetrahedralization. This version focuses on identifying the faces of each polyhedron and providing ordered vertices for each face.
-   **Input:**
    -   `points`: `(N, 3)` tensor of input point coordinates.
    -   `delaunay_tetrahedra`: `(M, 4)` tensor of Delaunay tetrahedra (point indices).
-   **Output:**
    -   `voronoi_cells_faces_indices`: A list of cells (one for each input point `points[i]`). Each cell is represented as a list of its faces. Each face, in turn, is a list of **ordered global indices**, where these indices refer to vertices in `unique_voronoi_vertices`. Faces with fewer than 3 vertices are typically filtered out.
    -   `unique_voronoi_vertices`: `(V, 3)` tensor of unique Voronoi vertex coordinates (circumcenters of Delaunay tetrahedra).
-   **Algorithm:**
    1.  Computes circumcenters for all valid Delaunay tetrahedra to get the set of unique Voronoi vertices.
    2.  **Identify Voronoi Faces:** A Voronoi face is dual to a Delaunay edge. The vertices of a Voronoi face are the circumcenters of all Delaunay tetrahedra that share this common Delaunay edge. The module builds a map from each Delaunay edge to these corresponding Voronoi vertex indices.
    3.  **Construct Cells and Order Face Vertices:** For each input point `P_i` (center of a Voronoi cell):
        *   Its Voronoi cell is defined by faces dual to Delaunay edges connected to `P_i`.
        *   For each such face, the collected Voronoi vertices (coordinates) are ordered geometrically using the helper function `_order_face_vertices_3d`. This function projects the face vertices onto a plane perpendicular to the dual Delaunay edge and sorts them angularly.
        *   The final output for each cell is a list of its faces, with each face represented by ordered indices pointing to the `unique_voronoi_vertices` tensor.
-   **Notes:**
    -   The ordering of vertices for each 3D Voronoi face is crucial for correct geometric representation and subsequent calculations (e.g., surface area, plotting).
    -   Similar to 2D, cells corresponding to input points on the convex hull will be unbounded. The returned face structure describes the finite part of these polyhedra. Explicit clipping is handled by other modules.
    -   Faces with fewer than 3 vertices (i.e., edges or points, which can occur for unbounded parts) are filtered out from the face list of a cell.
