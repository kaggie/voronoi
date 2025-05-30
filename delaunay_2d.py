"""
Computes 2D Delaunay triangulation using the Bowyer-Watson algorithm.

This module provides functions to generate a Delaunay triangulation for a given
set of 2D input points. It includes helper functions for geometric calculations
like finding triangle circumcircles and checking if points lie within them,
relying on `geometry_core.EPSILON` for robust floating-point comparisons.
"""
import torch

from .geometry_core import EPSILON # Epsilon for geometric predicates in Delaunay

def get_triangle_circumcircle_details_2d(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor):
    """
    Computes the circumcenter and squared circumradius of a 2D triangle.

    The circumcenter is the center of the circumcircle, the unique circle that
    passes through all three vertices of the triangle.

    Args:
        p1 (torch.Tensor): Tensor of shape (2,) representing the first vertex.
        p2 (torch.Tensor): Tensor of shape (2,) representing the second vertex.
        p3 (torch.Tensor): Tensor of shape (2,) representing the third vertex.

    Returns:
        Tuple[torch.Tensor | None, torch.Tensor | None]:
            - circumcenter (torch.Tensor | None): Coordinates of the circumcenter (shape (2,)).
                                                  Returns `None` if points are collinear.
            - squared_radius (torch.Tensor | None): Squared circumradius (scalar).
                                                   Returns `None` if points are collinear.
    """
    # Using formulas from Wikipedia / mathworld.wolfram.com
    # Denominator D = 2 * (x1(y2-y3) + x2(y3-y1) + x3(y1-y2))
    # If D is close to zero (within EPSILON), points are considered collinear.
    
    p1x, p1y = p1[0], p1[1]
    p2x, p2y = p2[0], p2[1]
    p3x, p3y = p3[0], p3[1]

    # Calculate D with original precision first, then check with EPSILON
    D_val = 2 * (p1x * (p2y - p3y) + p2x * (p3y - p1y) + p3x * (p1y - p2y))

    if torch.abs(D_val) < EPSILON: # Collinear points
        return None, None 

    # Use float64 for precision in intermediate calculations if original dtype is lower
    # This helps maintain accuracy for the circumcenter calculation.
    # However, if input is already float64, this is not strictly necessary but doesn't harm.
    # For simplicity here, we'll use the input dtype, assuming it's sufficient.
    # If numerical issues arise, casting to float64 for intermediate steps is a common solution.
    
    p1_sq = p1x**2 + p1y**2
    p2_sq = p2x**2 + p2y**2
    p3_sq = p3x**2 + p3y**2

    # Circumcenter coordinates (Ux, Uy)
    Ux = (p1_sq * (p2y - p3y) + p2_sq * (p3y - p1y) + p3_sq * (p1y - p2y)) / D_val
    Uy = (p1_sq * (p3x - p2x) + p2_sq * (p1x - p3x) + p3_sq * (p2x - p1x)) / D_val
    
    circumcenter = torch.tensor([Ux, Uy], dtype=p1.dtype, device=p1.device)
    
    # Squared radius: (x1-Ux)^2 + (y1-Uy)^2
    squared_radius = (p1x - Ux)**2 + (p1y - Uy)**2
    
    return circumcenter, squared_radius

def is_point_in_circumcircle(point: torch.Tensor, 
                                 tri_p1: torch.Tensor, tri_p2: torch.Tensor, tri_p3: torch.Tensor) -> bool:
    """
    Checks if a point is strictly inside the circumcircle of a triangle.

    This is a key predicate for the Bowyer-Watson algorithm. A point is "in" if its
    distance from the circumcenter is less than the circumradius (minus EPSILON
    for strict check).

    Args:
        point (torch.Tensor): The 2D point to check (shape (2,)).
        tri_p1 (torch.Tensor): First vertex of the triangle (shape (2,)).
        tri_p2 (torch.Tensor): Second vertex of the triangle (shape (2,)).
        tri_p3 (torch.Tensor): Third vertex of the triangle (shape (2,)).

    Returns:
        bool: True if the point is strictly inside the circumcircle.
              False if on or outside the circumcircle, or if the triangle is degenerate
              (collinear vertices, for which a circumcircle isn't well-defined by
              `get_triangle_circumcircle_details_2d`).
    """
    circumcenter, squared_radius = get_triangle_circumcircle_details_2d(tri_p1, tri_p2, tri_p3)
    
    if circumcenter is None: # Degenerate triangle, no valid circumcircle
        return False 
        
    dist_sq_to_center = torch.sum((point - circumcenter)**2)
    
    # Check if point is strictly inside: dist_sq < radius_sq - EPSILON
    # Subtracting EPSILON from squared_radius ensures that points lying very close
    # to or on the circle boundary are not considered strictly inside.
    return dist_sq_to_center < squared_radius - EPSILON


def delaunay_triangulation_2d(points: torch.Tensor) -> torch.Tensor:
    """
    Computes the 2D Delaunay triangulation of a set of points using the Bowyer-Watson algorithm.

    The algorithm incrementally inserts points into an existing triangulation, which is
    initialized with a "super-triangle" encompassing all input points. For each
    inserted point, "bad" triangles (those whose circumcircles contain the new point)
    are identified and removed, forming a polygonal cavity. This cavity is then
    re-triangulated by connecting the new point to the cavity edges. Finally,
    triangles connected to the super-triangle vertices are discarded.

    Args:
        points (torch.Tensor): Tensor of shape (N, 2) representing N points in 2D.
                               Coordinates should be in standard Cartesian format.

    Returns:
        torch.Tensor: Tensor of shape (M, 3) representing M Delaunay triangles.
                      Each row contains the original indices (0 to N-1) of the three 
                      points from the input `points` tensor that form a triangle.
                      Returns an empty tensor `(0,3)` if N < 3, as a triangle cannot be formed.
    """
    n_points = points.shape[0]
    if n_points < 3:
        return torch.empty((0, 3), dtype=torch.long, device=points.device)

    device = points.device
    dtype = points.dtype # Should be a float type for coordinate calculations

    # 1. Initialization: Create a super-triangle that encompasses all input points.
    min_coords, _ = torch.min(points, dim=0)
    max_coords, _ = torch.max(points, dim=0)
    
    center = (min_coords + max_coords) / 2.0
    range_coords = max_coords - min_coords
    # max_range is the larger of the width or height of the bounding box of points
    max_range = torch.max(range_coords) 
    if max_range < EPSILON : # Handle cases where all points are nearly coincident
        max_range = torch.tensor(1.0, device=device, dtype=dtype)

    # Define super-triangle vertices. Make it significantly larger than the point cloud.
    # A common factor is 3 to 5 times the max_range.
    offset_val = max_range * 5.0 
    # Ensure super-triangle vertices are placed to avoid issues if points are near origin or axes.
    st_p0 = center + torch.tensor([-offset_val, -offset_val * 0.5], device=device, dtype=dtype) 
    st_p1 = center + torch.tensor([offset_val,  -offset_val * 0.5], device=device, dtype=dtype)
    st_p2 = center + torch.tensor([0.0,          offset_val * 1.5], device=device, dtype=dtype) # Apex above

    # `all_points` includes original points followed by super-triangle vertices.
    # Original points have indices 0 to n_points-1.
    # Super-triangle points have indices n_points, n_points+1, n_points+2.
    all_points = torch.cat([points, st_p0.unsqueeze(0), st_p1.unsqueeze(0), st_p2.unsqueeze(0)], dim=0)
    
    idx_st_p0 = n_points
    idx_st_p1 = n_points + 1
    idx_st_p2 = n_points + 2

    # Initial triangulation consists of only the super-triangle.
    # `triangulation` stores triangles as lists of point indices referencing `all_points`.
    triangulation = [[idx_st_p0, idx_st_p1, idx_st_p2]]

    # 2. Incremental Point Insertion: Add each original point to the triangulation.
    # Optional: Shuffling points (e.g., `torch.randperm(n_points)`) can improve 
    # average-case performance but is not strictly necessary for correctness.
    for i_orig_idx in range(n_points):
        current_point_coords = all_points[i_orig_idx] # Coordinates of the point being inserted

        bad_triangles_indices = [] # Indices of triangles in `triangulation` to be removed
        for tri_list_idx, tri_vertex_indices in enumerate(triangulation):
            p1_coords = all_points[tri_vertex_indices[0]]
            p2_coords = all_points[tri_vertex_indices[1]]
            p3_coords = all_points[tri_vertex_indices[2]]
            
            # If the current point is inside the circumcircle of this triangle, it's a "bad" triangle.
            if is_point_in_circumcircle(current_point_coords, p1_coords, p2_coords, p3_coords):
                bad_triangles_indices.append(tri_list_idx)

        # Form a polygonal cavity from the edges of the bad triangles that are not shared.
        polygon_cavity_edges = [] 
        edge_counts = {} # Counts occurrences of each edge (canonical form: sorted tuple)

        for tri_list_idx in bad_triangles_indices:
            tri_v_indices = triangulation[tri_list_idx] # Indices into `all_points`
            # Define the three edges of the current bad triangle
            edges_of_tri = [
                tuple(sorted((tri_v_indices[0], tri_v_indices[1]))),
                tuple(sorted((tri_v_indices[1], tri_v_indices[2]))),
                tuple(sorted((tri_v_indices[2], tri_v_indices[0])))
            ]
            for edge_tuple in edges_of_tri:
                edge_counts[edge_tuple] = edge_counts.get(edge_tuple, 0) + 1
        
        # Edges that appear once are on the boundary of the polygonal cavity.
        for edge_tuple, count in edge_counts.items():
            if count == 1: 
                polygon_cavity_edges.append(list(edge_tuple)) # Store as [idx1, idx2]

        # Remove bad triangles from the triangulation list (iterate in reverse to maintain valid indices).
        for tri_list_idx in sorted(bad_triangles_indices, reverse=True):
            triangulation.pop(tri_list_idx)

        # Re-triangulate the cavity: Add new triangles by connecting the current_point_original_idx
        # to each edge of the polygonal cavity.
        for edge_v_indices_list in polygon_cavity_edges:
            new_triangle = [i_orig_idx, edge_v_indices_list[0], edge_v_indices_list[1]]
            triangulation.append(new_triangle)
    
    # 3. Finalization: Remove triangles that include any vertex from the super-triangle.
    # These are triangles where any vertex index is >= n_points.
    final_triangulation_list = []
    for tri_v_indices_list in triangulation:
        is_real_triangle = True
        for v_idx_in_all_points in tri_v_indices_list:
            if v_idx_in_all_points >= n_points: # Vertex is part of the super-triangle
                is_real_triangle = False
                break
        if is_real_triangle:
            final_triangulation_list.append(tri_v_indices_list)
    
    if not final_triangulation_list:
        return torch.empty((0, 3), dtype=torch.long, device=device)
        
    return torch.tensor(final_triangulation_list, dtype=torch.long, device=device)
