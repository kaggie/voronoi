"""
Provides functions for analyzing Voronoi diagrams and their constituent cells.

This module includes functionalities to:
- Compute density compensation weights based on Voronoi cell measures (`compute_voronoi_density_weights`),
  including options for clipping to bounds and handling open cells.
- Calculate geometric properties of individual Voronoi cells, such as centroids
  (`compute_cell_centroid`), perimeters (2D), surface areas (3D).
- Determine topological information like cell neighbors (`get_cell_neighbors`).
- Compute shape descriptors like circularity (2D) and sphericity (3D).

It relies on other modules in the library for initial Voronoi diagram construction
(`voronoi_from_delaunay.py`) and core geometric primitives (`geometry_core.py`).
"""
import torch
import math # For pi in circularity/sphericity, log10 in get_cell_neighbors
from collections import defaultdict # For get_cell_neighbors

from .geometry_core import EPSILON, ConvexHull, clip_polygon_2d, clip_polyhedron_3d
from .delaunay_2d import delaunay_triangulation_2d
from .delaunay_3d import delaunay_triangulation_3d
from .voronoi_from_delaunay import construct_voronoi_polygons_2d, construct_voronoi_polyhedra_3d

# --- Weight Computation ---
def compute_voronoi_density_weights(
    points: torch.Tensor, 
    bounds: torch.Tensor | None = None,
    normalize_weights_flag: bool = True
) -> torch.Tensor:
    """
    Computes Voronoi-based density compensation weights for a set of points.

    This function handles both 2D and 3D points. It can clip Voronoi cells to 
    provided bounds before calculating their measure (area/volume). For cells that 
    would be unbounded (typically those corresponding to points on the convex hull 
    of the input set), if no `bounds` are given, a strategy is used to assign them 
    a very large effective measure, resulting in a small, stable weight.

    Args:
        points (torch.Tensor): Tensor of shape (N, D) representing N points in 
                               D-dimensional space (where D is 2 or 3).
        bounds (torch.Tensor | None, optional): If provided, a tensor of shape (2, D) 
                                                representing `[[min_x, min_y, ...], [max_x, max_y, ...]]`.
                                                Voronoi cells will be clipped to these bounds.
                                                Defaults to None (no explicit clipping, open cells handled by heuristic).
        normalize_weights_flag (bool, optional): If True (default), the computed raw weights 
                                                 (typically `1/measure`) are normalized to sum to 1.0.

    Returns:
        torch.Tensor: Tensor of shape (N,) containing the computed density weights for each input point.
    
    Raises:
        ValueError: If `points` are not 2D or 3D, or if `bounds` (if provided)
                    have an incorrect shape.
    """
    n_points, dim = points.shape
    device = points.device
    dtype = points.dtype
    
    current_epsilon = EPSILON 

    if n_points == 0:
        return torch.empty(0, dtype=dtype, device=device)

    if dim not in [2, 3]:
        raise ValueError(f"Points must be 2D or 3D, got {dim}D.")

    weights = torch.zeros(n_points, dtype=dtype, device=device)
    
    # Default measure for problematic cells (e.g., after clipping results in degenracy, or unexpected errors)
    # Using a power of EPSILON. If EPSILON is 1e-7, EPSILON**2 is 1e-14, EPSILON**3 is 1e-21.
    problematic_cell_measure = current_epsilon**(float(dim)) # Ensure float exponent
    # For open cells without bounds, assign a very large measure, leading to a tiny weight.
    open_cell_measure_without_bounds = 1.0 / problematic_cell_measure


    hull_point_indices = set()
    if n_points >= dim + 1: 
        try:
            cvx_hull_obj = ConvexHull(points, tol=current_epsilon)
            if cvx_hull_obj.vertices is not None: # vertices can be None if hull computation fails (e.g. < D+1 unique points)
                hull_point_indices = set(cvx_hull_obj.vertices.tolist())
        except (RuntimeError, ValueError): 
            if bounds is None: # If hull computation fails and no bounds, assume all points could lead to open cells
                 hull_point_indices = set(range(n_points))

    if dim == 2:
        delaunay_simplices = delaunay_triangulation_2d(points)
        if delaunay_simplices.shape[0] == 0 and n_points >=3 : # No Delaunay triangles for non-trivial cases (e.g., collinear points)
            # Fallback to uniform weights if no triangulation can be formed for >=3 points.
            # For <3 points, Delaunay is expected to be empty, specific weight handling occurs later.
            weights.fill_(1.0 / n_points if n_points > 0 else 1.0) # Avoid division by zero
            return weights if normalize_weights_flag else weights * (n_points if n_points > 0 else 1.0)


        voronoi_cells_verts_list, _ = construct_voronoi_polygons_2d(points, delaunay_simplices)

        for i in range(n_points):
            cell_vertices_coords_list = voronoi_cells_verts_list[i]
            measure = -1.0 # Initialize measure as uncomputable

            if bounds is not None:
                if len(cell_vertices_coords_list) >= 1: 
                    # Ensure cell_v_tensor_unclipped is correctly shaped (K,2)
                    cell_v_tensor_unclipped = torch.stack(cell_vertices_coords_list) if len(cell_vertices_coords_list) > 0 else torch.empty((0,dim), dtype=dtype, device=device)
                    if cell_v_tensor_unclipped.ndim == 1 and cell_v_tensor_unclipped.shape[0] == dim : 
                        cell_v_tensor_unclipped = cell_v_tensor_unclipped.unsqueeze(0)
                    
                    if cell_v_tensor_unclipped.shape[0] == 0 and len(cell_vertices_coords_list) > 0 : # Case of single vertex in list
                         cell_v_tensor_unclipped = cell_vertices_coords_list[0].unsqueeze(0)


                    clipped_cell_vertices = clip_polygon_2d(cell_v_tensor_unclipped, bounds)
                    
                    if clipped_cell_vertices.shape[0] >= 3:
                        try:
                            hull = ConvexHull(clipped_cell_vertices, tol=current_epsilon)
                            area = hull.area.item()
                            measure = area if area > current_epsilon else problematic_cell_measure
                        except (RuntimeError, ValueError): measure = problematic_cell_measure
                    else: measure = problematic_cell_measure
                else: measure = problematic_cell_measure
            else: # No bounds provided
                if i in hull_point_indices: 
                    measure = open_cell_measure_without_bounds
                else: 
                    if len(cell_vertices_coords_list) >= 3:
                        cell_v_tensor = torch.stack(cell_vertices_coords_list)
                        try:
                            hull = ConvexHull(cell_v_tensor, tol=current_epsilon)
                            area = hull.area.item()
                            measure = area if area > current_epsilon else problematic_cell_measure
                        except (RuntimeError, ValueError): measure = problematic_cell_measure
                    else: measure = problematic_cell_measure
            
            weights[i] = 1.0 / measure if measure > problematic_cell_measure / 2.0 else 1.0 / problematic_cell_measure
    
    elif dim == 3:
        delaunay_simplices = delaunay_triangulation_3d(points)
        if delaunay_simplices.shape[0] == 0 and n_points >=4: # No Delaunay tetrahedra for non-trivial cases
            weights.fill_(1.0 / n_points if n_points > 0 else 1.0)
            return weights if normalize_weights_flag else weights * (n_points if n_points > 0 else 1.0)

        voronoi_cells_faces_indices, unique_voronoi_vertices = construct_voronoi_polyhedra_3d(points, delaunay_simplices)

        for i in range(n_points):
            cell_faces_list_of_idx_lists = voronoi_cells_faces_indices[i]
            measure = -1.0

            cell_boundary_v_indices_set = set()
            if cell_faces_list_of_idx_lists:
                for face_idx_list in cell_faces_list_of_idx_lists:
                    for v_idx in face_idx_list:
                        cell_boundary_v_indices_set.add(v_idx)
            
            current_cell_v_coords = torch.empty((0,dim), dtype=dtype, device=device)
            if cell_boundary_v_indices_set: # Only try to index if set is not empty
                 current_cell_v_coords = unique_voronoi_vertices[list(cell_boundary_v_indices_set)]

            if bounds is not None:
                if current_cell_v_coords.shape[0] > 0: 
                    clipped_cell_vertices = clip_polyhedron_3d(current_cell_v_coords, bounds)
                    if clipped_cell_vertices.shape[0] >= 4:
                        try:
                            hull = ConvexHull(clipped_cell_vertices, tol=current_epsilon)
                            volume = hull.volume.item()
                            measure = volume if volume > current_epsilon else problematic_cell_measure
                        except (RuntimeError, ValueError): measure = problematic_cell_measure
                    else: measure = problematic_cell_measure
                else: measure = problematic_cell_measure
            else: 
                if i in hull_point_indices:
                    measure = open_cell_measure_without_bounds
                else: 
                    if current_cell_v_coords.shape[0] >= 4:
                        try:
                            hull = ConvexHull(current_cell_v_coords, tol=current_epsilon)
                            volume = hull.volume.item()
                            measure = volume if volume > current_epsilon else problematic_cell_measure
                        except (RuntimeError, ValueError): measure = problematic_cell_measure
                    else: measure = problematic_cell_measure
            
            weights[i] = 1.0 / measure if measure > problematic_cell_measure / 2.0 else 1.0 / problematic_cell_measure

    # Handle cases for < D+1 points where Delaunay might be empty but weights are still expected
    if (dim == 2 and n_points < 3) or (dim == 3 and n_points < 4):
        if n_points > 0: weights.fill_(1.0 / n_points)
        else: return torch.empty(0, dtype=dtype, device=device) # No points, no weights
        return weights if normalize_weights_flag else weights * n_points


    if normalize_weights_flag:
        current_sum = torch.sum(weights)
        if current_sum > current_epsilon : 
            weights = weights / current_sum
        else: # Fallback if sum is too small (e.g., all cells were uncomputable with tiny measure)
            weights.fill_(1.0 / n_points if n_points > 0 else 1.0)
            
    return weights

# --- Helper Functions (Moved from temp_voronoi_helpers.py) ---

def compute_cell_centroid(cell_vertices_list: list[torch.Tensor]) -> torch.Tensor | None:
    """
    Computes the geometric centroid (average of vertex coordinates) of a Voronoi cell.
    Located in: `voronoi_analysis.py`.

    Args:
        cell_vertices_list (list[torch.Tensor]): A list of PyTorch Tensors, 
                                                 where each Tensor represents a vertex 
                                                 of the cell (e.g., shape (Dim,)).
                                                 All tensors should have the same dimension and device.

    Returns:
        torch.Tensor | None: A PyTorch Tensor of shape (Dim,) representing the centroid.
                             Returns `None` if:
                             - The `cell_vertices_list` is empty.
                             - Any element in the list is not a valid tensor.
                             - Vertices are not 2D or 3D.
                             - An error occurs during stacking (e.g., inconsistent tensor shapes).
    """
    if not cell_vertices_list:
        return None
    try:
        # Use device and dtype of the first vertex as reference
        ref_device = cell_vertices_list[0].device
        ref_dtype = cell_vertices_list[0].dtype
        
        processed_vertices = []
        for v in cell_vertices_list:
            if not isinstance(v, torch.Tensor) or v.ndim == 0: # Ensure it's a tensor with at least 1D
                return None # Invalid vertex found
            processed_vertices.append(v.to(device=ref_device, dtype=ref_dtype))
        
        if not processed_vertices: return None 
        vertices_tensor = torch.stack(processed_vertices)

    except Exception: # Catch errors during processing or stacking
        return None 

    if vertices_tensor.ndim != 2: return None # Expected shape (N_vertices, Dim)
    num_vertices, dim = vertices_tensor.shape
    if num_vertices == 0: return None
    if dim not in [2, 3]: return None # Only supports 2D and 3D centroids

    centroid = torch.mean(vertices_tensor.float(), dim=0) # Compute mean using float32 for stability
    return centroid.to(ref_dtype) # Cast back to original (reference) dtype

def get_cell_neighbors(
    target_cell_index: int, 
    all_cells_vertices_list: list[list[torch.Tensor]],
    shared_vertices_threshold: int = 2,
    tolerance: float = 1e-5 
) -> list[int]:
    """
    Finds neighboring cells for a target cell by comparing shared (rounded) vertices.
    Located in: `voronoi_analysis.py`.

    This method is primarily suited for 2D Voronoi diagrams where cells sharing an
    edge (i.e., 2 vertices) are considered neighbors. For 3D, sharing a face
    (>=3 coplanar vertices) is more common for defining neighbors, which this
    function does not explicitly check beyond vertex sharing count.

    Args:
        target_cell_index (int): Index of the target cell in `all_cells_vertices_list`.
        all_cells_vertices_list (list[list[torch.Tensor]]): A list where each element
                                 is a list of vertex coordinate Tensors for a cell.
                                 Each vertex tensor is expected to be 1D (e.g., shape (2,) for 2D).
        shared_vertices_threshold (int, optional): Minimum number of shared vertices for two 
                                                   cells to be considered neighbors. Defaults to 2.
        tolerance (float, optional): The tolerance used for rounding vertex coordinates
                                     before comparison. This helps match vertices that are
                                     semantically identical but may differ due to floating-point
                                     arithmetic. Coordinates are effectively rounded to a number
                                     of decimal places derived from this tolerance. Defaults to 1e-5.

    Returns:
        list[int]: A list of integer indices of the cells that are neighbors
                   to the `target_cell_index`.
    """
    if not (0 <= target_cell_index < len(all_cells_vertices_list)): return []
    
    target_cell_v_list = all_cells_vertices_list[target_cell_index]
    if not target_cell_v_list or not isinstance(target_cell_v_list[0], torch.Tensor) or target_cell_v_list[0].ndim == 0:
        return [] # Target cell has no valid vertices

    # Determine rounding precision from tolerance
    # E.g., tol=1e-5 -> round to 5 decimal places.
    # If tolerance is 0 or negative, no rounding is applied (exact match).
    num_decimal_places = 0
    if tolerance > 0:
        num_decimal_places = int(-math.log10(tolerance)) if tolerance < 1.0 else 0
        if num_decimal_places < 0: num_decimal_places = 0 # Ensure non-negative

    def round_tensor_to_tuple(v_tensor: torch.Tensor):
        if tolerance > 0:
            # Multiply, round, then divide to achieve decimal place rounding for tensors
            multiplier = 10**num_decimal_places
            return tuple((v_tensor * multiplier).round().tolist())
        else: # Exact comparison
            return tuple(v_tensor.tolist())

    try:
        target_cell_v_tuples = set(
            round_tensor_to_tuple(v) for v in target_cell_v_list if isinstance(v, torch.Tensor) and v.ndim > 0
        )
    except Exception: return [] # Error processing target cell vertices
    if not target_cell_v_tuples: return []

    neighbor_indices = []
    for i in range(len(all_cells_vertices_list)):
        if i == target_cell_index: continue

        current_cell_v_list = all_cells_vertices_list[i]
        if not current_cell_v_list or not isinstance(current_cell_v_list[0], torch.Tensor) or current_cell_v_list[0].ndim == 0:
            continue # Current cell has no valid vertices
        
        # Basic check for dimension compatibility (using first vertex)
        if target_cell_v_list[0].shape != current_cell_v_list[0].shape:
            continue

        try:
            current_cell_v_tuples = set(
                round_tensor_to_tuple(v) for v in current_cell_v_list if isinstance(v, torch.Tensor) and v.ndim > 0
            )
        except Exception: continue # Error processing current cell's vertices
        if not current_cell_v_tuples: continue
        
        if len(target_cell_v_tuples.intersection(current_cell_v_tuples)) >= shared_vertices_threshold:
            neighbor_indices.append(i)
    return neighbor_indices

# --- New Analysis Functions ---

def compute_cell_perimeter_2d(cell_vertices: torch.Tensor) -> torch.Tensor:
    """
    Computes the perimeter of a 2D polygon defined by ordered vertices.
    Located in: `voronoi_analysis.py`.

    Args:
        cell_vertices (torch.Tensor): Tensor of shape (N, 2) representing N ordered 
                                      vertices of the 2D polygon.

    Returns:
        torch.Tensor: Scalar tensor representing the perimeter. Returns 0.0 if N < 2.
    """
    if cell_vertices.shape[0] < 2:
        return torch.tensor(0.0, dtype=cell_vertices.dtype, device=cell_vertices.device)
    
    # Calculate lengths of segments between consecutive vertices, including closing segment
    segments = cell_vertices - torch.roll(cell_vertices, shifts=-1, dims=0) # v_i - v_{i+1}
    perimeter = torch.sum(torch.linalg.norm(segments, dim=1))
    return perimeter

def compute_cell_surface_area_3d(cell_vertices: torch.Tensor) -> torch.Tensor:
    """
    Computes the surface area of a 3D polyhedron by calculating the area of its convex hull.
    Located in: `voronoi_analysis.py`.

    Args:
        cell_vertices (torch.Tensor): Tensor of shape (N, 3) representing the vertices 
                                      of the 3D polyhedron. The order of vertices does
                                      not strictly matter as the convex hull is computed.

    Returns:
        torch.Tensor: Scalar tensor representing the surface area. Returns 0.0 if N < 3
                      (not enough points for a planar face) or if hull computation fails.
    """
    if cell_vertices.shape[0] < 3: 
        return torch.tensor(0.0, dtype=cell_vertices.dtype, device=cell_vertices.device)
    try:
        # ConvexHull.area calculates surface area for 3D objects.
        # Ensure tolerance is passed for robust hull computation.
        hull = ConvexHull(cell_vertices, tol=EPSILON) 
        surface_area = hull.area 
    except (RuntimeError, ValueError): # Handle cases where hull computation fails (e.g., degenerate)
        surface_area = torch.tensor(0.0, dtype=cell_vertices.dtype, device=cell_vertices.device)
    return surface_area


def compute_circularity_2d(area: torch.Tensor, perimeter: torch.Tensor) -> torch.Tensor:
    """
    Computes the circularity (Polsky-Pólya number) of a 2D shape. 
    Circularity = (4 * pi * Area) / (Perimeter^2).
    A perfect circle has circularity 1. Values are <= 1.
    Located in: `voronoi_analysis.py`.

    Args:
        area (torch.Tensor): Scalar tensor for the area of the shape.
        perimeter (torch.Tensor): Scalar tensor for the perimeter of the shape.

    Returns:
        torch.Tensor: Scalar tensor representing the circularity. Returns 0.0 if 
                      perimeter is zero (or very close to it, based on `EPSILON`)
                      to prevent division by zero.
    """
    if torch.abs(perimeter) < EPSILON:
        return torch.tensor(0.0, dtype=area.dtype, device=area.device)
    
    circularity = (4 * math.pi * area) / (perimeter**2)
    return circularity

def compute_sphericity_3d(volume: torch.Tensor, surface_area: torch.Tensor) -> torch.Tensor:
    """
    Computes the sphericity of a 3D shape. 
    Sphericity = (pi^(1/3) * (6 * Volume)^(2/3)) / SurfaceArea.
    A perfect sphere has sphericity 1. Values are <= 1.
    Located in: `voronoi_analysis.py`.

    Args:
        volume (torch.Tensor): Scalar tensor for the volume of the shape.
        surface_area (torch.Tensor): Scalar tensor for the surface area of the shape.

    Returns:
        torch.Tensor: Scalar tensor representing the sphericity. Returns 0.0 if 
                      surface_area is zero (or very close to it, based on `EPSILON`)
                      to prevent division by zero.
    """
    if torch.abs(surface_area) < EPSILON:
        return torch.tensor(0.0, dtype=volume.dtype, device=volume.device)
    
    # Ensure volume is non-negative for (6*V)^(2/3)
    safe_volume = torch.clamp(volume, min=0.0)

    pi_one_third = math.pi**(1/3)
    six_volume_two_thirds = (6 * safe_volume)**(2/3)
    
    sphericity = (pi_one_third * six_volume_two_thirds) / surface_area
    return sphericity
