"""
Constructs Voronoi diagrams from Delaunay triangulations in 2D and 3D.

This module takes a set of input points and their corresponding Delaunay
triangulation (or tetrahedralization) and computes the Voronoi diagram.
The Voronoi vertices are the circumcenters of the Delaunay simplices.

For 2D, it returns ordered lists of vertex coordinates for each Voronoi polygon.
For 3D, it returns a more complex structure representing polyhedral cells,
including lists of faces, where each face's vertices are ordered and globally indexed.
It relies on `circumcenter_calculations.py` for finding circumcenters and
`geometry_core.py` for `EPSILON` and geometric operations used in helper functions
(e.g., for ordering 3D face vertices).
"""
import torch
# unittest import removed from here, should be in test files only
from collections import defaultdict
from .circumcenter_calculations import compute_triangle_circumcenter_2d, compute_tetrahedron_circumcenter_3d
from .geometry_core import EPSILON # Import EPSILON
import math # For _order_face_vertices_3d if it uses math.pi, atan2 is torch.atan2

# --- Voronoi Diagram Construction from Delaunay ---

def construct_voronoi_polygons_2d(points: torch.Tensor, delaunay_triangles: torch.Tensor):
    """
    Constructs Voronoi cells (polygons) from a 2D Delaunay triangulation.

    The vertices of the Voronoi cells are the circumcenters of the Delaunay triangles.
    For each input point, this function identifies the set of circumcenters that
    form its Voronoi cell and then orders these vertices by angle around their
    centroid to define the Voronoi polygon.

    Args:
        points (torch.Tensor): Tensor of shape (N, 2), coordinates of the N input seed points.
        delaunay_triangles (torch.Tensor): Tensor of shape (M, 3), representing M Delaunay
                                           triangles. Each row contains indices referring to `points`.

    Returns:
        Tuple[List[List[torch.Tensor]], torch.Tensor]:
            - voronoi_cells_vertices_list (List[List[torch.Tensor]]): A list of N elements.
              Each element `voronoi_cells_vertices_list[i]` is a list of 2D coordinate
              tensors (shape (2,)), representing the ordered vertices of the Voronoi cell
              for `points[i]`. For unbounded cells, this list may represent the finite
              part of the cell. Cells with < 3 vertices are returned as is.
            - unique_voronoi_vertices (torch.Tensor): Tensor of shape (V, 2) containing the
              coordinates of all V unique Voronoi vertices (circumcenters).
    """
    if points.shape[0] == 0 or delaunay_triangles.shape[0] == 0:
        return [[] for _ in range(points.shape[0])], torch.empty((0, 2), dtype=points.dtype, device=points.device)

    tet_to_circumcenter_coords = {} 
    all_circumcenters_list = []
    for i, tri_indices in enumerate(delaunay_triangles):
        p1, p2, p3 = points[tri_indices[0]], points[tri_indices[1]], points[tri_indices[2]]
        circumcenter = compute_triangle_circumcenter_2d(p1, p2, p3)
        if circumcenter is not None:
            tet_to_circumcenter_coords[i] = circumcenter
            all_circumcenters_list.append(circumcenter)
    
    if not all_circumcenters_list: 
         return [[] for _ in range(points.shape[0])], torch.empty((0,2), dtype=points.dtype, device=points.device)

    unique_voronoi_vertices, inverse_indices = torch.unique(torch.stack(all_circumcenters_list), dim=0, return_inverse=True)
    
    tri_idx_to_unique_voronoi_v_idx = {}
    processed_original_circumcenters = 0
    for i in range(delaunay_triangles.shape[0]): 
        if i in tet_to_circumcenter_coords: 
            tri_idx_to_unique_voronoi_v_idx[i] = inverse_indices[processed_original_circumcenters].item()
            processed_original_circumcenters += 1
            
    point_to_triangles_map = defaultdict(list)
    for tri_idx, tri_point_indices in enumerate(delaunay_triangles):
        if tri_idx not in tri_idx_to_unique_voronoi_v_idx: continue 
        for pt_idx in tri_point_indices:
            point_to_triangles_map[pt_idx.item()].append(tri_idx)

    voronoi_cells_vertices_list = [[] for _ in range(points.shape[0])]

    for pt_idx, incident_triangle_indices in point_to_triangles_map.items():
        if not incident_triangle_indices:
            continue
        
        cell_voronoi_v_unique_indices = list(set(tri_idx_to_unique_voronoi_v_idx[tri_idx] for tri_idx in incident_triangle_indices if tri_idx in tri_idx_to_unique_voronoi_v_idx))
        
        if not cell_voronoi_v_unique_indices: continue
        
        cell_v_coords = unique_voronoi_vertices[torch.tensor(cell_voronoi_v_unique_indices, dtype=torch.long, device=points.device)]
        
        if cell_v_coords.shape[0] < 3: 
            voronoi_cells_vertices_list[pt_idx] = [cv_coord for cv_coord in cell_v_coords] 
            continue

        centroid = torch.mean(cell_v_coords, dim=0)
        angles = torch.atan2(cell_v_coords[:,1] - centroid[1], cell_v_coords[:,0] - centroid[0])
        sorted_indices_for_ordering = torch.argsort(angles)
        
        ordered_cell_v_coords = cell_v_coords[sorted_indices_for_ordering]
        voronoi_cells_vertices_list[pt_idx] = [v_coord for v_coord in ordered_cell_v_coords] 

    return voronoi_cells_vertices_list, unique_voronoi_vertices

def _order_face_vertices_3d(face_vertices_coords: torch.Tensor, edge_points_A: torch.Tensor, edge_points_B: torch.Tensor):
    """ 
    Orders vertices of a 3D Voronoi face angularly around the dual Delaunay edge.

    The ordering is achieved by projecting the face vertices (which are circumcenters
    of tetrahedra sharing the Delaunay edge A-B) onto a plane perpendicular to A-B.
    Angles are then computed in this plane relative to the face centroid, and vertices
    are sorted by these angles.

    Args:
        face_vertices_coords (torch.Tensor): Tensor of shape (M, 3) containing the
                                             coordinates of M Voronoi vertices forming the face.
        edge_points_A (torch.Tensor): Coordinates of the first point of the dual Delaunay edge (shape (3,)).
        edge_points_B (torch.Tensor): Coordinates of the second point of the dual Delaunay edge (shape (3,)).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - ordered_face_vertices_coords (torch.Tensor): The input `face_vertices_coords`
              sorted according to the computed angles. Shape (M,3).
            - sorted_indices (torch.Tensor): The indices that sort the input `face_vertices_coords`.
              Shape (M,).
        If the face has fewer than 3 vertices or the dual Delaunay edge is degenerate,
        the original coordinates and unsorted indices are returned.
    """
    if face_vertices_coords.shape[0] < 3:
        return face_vertices_coords, torch.arange(face_vertices_coords.shape[0], device=face_vertices_coords.device)

    v_axis = edge_points_B - edge_points_A
    v_axis_norm = torch.linalg.norm(v_axis)
    if v_axis_norm < EPSILON: # Degenerate edge
        return face_vertices_coords, torch.arange(face_vertices_coords.shape[0], device=face_vertices_coords.device)
    v_axis = v_axis / v_axis_norm

    centroid = torch.mean(face_vertices_coords, dim=0)

    a_random_vec = torch.tensor([1.0, 0.0, 0.0], dtype=v_axis.dtype, device=v_axis.device)
    if torch.abs(torch.dot(v_axis, a_random_vec)) > 1.0 - EPSILON:
        a_random_vec = torch.tensor([0.0, 1.0, 0.0], dtype=v_axis.dtype, device=v_axis.device)

    u1 = torch.cross(v_axis, a_random_vec)
    u1_norm = torch.linalg.norm(u1)
    if u1_norm < EPSILON: 
        return face_vertices_coords, torch.arange(face_vertices_coords.shape[0], device=face_vertices_coords.device)
    u1 = u1 / u1_norm
    
    u2 = torch.cross(v_axis, u1) 

    projected_coords_list = []
    for v_coord in face_vertices_coords:
        v_rel = v_coord - centroid
        coord_u1 = torch.dot(v_rel, u1)
        coord_u2 = torch.dot(v_rel, u2)
        projected_coords_list.append(torch.tensor([coord_u1, coord_u2], device=v_coord.device, dtype=v_coord.dtype))
    
    projected_coords_tensor = torch.stack(projected_coords_list)
    
    angles = torch.atan2(projected_coords_tensor[:,1], projected_coords_tensor[:,0])
    sorted_indices = torch.argsort(angles)
    
    return face_vertices_coords[sorted_indices], sorted_indices

def construct_voronoi_polyhedra_3d(points: torch.Tensor, delaunay_tetrahedra: torch.Tensor):
    """
    Constructs Voronoi polyhedra from a 3D Delaunay tetrahedralization.

    Each Voronoi cell is defined by a set of faces. Each face is dual to a
    Delaunay edge and is formed by the circumcenters of the Delaunay tetrahedra
    sharing that edge. The vertices of each face are ordered angularly.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) of N input seed points.
        delaunay_tetrahedra (torch.Tensor): Tensor of shape (M, 4) of M Delaunay
                                            tetrahedra, with indices referring to `points`.

    Returns:
        Tuple[List[List[List[int]]], torch.Tensor]:
            - voronoi_cells_faces_indices (List[List[List[int]]]): A list of N elements.
              Each element `voronoi_cells_faces_indices[i]` corresponds to the Voronoi cell
              for `points[i]`. This cell is represented as a list of its faces.
              Each face is a list of **ordered integer indices**, where these indices
              refer to rows in `unique_voronoi_vertices`. Faces with fewer than 3
              vertices are filtered out.
            - unique_voronoi_vertices (torch.Tensor): Tensor of shape (V, 3) containing
              the coordinates of all V unique Voronoi vertices (circumcenters).
    """
    num_input_points = points.shape[0]
    if num_input_points == 0 or delaunay_tetrahedra.shape[0] == 0:
        return [[] for _ in range(num_input_points)], torch.empty((0, 3), dtype=points.dtype, device=points.device)

    tet_to_circumcenter_coords = {}
    all_circumcenters_list = []
    for i, tet_indices in enumerate(delaunay_tetrahedra):
        p_indices = tet_indices
        p0, p1, p2, p3 = points[p_indices[0]], points[p_indices[1]], points[p_indices[2]], points[p_indices[3]]
        circumcenter = compute_tetrahedron_circumcenter_3d(p0, p1, p2, p3)
        if circumcenter is not None:
            tet_to_circumcenter_coords[i] = circumcenter
            all_circumcenters_list.append(circumcenter)

    if not all_circumcenters_list:
        return [[] for _ in range(num_input_points)], torch.empty((0, 3), dtype=points.dtype, device=points.device)

    unique_voronoi_vertices, inverse_indices = torch.unique(torch.stack(all_circumcenters_list), dim=0, return_inverse=True)
    
    tet_idx_to_unique_voronoi_v_idx = {}
    processed_idx = 0
    for i in range(delaunay_tetrahedra.shape[0]):
        if i in tet_to_circumcenter_coords:
            tet_idx_to_unique_voronoi_v_idx[i] = inverse_indices[processed_idx].item()
            processed_idx += 1

    edge_to_voronoi_vertex_indices_map = defaultdict(list)
    for tet_idx, tet_point_indices in enumerate(delaunay_tetrahedra):
        if tet_idx not in tet_idx_to_unique_voronoi_v_idx:
            continue
        current_voronoi_v_idx = tet_idx_to_unique_voronoi_v_idx[tet_idx]
        for i_edge_pt in range(4):
            for j_edge_pt in range(i_edge_pt + 1, 4):
                pt_idx_A = tet_point_indices[i_edge_pt].item()
                pt_idx_B = tet_point_indices[j_edge_pt].item()
                edge = tuple(sorted((pt_idx_A, pt_idx_B)))
                if current_voronoi_v_idx not in edge_to_voronoi_vertex_indices_map[edge]:
                     edge_to_voronoi_vertex_indices_map[edge].append(current_voronoi_v_idx)
    
    voronoi_cells_faces_indices = [[] for _ in range(num_input_points)]

    for i_input_pt in range(num_input_points): 
        cell_faces_for_point_i = []
        incident_delaunay_edges = [edge for edge in edge_to_voronoi_vertex_indices_map if i_input_pt in edge]

        for delaunay_edge_tuple in incident_delaunay_edges:
            # face_global_v_indices are indices into unique_voronoi_vertices
            face_global_v_indices = edge_to_voronoi_vertex_indices_map[delaunay_edge_tuple] 
            
            if len(face_global_v_indices) < 3:
                continue 

            face_v_coords = unique_voronoi_vertices[torch.tensor(face_global_v_indices, dtype=torch.long, device=points.device)]
            
            other_pt_idx_in_edge = delaunay_edge_tuple[0] if delaunay_edge_tuple[1] == i_input_pt else delaunay_edge_tuple[1]
            
            # _order_face_vertices_3d returns sorted_coords, local_indices_for_sorting
            _, local_sorted_indices = _order_face_vertices_3d(face_v_coords, points[i_input_pt], points[other_pt_idx_in_edge])
            
            # Use local_sorted_indices to reorder the global indices for this face
            globally_indexed_ordered_face = [face_global_v_indices[k.item()] for k in local_sorted_indices]
            cell_faces_for_point_i.append(globally_indexed_ordered_face)
        
        voronoi_cells_faces_indices[i_input_pt] = cell_faces_for_point_i
            
    return voronoi_cells_faces_indices, unique_voronoi_vertices

# Unit tests are in tests/test_voronoi_from_delaunay.py
# Removed TestVoronoiFromDelaunay class from here.
