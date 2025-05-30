"""
Computes 3D Delaunay tetrahedralization using a PyTorch-based Bowyer-Watson style algorithm.

This module provides the `delaunay_triangulation_3d` function for generating a
Delaunay tetrahedralization from a set of 3D input points. It relies on geometric
predicates (`_orientation3d_pytorch` and `_in_circumsphere3d_pytorch`) imported
from `geometry_core.py` and uses the `EPSILON` constant from `geometry_core.py`
for tolerance in these geometric calculations.
"""
import torch

from .geometry_core import EPSILON, _orientation3d_pytorch, _in_circumsphere3d_pytorch

# Note: Local EPSILON_DELAUNAY_3D was removed.
# The helper predicates _orientation3d_pytorch and _in_circumsphere3d_pytorch are now imported from geometry_core.

def delaunay_triangulation_3d(points: torch.Tensor, tol: float = EPSILON) -> torch.Tensor:
    """
    Computes the 3D Delaunay tetrahedralization of a set of points.

    This implementation is based on an incremental insertion algorithm, analogous to
    the Bowyer-Watson algorithm in 3D. It starts with a super-tetrahedron
    encompassing all input points, then incrementally inserts each point. For each
    insertion, "bad" tetrahedra (those whose circumspheres contain the new point)
    are identified and removed, creating a cavity. This cavity is then
    re-tetrahedralized by forming new tetrahedra connecting the inserted point to the
    boundary faces of the cavity. Geometric predicates for orientation and
    in-circumsphere tests are sourced from `geometry_core.py`.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) representing N points in 3D.
        tol (float, optional): Tolerance for geometric predicate calculations. 
                               Defaults to `EPSILON` from `geometry_core.py`.

    Returns:
        torch.Tensor: Tensor of shape (M, 4) representing M Delaunay tetrahedra.
                      Each row contains the original indices (0 to N-1) of the four 
                      points from the input `points` tensor that form a tetrahedron.
                      Returns an empty tensor `(0,4)` if N < 4 (not enough points to 
                      form a tetrahedron).
    
    Raises:
        ValueError: If input `points` are not 3-dimensional.
    """
    n_input_points, dim = points.shape
    if dim != 3: raise ValueError("Input points must be 3-dimensional.")
    if n_input_points < 4: 
        return torch.empty((0, 4), dtype=torch.long, device=points.device)

    device = points.device
    original_dtype = points.dtype 

    # Create a super-tetrahedron that encloses all input points
    min_coords, _ = torch.min(points, dim=0)
    max_coords, _ = torch.max(points, dim=0)
    center_coords = (min_coords + max_coords) / 2.0
    coord_range = max_coords - min_coords
    max_coord_range = torch.max(coord_range)
    # Handle cases where all points are nearly coincident or lie on a small region
    if max_coord_range < tol : max_coord_range = torch.tensor(1.0, device=device, dtype=original_dtype)


    scale_factor_super = torch.max(
        torch.tensor(5.0, device=device, dtype=original_dtype) * max_coord_range, 
        torch.tensor(10.0, device=device, dtype=original_dtype)
    )
    
    sp_v0 = center_coords + torch.tensor([-1.0,-0.5,-0.25], device=device,dtype=original_dtype) * scale_factor_super
    sp_v1 = center_coords + torch.tensor([1.0,-0.5,-0.25], device=device,dtype=original_dtype) * scale_factor_super
    sp_v2 = center_coords + torch.tensor([0.0,1.0,-0.25], device=device,dtype=original_dtype) * scale_factor_super
    sp_v3 = center_coords + torch.tensor([0.0,0.0,1.25], device=device,dtype=original_dtype) * scale_factor_super
    
    super_tetra_vertices_coords = torch.stack([sp_v0, sp_v1, sp_v2, sp_v3], dim=0)
    all_points_coords = torch.cat([points, super_tetra_vertices_coords], dim=0)
    
    st_idx_start = n_input_points
    st_indices_list = [st_idx_start, st_idx_start + 1, st_idx_start + 2, st_idx_start + 3]
    
    # Ensure initial super-tetrahedron has positive orientation
    p0_st, p1_st, p2_st, p3_st = all_points_coords[st_indices_list[0]], all_points_coords[st_indices_list[1]], \
                                 all_points_coords[st_indices_list[2]], all_points_coords[st_indices_list[3]]
    if _orientation3d_pytorch(p0_st, p1_st, p2_st, p3_st, tol) < 0:
        st_indices_list[1], st_indices_list[2] = st_indices_list[2], st_indices_list[1]

    triangulation_tetra_indices = [st_indices_list] 
    
    for i_point_loop_idx in range(n_input_points):
        current_point_original_idx = i_point_loop_idx 
        current_point_coords = all_points_coords[current_point_original_idx]

        bad_tetrahedra_indices_in_list = [] 
        for tet_idx_in_list, tet_v_indices_list in enumerate(triangulation_tetra_indices):
            v_indices = torch.tensor(tet_v_indices_list, device=device, dtype=torch.long) # Ensure tensor for indexing
            v1_c,v2_c,v3_c,v4_c = all_points_coords[v_indices[0]], all_points_coords[v_indices[1]], \
                                  all_points_coords[v_indices[2]], all_points_coords[v_indices[3]]
            
            if _in_circumsphere3d_pytorch(current_point_coords, v1_c, v2_c, v3_c, v4_c, tol):
                bad_tetrahedra_indices_in_list.append(tet_idx_in_list)
        
        if not bad_tetrahedra_indices_in_list: 
            continue

        boundary_faces_v_indices = [] 
        face_counts = {} 

        for tet_idx in bad_tetrahedra_indices_in_list:
            tet_v_orig_indices = triangulation_tetra_indices[tet_idx] 
            faces_of_this_tet = [
                tuple(sorted((tet_v_orig_indices[0], tet_v_orig_indices[1], tet_v_orig_indices[2]))),
                tuple(sorted((tet_v_orig_indices[0], tet_v_orig_indices[1], tet_v_orig_indices[3]))),
                tuple(sorted((tet_v_orig_indices[0], tet_v_orig_indices[2], tet_v_orig_indices[3]))),
                tuple(sorted((tet_v_orig_indices[1], tet_v_orig_indices[2], tet_v_orig_indices[3])))
            ]
            for face_tuple_sorted_indices in faces_of_this_tet:
                face_counts[face_tuple_sorted_indices] = face_counts.get(face_tuple_sorted_indices, 0) + 1
        
        for face_tuple_sorted_indices, count in face_counts.items():
            if count == 1: 
                boundary_faces_v_indices.append(list(face_tuple_sorted_indices))

        for tet_idx in sorted(bad_tetrahedra_indices_in_list, reverse=True):
            triangulation_tetra_indices.pop(tet_idx)

        for face_v_indices_list in boundary_faces_v_indices:
            p_f0, p_f1, p_f2 = all_points_coords[face_v_indices_list[0]], \
                               all_points_coords[face_v_indices_list[1]], \
                               all_points_coords[face_v_indices_list[2]]
            
            current_orientation = _orientation3d_pytorch(p_f0, p_f1, p_f2, current_point_coords, tol)
            
            if current_orientation == 0: 
                continue 

            new_tet_final_v_indices = [face_v_indices_list[0], face_v_indices_list[1], 
                                       face_v_indices_list[2], current_point_original_idx]
            if current_orientation < 0: 
                new_tet_final_v_indices = [face_v_indices_list[0], face_v_indices_list[2], 
                                           face_v_indices_list[1], current_point_original_idx]
            
            triangulation_tetra_indices.append(new_tet_final_v_indices)

    final_triangulation_list_of_lists = []
    for tet_v_indices_list in triangulation_tetra_indices:
        is_real_tetrahedron = True
        for v_idx_in_all_points in tet_v_indices_list:
            if v_idx_in_all_points >= n_input_points: 
                is_real_tetrahedron = False
                break
        if is_real_tetrahedron:
            final_triangulation_list_of_lists.append(tet_v_indices_list)
    
    if not final_triangulation_list_of_lists:
        return torch.empty((0, 4), dtype=torch.long, device=device)
        
    return torch.tensor(final_triangulation_list_of_lists, dtype=torch.long, device=device)
