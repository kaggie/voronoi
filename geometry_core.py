"""
Core geometric primitives and algorithms implemented using PyTorch.

This module provides foundational geometric functionalities, including:
- A global EPSILON constant for numerical precision.
- Convex hull computation for 2D and 3D point sets (`ConvexHull` class).
- Polygon and polyhedron clipping algorithms (`clip_polygon_2d`, `clip_polyhedron_3d`).
- Area and volume calculation for convex shapes (via `ConvexHull` or standalone functions).
- Fundamental geometric predicates, particularly for 3D Delaunay calculations.
- Tensor normalization utilities.

The aim is to offer a PyTorch-centric toolkit for geometric operations, enabling
potential GPU acceleration and seamless integration into PyTorch-based workflows.
"""
import torch
import numpy as np # Only for type hints in future scipy.spatial.Voronoi replacement

EPSILON = 1e-7 # Global epsilon for float comparisons, updated value.

def monotone_chain_2d(points: torch.Tensor, tol: float = EPSILON):
    """
    Computes the convex hull of 2D points using the Monotone Chain algorithm.

    The Monotone Chain algorithm (also known as Andrew's algorithm) sorts points
    by x-coordinate and then constructs the upper and lower hulls of the point set.

    Args:
        points (torch.Tensor): Tensor of shape (N, 2) representing N points in 2D.
        tol (float, optional): Tolerance for floating point comparisons, particularly
                               for cross-product orientation tests. Defaults to `EPSILON`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - hull_vertices_indices (torch.Tensor): Integer tensor of shape (H,) 
              containing indices of points from the input `points` tensor that 
              form the convex hull, ordered counter-clockwise.
            - hull_simplices (torch.Tensor): Integer tensor of shape (H, 2) 
              representing the edges of the convex hull. Each row contains two indices
              referring to `hull_vertices_indices` (or directly to `points` if preferred,
              current implementation refers to original `points` indices).
              Returns empty if hull has < 2 vertices.
    """
    if not isinstance(points, torch.Tensor):
        raise ValueError("Input points must be a PyTorch tensor.")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Input points tensor must be 2-dimensional with shape (N, 2).")
    
    n_points = points.shape[0]
    if n_points < 3: # Handle cases with less than 3 points
        indices = torch.arange(n_points, device=points.device)
        if n_points == 2: 
            # Hull is the line segment itself, indices of the two points. Simplices are (0,1).
            return indices, torch.tensor([[indices[0], indices[1]]], device=points.device, dtype=torch.long)
        elif n_points == 1: 
            # Hull is the single point. No simplices.
            return indices, torch.empty((0, 2), device=points.device, dtype=torch.long)
        else: # 0 points
            return torch.empty((0,), device=points.device, dtype=torch.long), \
                   torch.empty((0, 2), device=points.device, dtype=torch.long)

    # Sort points lexicographically (by x, then by y)
    # torch.lexsort sorts based on the last key, then the second to last, etc.
    # So, to sort by x then y, pass (points[:, 1], points[:, 0])
    sorted_indices = torch.lexsort((points[:, 1], points[:, 0]))
    
    upper_hull = [] 
    lower_hull = [] 

    # Helper for orientation check (cross product)
    # Returns > 0 if (p1, p2, p3) makes a counter-clockwise turn
    # Returns < 0 if (p1, p2, p3) makes a clockwise turn
    # Returns = 0 if (p1, p2, p3) are collinear
    def cross_product_orientation(p1_idx, p2_idx, p3_idx, pts_tensor):
        p1 = pts_tensor[p1_idx]
        p2 = pts_tensor[p2_idx]
        p3 = pts_tensor[p3_idx]
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    # Build upper hull
    for i in range(n_points):
        current_original_idx = sorted_indices[i] 
        # While last two points in upper_hull and current_point make a non-left turn (or are collinear within tolerance)
        while len(upper_hull) >= 2:
            # Using original indices in cross_product_orientation for direct access to points tensor
            orientation = cross_product_orientation(upper_hull[-2], upper_hull[-1], current_original_idx, points)
            if orientation >= -tol: # Non-left turn or collinear (pop if not strictly left)
                upper_hull.pop()
            else: # Strictly left turn
                break
        upper_hull.append(current_original_idx.item()) 

    # Build lower hull
    for i in range(n_points - 1, -1, -1): # Iterate in reverse order
        current_original_idx = sorted_indices[i] 
        while len(lower_hull) >= 2:
            orientation = cross_product_orientation(lower_hull[-2], lower_hull[-1], current_original_idx, points)
            if orientation >= -tol: # Non-left turn or collinear
                lower_hull.pop()
            else: # Strictly left turn
                break
        lower_hull.append(current_original_idx.item()) 

    # Concatenate hulls: upper_hull contains all points from left-most to right-most (upper chain)
    # lower_hull contains points from right-most to left-most (lower chain, in reverse order of traversal)
    # Remove duplicate start/end points shared by both hulls.
    # The first point of upper_hull and last of lower_hull are the same (leftmost).
    # The last point of upper_hull and first of lower_hull are the same (rightmost).
    hull_vertices_indices_list = upper_hull[:-1] + lower_hull[:-1] 
    # Preserve order while making unique (though list(dict.fromkeys()) is typical Python, order is important here)
    # The construction naturally gives ordered points if concatenated correctly.
    
    # Ensure uniqueness while preserving order (important for consistent output)
    # dict.fromkeys preserves insertion order in Python 3.7+
    hull_vertices_indices_unique_ordered = list(dict.fromkeys(hull_vertices_indices_list))
    hull_vertices_indices = torch.tensor(hull_vertices_indices_unique_ordered, dtype=torch.long, device=points.device)

    # Create simplices (edges of the hull)
    num_hull_vertices = hull_vertices_indices.shape[0]
    if num_hull_vertices < 2: 
        simplices = torch.empty((0, 2), dtype=torch.long, device=points.device)
    elif num_hull_vertices == 2: # A line segment
        simplices = torch.tensor([[hull_vertices_indices[0], hull_vertices_indices[1]]], dtype=torch.long, device=points.device)
    else: # A polygon
        simplices_list = []
        for i in range(num_hull_vertices):
            simplices_list.append([hull_vertices_indices[i].item(), hull_vertices_indices[(i + 1) % num_hull_vertices].item()])
        simplices = torch.tensor(simplices_list, dtype=torch.long, device=points.device)

    return hull_vertices_indices, simplices


def monotone_chain_convex_hull_3d(points: torch.Tensor, tol: float = EPSILON):
    """
    Computes the convex hull of 3D points using an incremental construction algorithm.
    This is a complex algorithm, often a variant of Quickhull or Gift Wrapping.
    The implementation aims to find an initial simplex and incrementally add points,
    updating the hull faces.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) representing N points in 3D.
        tol (float, optional): Tolerance for floating point comparisons (e.g., for coplanarity,
                               collinearity checks). Defaults to `EPSILON`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - hull_vertices_indices (torch.Tensor): Integer tensor of shape (H,)
              containing unique indices of points from `points` that form the convex hull.
              Order is not strictly guaranteed beyond being vertices of the hull faces.
            - hull_faces (torch.Tensor): Integer tensor of shape (F, 3) representing
              the triangular faces of the convex hull. Each row contains three indices
              referring to points in the `points` tensor (original indices).
              Faces are generally oriented outwards.
              Returns empty if hull has < 3 vertices (not forming a face).
    """
    n, dim = points.shape
    device = points.device
    if dim != 3: raise ValueError("Points must be 3D.")
    if n < 4: # Not enough points to form a 3D simplex (tetrahedron)
        # For <4 points, the hull is the set of points themselves.
        # Faces are ill-defined or empty. Conventionally, return unique points and no faces.
        unique_indices = torch.unique(torch.arange(n, device=device)) # Should be just arange(n) if all unique
        return unique_indices, torch.empty((0, 3), dtype=torch.long, device=device)

    # Find initial simplex (4 non-coplanar points)
    # Step 1: Find p0 (e.g., point with min x-coordinate)
    p0_idx = torch.argmin(points[:, 0]) 
    p0 = points[p0_idx]
    
    # Step 2: Find p1 (point furthest from p0)
    dists_from_p0_sq = torch.sum((points - p0)**2, dim=1)
    dists_from_p0_sq[p0_idx] = -1 # Temporarily ignore p0 itself
    p1_idx = torch.argmax(dists_from_p0_sq) 
    p1 = points[p1_idx]
    dists_from_p0_sq[p0_idx] = 0 # Restore original distance for p0 if needed later

    # Step 3: Find p2 (point furthest from line p0-p1)
    line_vec = p1 - p0
    if torch.linalg.norm(line_vec) < tol: # p0 and p1 are coincident or very close
        # Fallback: search for any other distinct point to form a line
        for i in range(n):
            if i != p0_idx.item() and torch.linalg.norm(points[i] - p0) > tol:
                p1_idx = torch.tensor(i, device=device, dtype=torch.long); p1 = points[p1_idx]; line_vec = p1 - p0
                break
        if torch.linalg.norm(line_vec) < tol: # Still couldn't find a distinct p1 (all points likely coincident)
            return torch.unique(torch.tensor([p0_idx.item(), p1_idx.item()], device=device, dtype=torch.long)), \
                   torch.empty((0,3),dtype=torch.long,device=device)

    ap = points - p0 
    # Projection of (ap) onto line_vec: t = dot(ap, line_vec) / dot(line_vec, line_vec)
    # Projected point = p0 + t * line_vec
    # Using EPSILON in denominator for numerical stability
    t = torch.matmul(ap, line_vec) / (torch.dot(line_vec, line_vec) + EPSILON) 
    projections_on_line = p0.unsqueeze(0) + t.unsqueeze(1) * line_vec.unsqueeze(0)
    dists_sq_from_line = torch.sum((points - projections_on_line)**2, dim=1)
    dists_sq_from_line[p0_idx] = -1; dists_sq_from_line[p1_idx] = -1 # Ignore p0 and p1
    p2_idx = torch.argmax(dists_sq_from_line)
    p2 = points[p2_idx]
    dists_sq_from_line[p0_idx] = 0; dists_sq_from_line[p1_idx] = 0

    # Helper to compute normal of a plane defined by 3 points (oriented by p0,p1,p2)
    def compute_plane_normal(pt0, pt1, pt2): return torch.cross(pt1 - pt0, pt2 - pt0)
    
    normal_p0p1p2 = compute_plane_normal(p0, p1, p2)
    if torch.linalg.norm(normal_p0p1p2) < tol: # p0, p1, p2 are collinear
        # Fallback: search for a non-collinear p2
        found_non_collinear_for_plane = False
        for i in range(n):
            if i != p0_idx.item() and i != p1_idx.item(): 
                temp_normal = compute_plane_normal(p0, p1, points[i])
                if torch.linalg.norm(temp_normal) > tol: 
                    p2_idx = torch.tensor(i, device=device, dtype=torch.long); p2 = points[p2_idx]
                    normal_p0p1p2 = temp_normal
                    found_non_collinear_for_plane = True; break
        if not found_non_collinear_for_plane: # All points are collinear
            return torch.unique(torch.tensor([p0_idx.item(), p1_idx.item(), p2_idx.item()], device=device, dtype=torch.long)), \
                   torch.empty((0,3), dtype=torch.long, device=device)

    # Step 4: Find p3 (point furthest from plane p0-p1-p2)
    # Signed distance: dot(point - p0, normal_p0p1p2)
    signed_dists_from_plane = torch.matmul(points - p0.unsqueeze(0), normal_p0p1p2)
    # Zero out distances for points already in the tentative plane
    signed_dists_from_plane[p0_idx] = 0; signed_dists_from_plane[p1_idx] = 0; signed_dists_from_plane[p2_idx] = 0
    
    p3_idx = torch.argmax(torch.abs(signed_dists_from_plane))
    p3 = points[p3_idx]

    # Check if all points are coplanar
    if torch.abs(signed_dists_from_plane[p3_idx]) < tol:
        # All points are coplanar. The convex hull is a 2D polygon in 3D space.
        # We can use the 2D algorithm on projected points or return the boundary of these coplanar points.
        # For simplicity, this 3D hull function will return these points and no 3D faces.
        # A more advanced implementation might project and run a 2D hull algorithm.
        all_coplanar_indices = torch.tensor([p0_idx.item(), p1_idx.item(), p2_idx.item(), p3_idx.item()] + torch.where(torch.abs(signed_dists_from_plane) < tol)[0].tolist(), device=device, dtype=torch.long)
        return torch.unique(all_coplanar_indices), torch.empty((0, 3), dtype=torch.long, device=device)

    # Initial simplex indices. Ensure consistent orientation (e.g., p3 on positive side of plane p0-p1-p2)
    initial_simplex_indices_list = [p0_idx.item(), p1_idx.item(), p2_idx.item(), p3_idx.item()]
    if torch.dot(p3 - p0, normal_p0p1p2) < 0: # If p3 is on the "negative" side, swap p1 and p2 to flip normal
        initial_simplex_indices_list = [p0_idx.item(), p2_idx.item(), p1_idx.item(), p3_idx.item()] 

    s = initial_simplex_indices_list # s0, s1, s2 form base, s3 is apex
    vtx_coords_initial_simplex = points[torch.tensor(s, device=device)]
    
    # Initial faces of the tetrahedron, ensuring outward orientation
    # Face (s0,s1,s2) must be oriented such that s3 is on its "negative" side (inside)
    # Face (s0,s3,s1) must be oriented such that s2 is on its "negative" side
    # etc.
    faces_list_of_lists = [
        [s[0],s[1],s[2]], [s[0],s[3],s[1]], [s[1],s[3],s[2]], [s[0],s[2],s[3]] # Base, then sides
    ]
    
    # Correct orientation for initial faces (example for one face, repeat for all 4)
    # For face (s[0],s[1],s[2]), its normal should point away from s[3]
    n012 = compute_plane_normal(vtx_coords_initial_simplex[0],vtx_coords_initial_simplex[1],vtx_coords_initial_simplex[2])
    if torch.dot(vtx_coords_initial_simplex[3]-vtx_coords_initial_simplex[0], n012) > tol : # If s3 is on positive side, flip face
        faces_list_of_lists[0] = [s[0],s[2],s[1]]
    # Similar checks for other 3 faces against their respective 4th points of the simplex
    n031 = compute_plane_normal(vtx_coords_initial_simplex[0],vtx_coords_initial_simplex[3],vtx_coords_initial_simplex[1])
    if torch.dot(vtx_coords_initial_simplex[2]-vtx_coords_initial_simplex[0], n031) > tol : faces_list_of_lists[1] = [s[0],s[1],s[3]]
    n132 = compute_plane_normal(vtx_coords_initial_simplex[1],vtx_coords_initial_simplex[3],vtx_coords_initial_simplex[2])
    if torch.dot(vtx_coords_initial_simplex[0]-vtx_coords_initial_simplex[1], n132) > tol : faces_list_of_lists[2] = [s[1],s[2],s[3]]
    n023 = compute_plane_normal(vtx_coords_initial_simplex[0],vtx_coords_initial_simplex[2],vtx_coords_initial_simplex[3])
    if torch.dot(vtx_coords_initial_simplex[1]-vtx_coords_initial_simplex[0], n023) > tol : faces_list_of_lists[3] = [s[0],s[3],s[2]]


    current_faces = torch.tensor(faces_list_of_lists, dtype=torch.long, device=device)
    hull_vertex_indices_set = set(initial_simplex_indices_list) 
    
    # Mark initial simplex points as processed
    is_processed_mask = torch.zeros(n, dtype=torch.bool, device=device)
    for idx_val in hull_vertex_indices_set: is_processed_mask[idx_val] = True 
    
    candidate_points_original_indices = torch.arange(n, device=device)[~is_processed_mask]

    # Incremental construction
    for pt_orig_idx_tensor in candidate_points_original_indices:
        pt_orig_idx = pt_orig_idx_tensor.item()
        current_point_coords = points[pt_orig_idx]
        
        # Find visible faces from current_point
        visible_faces_indices_in_current_faces = [] 
        for i_face, face_v_orig_indices_tensor in enumerate(current_faces):
            p_f0 = points[face_v_orig_indices_tensor[0]]
            p_f1 = points[face_v_orig_indices_tensor[1]]
            p_f2 = points[face_v_orig_indices_tensor[2]]
            
            face_normal = compute_plane_normal(p_f0, p_f1, p_f2) 
            # Check for degenerate face normal, though ideally current_faces should not have them
            if torch.linalg.norm(face_normal) < EPSILON: continue 

            # If current_point is on the positive side of the face's plane, it's visible
            if torch.dot(current_point_coords - p_f0, face_normal) > tol:
                visible_faces_indices_in_current_faces.append(i_face)
        
        if not visible_faces_indices_in_current_faces: 
            continue # Point is inside or on the current hull, not visible from outside
            
        hull_vertex_indices_set.add(pt_orig_idx) # Add current point to the hull
        
        # Identify horizon edges (edges of visible faces that are not shared by two visible faces)
        edge_count = {} 
        for i_face_idx in visible_faces_indices_in_current_faces:
            face_orig_indices_tensor = current_faces[i_face_idx]
            face_orig_indices_list = [idx.item() for idx in face_orig_indices_tensor]
            # Define edges of the face, sorted to make them canonical for dict keys
            edges_on_face = [
                tuple(sorted((face_orig_indices_list[0], face_orig_indices_list[1]))),
                tuple(sorted((face_orig_indices_list[1], face_orig_indices_list[2]))),
                tuple(sorted((face_orig_indices_list[2], face_orig_indices_list[0])))
            ]
            for edge_tuple in edges_on_face:
                edge_count[edge_tuple] = edge_count.get(edge_tuple, 0) + 1
                
        horizon_edges_orig_indices_tuples = [edge_orig_idx_tuple for edge_orig_idx_tuple, count in edge_count.items() if count == 1]
        
        # Remove visible faces
        faces_to_keep_mask = torch.ones(current_faces.shape[0], dtype=torch.bool, device=device)
        for i_face_idx in visible_faces_indices_in_current_faces:
            faces_to_keep_mask[i_face_idx] = False
        
        temp_new_faces_list_of_lists = [f.tolist() for f in current_faces[faces_to_keep_mask]] 
        
        # Add new faces formed by the current point and horizon edges
        for edge_orig_idx_tuple in horizon_edges_orig_indices_tuples:
            # New face: (pt_orig_idx, edge_p0, edge_p1)
            # Ensure consistent orientation (e.g., outward pointing normals)
            # The new face should be oriented such that the previous hull interior is on one side
            # and the new point (and new hull interior) is on the other.
            # A common way: new_face = [pt_orig_idx, edge_idx1, edge_idx0] (reversed order of edge)
            # Check orientation against a known interior point of the old hull, or a point from the removed faces.
            # For simplicity, assume [pt_orig_idx, edge_orig_idx_tuple[1], edge_orig_idx_tuple[0]] forms outward normal
            # This needs careful handling based on how horizon edges are oriented.
            # If edge is (v0,v1), new face could be (pt, v1, v0) to ensure consistent winding if visible faces were CCW.
            temp_new_faces_list_of_lists.append([pt_orig_idx, edge_orig_idx_tuple[1], edge_orig_idx_tuple[0]])

        if not temp_new_faces_list_of_lists and n >=4 : # Should not happen if point was visible
             # This might indicate an issue or a very simple hull already formed
             if current_faces.numel() == 0 : break # All faces were visible, point encloses everything (unlikely for convex)
        
        current_faces = torch.tensor(temp_new_faces_list_of_lists, dtype=torch.long, device=device) if temp_new_faces_list_of_lists else torch.empty((0,3),dtype=torch.long,device=device)
        if current_faces.numel() == 0 and n >=4 : break # Hull collapsed or error

    # Finalize hull vertices and faces
    final_hull_vertex_indices_tensor = torch.tensor(list(hull_vertex_indices_set), dtype=torch.long, device=device)
    
    # Filter current_faces to ensure they only use vertices in the final_hull_vertex_indices_set
    # and are valid triangles (3 unique vertices).
    valid_faces_list_final = []
    if current_faces.numel() > 0:
        all_final_hull_indices_list = final_hull_vertex_indices_tensor.tolist() # For quick check
        for face_indices_tensor in current_faces:
            face_indices_list = face_indices_tensor.tolist()
            if len(set(face_indices_list)) == 3 and all(idx in all_final_hull_indices_list for idx in face_indices_list):
                valid_faces_list_final.append(face_indices_list)
    
    final_faces_tensor = torch.tensor(valid_faces_list_final, dtype=torch.long, device=device) if valid_faces_list_final else torch.empty((0,3),dtype=torch.long,device=device)
    
    # The returned vertices should be only those that appear in the final faces, if faces exist
    if final_faces_tensor.numel() > 0:
        final_hull_vertex_indices_tensor = torch.unique(final_faces_tensor.flatten())
    # Else, if no faces, final_hull_vertex_indices_tensor remains as computed from hull_vertex_indices_set
    # which is correct for degenerate cases (plane, line, point) where faces might be empty or ill-defined by this 3D func.

    return final_hull_vertex_indices_tensor, final_faces_tensor


class ConvexHull:
    """
    Computes the convex hull of a set of 2D or 3D points.

    Provides access to the hull vertices, simplices (edges in 2D, faces in 3D),
    area (polygon area in 2D, surface area in 3D), and volume (for 3D hulls).

    Attributes:
        points (torch.Tensor): The input points.
        dim (int): Dimensionality of the points (2 or 3).
        tol (float): Tolerance used for geometric computations.
        vertices (torch.Tensor | None): Indices of points forming the convex hull.
                                        For 2D, ordered counter-clockwise.
                                        For 3D, unique indices of vertices on the hull.
        simplices (torch.Tensor | None): Edges of the 2D hull (N_edges, 2) or
                                         triangular faces of the 3D hull (N_faces, 3).
                                         Indices refer to original `points` tensor.
        area (torch.Tensor): Area of the 2D convex polygon or surface area of the 3D convex polyhedron.
        volume (torch.Tensor): Volume of the 3D convex polyhedron (0.0 for 2D).
    """
    def __init__(self, points: torch.Tensor, tol: float = EPSILON):
        """
        Initializes and computes the convex hull.

        Args:
            points (torch.Tensor): Tensor of shape (N, D) representing N points,
                                   where D is the dimension (2 or 3).
            tol (float, optional): Tolerance for floating point comparisons. 
                                   Defaults to `EPSILON` from `geometry_core`.
        
        Raises:
            ValueError: If input points are not a PyTorch tensor, not 2D/3D, 
                        or have inconsistent dimensions.
        """
        if not isinstance(points, torch.Tensor): raise ValueError("Input points must be a PyTorch tensor.")
        if points.ndim != 2: raise ValueError("Input points tensor must be 2-dimensional (N, D).")
        
        self.points = points
        self.device = points.device
        self.dtype = points.dtype
        self.dim = points.shape[1]
        self.tol = tol
        
        if self.dim not in [2, 3]: raise ValueError("Only 2D and 3D points are supported.")
        
        self.vertices: torch.Tensor | None = None 
        self.simplices: torch.Tensor | None = None 
        self._area: torch.Tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)   
        self._volume: torch.Tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        self._compute_hull()

    def _compute_hull(self):
        """Internal method to dispatch to 2D or 3D hull computation."""
        if self.points.shape[0] == 0: 
            self.vertices = torch.empty((0,), dtype=torch.long, device=self.device)
            self.simplices = torch.empty((0, self.dim), dtype=torch.long, device=self.device)
            # Area and Volume remain 0.0 as initialized.
            return

        if self.dim == 2: 
            self._convex_hull_2d()
        else: # self.dim == 3
            self._convex_hull_3d()

    def _convex_hull_2d(self):
        """Computes the 2D convex hull and its area."""
        self.vertices, self.simplices = monotone_chain_2d(self.points, self.tol)
        if self.vertices is not None and self.vertices.shape[0] >= 3:
            # Shoelace formula for area of polygon defined by self.points[self.vertices]
            hull_pts_coords = self.points[self.vertices] 
            x, y = hull_pts_coords[:,0], hull_pts_coords[:,1]
            # Area = 0.5 * |sum_{i=0}^{n-1} (x_i * y_{i+1} - x_{i+1} * y_i)|
            self._area = (0.5 * torch.abs(torch.sum(x * torch.roll(y,-1) - torch.roll(x,-1) * y))).to(self.dtype)
        else:
            self._area = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self._volume = torch.tensor(0.0, device=self.device, dtype=self.dtype) # Volume is 0 for 2D

    def _compute_face_normal(self, v0_coords: torch.Tensor, v1_coords: torch.Tensor, v2_coords: torch.Tensor) -> torch.Tensor:
        """Helper to compute normal of a face defined by three vertex coordinates."""
        return torch.cross(v1_coords - v0_coords, v2_coords - v0_coords)

    def _convex_hull_3d(self):
        """Computes the 3D convex hull, its surface area, and volume."""
        self.vertices, self.simplices = monotone_chain_convex_hull_3d(self.points, self.tol) # simplices are faces (F,3)
        
        surface_area = torch.tensor(0.0, device=self.device, dtype=self.points.dtype)
        if self.simplices is not None and self.simplices.shape[0] > 0:
            # Validate simplex indices against number of points
            if torch.max(self.simplices) >= self.points.shape[0] or torch.min(self.simplices) < 0:
                # This indicates an issue with hull algorithm returning invalid indices.
                # print(f"Warning: Invalid simplex indices from 3D hull for points shape {self.points.shape}.")
                self._area = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            else:
                for face_indices in self.simplices: # Each face is [idx0, idx1, idx2]
                    p0_c,p1_c,p2_c = self.points[face_indices[0]], self.points[face_indices[1]], self.points[face_indices[2]]
                    face_normal = self._compute_face_normal(p0_c,p1_c,p2_c)
                    # Area of triangle = 0.5 * ||normal||
                    surface_area += 0.5 * torch.linalg.norm(face_normal)
        self._area = surface_area

        # Volume calculation using divergence theorem: sum over faces of (ref_pt - p0) . (p1 - p0) x (p2 - p0) / 6
        # Requires consistently oriented faces (e.g., outward pointing normals)
        # and a reference point (e.g., centroid or any point on the hull).
        total_signed_volume = torch.tensor(0.0, device=self.device, dtype=self.points.dtype)
        if self.vertices is not None and self.vertices.numel() >= 4 and \
           self.simplices is not None and self.simplices.shape[0] > 0:
            
            # Use the first vertex of the hull as the reference point for tetrahedron summation
            ref_pt_coords = self.points[self.vertices[0]] 
            
            for face_indices in self.simplices:
                p0_c = self.points[face_indices[0]]
                p1_c = self.points[face_indices[1]]
                p2_c = self.points[face_indices[2]]
                # Volume of tetrahedron (ref_pt, p0, p1, p2)
                # = 1/6 * | dot(p0-ref, cross(p1-ref, p2-ref)) |
                # Sign matters if faces are consistently oriented.
                # Assuming faces from monotone_chain_convex_hull_3d are oriented outwards.
                total_signed_volume += torch.dot(p0_c - ref_pt_coords, torch.cross(p1_c - ref_pt_coords, p2_c - ref_pt_coords))
            
            self._volume = torch.abs(total_signed_volume) / 6.0
        else:
            self._volume = torch.tensor(0.0, device=self.device, dtype=self.dtype)


    @property
    def area(self) -> torch.Tensor: 
        """
        Area of the 2D convex polygon or surface area of the 3D convex polyhedron.
        """
        return self._area

    @property
    def volume(self) -> torch.Tensor: 
        """
        Volume of the 3D convex polyhedron. Returns 0.0 for 2D cases.
        """
        return self._volume

# --- Polygon Clipping (Sutherland-Hodgman) ---

def _sutherland_hodgman_is_inside(point: torch.Tensor, edge_type: str, clip_value: float) -> bool:
    """
    Checks if a point is 'inside' a given clip edge (for Sutherland-Hodgman).
    'Inside' means on the side of the edge that keeps the point within the clip area.

    Args:
        point (torch.Tensor): The 2D point to check (shape (2,)).
        edge_type (str): Type of clipping edge ('left', 'top', 'right', 'bottom').
        clip_value (float): The coordinate value defining the clip edge 
                            (e.g., x_min for 'left', y_max for 'top').
    Returns:
        bool: True if the point is inside or on the edge, False otherwise.
    """
    if edge_type == 'left': return point[0] >= clip_value  # x >= x_min
    elif edge_type == 'top': return point[1] <= clip_value   # y <= y_max
    elif edge_type == 'right': return point[0] <= clip_value # x <= x_max
    elif edge_type == 'bottom': return point[1] >= clip_value # y >= y_min
    return False # Should not happen with valid edge_type

def _sutherland_hodgman_intersect(
    p1: torch.Tensor, p2: torch.Tensor, 
    clip_edge_p1: torch.Tensor, clip_edge_p2: torch.Tensor
) -> torch.Tensor:
    """
    Computes the intersection of line segment p1-p2 with an infinite clip edge.
    The clip edge is defined by clip_edge_p1 and clip_edge_p2.
    This is a standard line segment intersection formula.

    Args:
        p1 (torch.Tensor): First vertex of the polygon edge (shape (2,)).
        p2 (torch.Tensor): Second vertex of the polygon edge (shape (2,)).
        clip_edge_p1 (torch.Tensor): First vertex defining the clipping edge (shape (2,)).
        clip_edge_p2 (torch.Tensor): Second vertex defining the clipping edge (shape (2,)).

    Returns:
        torch.Tensor: The intersection point (shape (2,)). If lines are parallel and
                      denominator is zero, returns p2 as a fallback (though ideally,
                      this case means no unique intersection or segment lies on edge).
    """
    # Using torch.float64 for precision in intersection calculation
    x1, y1 = p1[0].to(torch.float64), p1[1].to(torch.float64)
    x2, y2 = p2[0].to(torch.float64), p2[1].to(torch.float64)
    x3, y3 = clip_edge_p1[0].to(torch.float64), clip_edge_p1[1].to(torch.float64)
    x4, y4 = clip_edge_p2[0].to(torch.float64), clip_edge_p2[1].to(torch.float64)

    # Denominator for intersection formula
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if torch.abs(denominator) < EPSILON: # Lines are parallel or collinear
        # Fallback: if parallel, effectively no intersection in SH algorithm context,
        # or edge cases are handled by inside/outside checks.
        # Returning p2 can be problematic if it's outside.
        # However, SH algorithm usually relies on one point in, one out for this.
        # A more robust handler might be needed if segments can lie on the clip edge.
        return p2 # Or raise error, or return None if that's handled by caller
    
    # Numerator for parameter t (along line p1-p2)
    t_numerator = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    # Numerator for parameter u (along line clip_edge_p1-clip_edge_p2) is not needed here.
    
    # Calculate intersection point coordinates
    intersect_x = x1 + (t_numerator / denominator) * (x2 - x1)
    intersect_y = y1 + (t_numerator / denominator) * (y2 - y1)
    
    return torch.tensor([intersect_x, intersect_y], dtype=p1.dtype, device=p1.device)

def clip_polygon_2d(
    polygon_vertices: torch.Tensor, 
    clip_bounds: torch.Tensor
) -> torch.Tensor:
    """
    Clips a 2D polygon against an axis-aligned rectangular bounding box using the
    Sutherland-Hodgman algorithm.

    Args:
        polygon_vertices (torch.Tensor): A tensor of shape (N, 2) representing the
                                         ordered vertices of the input polygon.
        clip_bounds (torch.Tensor): A tensor of shape (2, 2) defining the rectangular
                                    clipping window: [[min_x, min_y], [max_x, max_y]].

    Returns:
        torch.Tensor: A tensor of shape (M, 2) representing the ordered vertices
                      of the clipped polygon. Returns an empty tensor (0, 2)
                      if the polygon is entirely outside the clip_bounds or
                      results in a degenerate shape (e.g., less than 3 vertices after clipping).
    """
    if not isinstance(polygon_vertices, torch.Tensor) or polygon_vertices.ndim != 2 or polygon_vertices.shape[1] != 2:
        raise ValueError("polygon_vertices must be a tensor of shape (N, 2).")
    if polygon_vertices.shape[0] == 0: 
        return torch.empty((0,2), dtype=polygon_vertices.dtype, device=polygon_vertices.device)
    if not isinstance(clip_bounds, torch.Tensor) or clip_bounds.shape != (2,2):
        raise ValueError("clip_bounds must be a tensor of shape (2, 2) [[min_x, min_y], [max_x, max_y]].")

    min_x, min_y = clip_bounds[0,0], clip_bounds[0,1]
    max_x, max_y = clip_bounds[1,0], clip_bounds[1,1]
    if not (min_x <= max_x and min_y <= max_y): 
        raise ValueError("Clip bounds min must be less than or equal to max for each dimension.")

    device = polygon_vertices.device
    dtype = polygon_vertices.dtype
    
    # Define clip edges: (edge_type, clip_value, clip_edge_p1_for_intersection, clip_edge_p2_for_intersection)
    # These p1,p2 for intersection define the infinite line of the clipping edge.
    clip_edges_params = [
        ('left',   min_x, torch.tensor([min_x, min_y], device=device, dtype=dtype), torch.tensor([min_x, max_y], device=device, dtype=dtype)),
        ('top',    max_y, torch.tensor([min_x, max_y], device=device, dtype=dtype), torch.tensor([max_x, max_y], device=device, dtype=dtype)),
        ('right',  max_x, torch.tensor([max_x, max_y], device=device, dtype=dtype), torch.tensor([max_x, min_y], device=device, dtype=dtype)),
        ('bottom', min_y, torch.tensor([max_x, min_y], device=device, dtype=dtype), torch.tensor([min_x, min_y], device=device, dtype=dtype))
    ]
    
    # Convert to list of Python lists for intermediate processing if original is tensor
    output_vertices_py_list = polygon_vertices.tolist() 

    for edge_type, clip_val, clip_e_p1, clip_e_p2 in clip_edges_params:
        if not output_vertices_py_list: break # Polygon fully clipped
        
        input_verts_current_stage = [torch.tensor(v, dtype=dtype, device=device) for v in output_vertices_py_list]
        output_vertices_py_list = [] # Reset for this clipping edge
        
        if not input_verts_current_stage: break # Should not happen if check above works

        S_pt = input_verts_current_stage[-1] # Start with the last vertex to form edge with the first
        for P_pt in input_verts_current_stage:
            S_is_inside = _sutherland_hodgman_is_inside(S_pt, edge_type, clip_val)
            P_is_inside = _sutherland_hodgman_is_inside(P_pt, edge_type, clip_val)

            if S_is_inside and P_is_inside: # Case 1: Both points inside -> Add P
                output_vertices_py_list.append(P_pt.tolist())
            elif S_is_inside and not P_is_inside: # Case 2: S in, P out (outgoing edge) -> Add intersection
                output_vertices_py_list.append(_sutherland_hodgman_intersect(S_pt, P_pt, clip_e_p1, clip_e_p2).tolist())
            elif not S_is_inside and P_is_inside: # Case 4: S out, P in (incoming edge) -> Add intersection, then P
                output_vertices_py_list.append(_sutherland_hodgman_intersect(S_pt, P_pt, clip_e_p1, clip_e_p2).tolist())
                output_vertices_py_list.append(P_pt.tolist())
            # Case 3: Both S and P out -> Add nothing
            S_pt = P_pt # Advance S to P for next edge
            
    if not output_vertices_py_list: 
        return torch.empty((0,2), dtype=dtype, device=device)
    
    # Remove duplicate consecutive vertices that might arise from clipping precision issues
    # especially if a vertex lies very close to a clipping edge.
    final_clipped_verts_py_dedup = []
    if len(output_vertices_py_list) > 1:
        final_clipped_verts_py_dedup.append(output_vertices_py_list[0])
        for i in range(1, len(output_vertices_py_list)):
            # Compare current vertex with the last one added to the de-duplicated list
            # Using a slightly larger tolerance for this de-duplication step
            if not torch.allclose(torch.tensor(output_vertices_py_list[i], device=device, dtype=dtype), 
                                  torch.tensor(final_clipped_verts_py_dedup[-1], device=device, dtype=dtype), atol=EPSILON*10):
                final_clipped_verts_py_dedup.append(output_vertices_py_list[i])
        
        # Check if the last and first points of the potentially closed polygon are duplicates
        if len(final_clipped_verts_py_dedup) > 1 and \
           torch.allclose(torch.tensor(final_clipped_verts_py_dedup[0], device=device, dtype=dtype), 
                          torch.tensor(final_clipped_verts_py_dedup[-1], device=device, dtype=dtype), atol=EPSILON*10):
            final_clipped_verts_py_dedup.pop() # Remove redundant closing vertex
            
        if not final_clipped_verts_py_dedup: 
            return torch.empty((0,2), dtype=dtype, device=device)
        return torch.tensor(final_clipped_verts_py_dedup, dtype=dtype, device=device)
        
    elif output_vertices_py_list: # Single vertex remaining (degenerate)
        return torch.tensor(output_vertices_py_list, dtype=dtype, device=device)
    else: # No vertices left
        return torch.empty((0,2), dtype=dtype, device=device)

# --- Polyhedron Clipping ---

def _point_plane_signed_distance(
    point_coords: torch.Tensor, 
    plane_normal: torch.Tensor, 
    plane_d_offset: torch.Tensor # Assuming plane_d_offset is scalar: dot(n,x) - d = 0
) -> torch.Tensor:
    """
    Computes signed distance from a point to a plane. Plane: dot(n,x) - d = 0.
    Positive distance indicates point is on the side of the normal.

    Args:
        point_coords (torch.Tensor): Coordinates of the point (Dim,).
        plane_normal (torch.Tensor): Normal vector of the plane (Dim,). Assumed normalized.
        plane_d_offset (torch.Tensor): Offset 'd' of the plane (scalar).

    Returns:
        torch.Tensor: Scalar signed distance.
    """
    return torch.dot(point_coords, plane_normal) - plane_d_offset

def _segment_plane_intersection(
    p1_coords: torch.Tensor, p2_coords: torch.Tensor, 
    plane_normal: torch.Tensor, plane_d_offset: torch.Tensor, 
    tol: float = EPSILON
) -> torch.Tensor | None:
    """
    Computes intersection of line segment p1-p2 with a plane dot(n,x) - d = 0.

    Args:
        p1_coords (torch.Tensor): Start point of segment (Dim,).
        p2_coords (torch.Tensor): End point of segment (Dim,).
        plane_normal (torch.Tensor): Normal vector of the plane (Dim,).
        plane_d_offset (torch.Tensor): Offset 'd' of the plane (scalar).
        tol (float, optional): Tolerance for checking parallel lines and if 
                               intersection lies within segment. Defaults to EPSILON.

    Returns:
        torch.Tensor | None: Intersection point (Dim,) if it exists within the segment 
                             (inclusive of endpoints within tolerance). None otherwise.
    """
    dp = p2_coords - p1_coords 
    den = torch.dot(dp, plane_normal) 

    if torch.abs(den) < tol: # Segment parallel to plane (or very close to parallel)
        # Optionally, could check if p1 is on the plane:
        # if torch.abs(torch.dot(p1_coords, plane_normal) - plane_d_offset) < tol: return p1 (lies on plane)
        return None # No unique intersection, or segment lies in plane.

    # Parameter t for intersection point P(t) = p1 + t * (p2-p1)
    t = (plane_d_offset - torch.dot(p1_coords, plane_normal)) / den
    
    # Check if intersection is within segment [p1, p2] (t in [0,1] within tolerance)
    if -tol <= t <= 1.0 + tol: 
        return p1_coords + t * dp
    return None

def clip_polyhedron_3d(
    input_poly_vertices_coords: torch.Tensor, 
    bounding_box_minmax: torch.Tensor, 
    tol: float = EPSILON
) -> torch.Tensor:
    """
    Clips a 3D convex polyhedron (defined by its vertices) against an axis-aligned bounding box.

    The method collects vertices of the input polyhedron that are inside the box,
    and all intersection points of the polyhedron's edges (derived from its convex hull)
    with the bounding box planes. The convex hull of this combined set of points forms
    the clipped polyhedron.

    Args:
        input_poly_vertices_coords (torch.Tensor): Tensor of shape (N, 3) representing
                                                   the vertices of the input convex polyhedron.
        bounding_box_minmax (torch.Tensor): Tensor of shape (2, 3), where row 0 contains
                                            [min_x, min_y, min_z] and row 1 contains
                                            [max_x, max_y, max_z] of the bounding box.
        tol (float, optional): Tolerance for geometric computations (e.g., point containment,
                               intersection checks). Defaults to `EPSILON`.

    Returns:
        torch.Tensor: Tensor of shape (M, 3) representing the vertices of the clipped
                      polyhedron. If the polyhedron is fully outside the box or clipping
                      results in a degenerate shape (less than 4 vertices for a 3D hull),
                      an empty tensor or a tensor with fewer than 4 vertices may be returned.
    """
    if not (isinstance(input_poly_vertices_coords, torch.Tensor) and \
            input_poly_vertices_coords.ndim == 2 and input_poly_vertices_coords.shape[1] == 3):
        raise ValueError("input_poly_vertices_coords must be a tensor of shape (N, 3).")
    
    if input_poly_vertices_coords.shape[0] == 0: 
        return torch.empty_like(input_poly_vertices_coords)

    if not (isinstance(bounding_box_minmax, torch.Tensor) and bounding_box_minmax.shape == (2,3)):
        raise ValueError("bounding_box_minmax must be a tensor of shape (2, 3).")

    device = input_poly_vertices_coords.device 
    dtype = input_poly_vertices_coords.dtype
    
    min_coords,max_coords = bounding_box_minmax[0].to(device=device, dtype=dtype), \
                            bounding_box_minmax[1].to(device=device, dtype=dtype)
    if not (torch.all(min_coords <= max_coords)):
        raise ValueError("Bounding box min_coords must be less than or equal to max_coords for each dimension.")

    # Optimization: If input has too few points for a 3D shape, just filter those inside bounds
    if input_poly_vertices_coords.shape[0] < 4:
        is_inside_box = torch.ones(input_poly_vertices_coords.shape[0], dtype=torch.bool, device=device)
        for dim_idx in range(3):
            is_inside_box &= (input_poly_vertices_coords[:, dim_idx] >= min_coords[dim_idx] - tol)
            is_inside_box &= (input_poly_vertices_coords[:, dim_idx] <= max_coords[dim_idx] + tol)
        return input_poly_vertices_coords[is_inside_box]

    # Define the 6 planes of the bounding box (normal, d_offset), normals point inwards.
    # Plane equation: dot(normal, x) - d_offset = 0. Inside if >= 0.
    planes_params = [
        (torch.tensor([1,0,0],dtype=dtype,device=device),  min_coords[0]), # x >= min_x  =>  1*x - min_x >= 0
        (torch.tensor([-1,0,0],dtype=dtype,device=device),-max_coords[0]),# x <= max_x  => -1*x + max_x >= 0 => -1*x - (-max_x) >=0
        (torch.tensor([0,1,0],dtype=dtype,device=device),  min_coords[1]), # y >= min_y
        (torch.tensor([0,-1,0],dtype=dtype,device=device),-max_coords[1]),# y <= max_y
        (torch.tensor([0,0,1],dtype=dtype,device=device),  min_coords[2]), # z >= min_z
        (torch.tensor([0,0,-1],dtype=dtype,device=device),-max_coords[2]) # z <= max_z
    ]
    
    candidate_v_list = []
    # 1. Add original vertices of the input polyhedron if they are inside the bounding box
    for v_idx in range(input_poly_vertices_coords.shape[0]):
        v_coord = input_poly_vertices_coords[v_idx]
        is_fully_inside = True
        for plane_normal, plane_d_offset in planes_params:
            if _point_plane_signed_distance(v_coord, plane_normal, plane_d_offset) < -tol:
                is_fully_inside = False
                break
        if is_fully_inside:
            candidate_v_list.append(v_coord)
            
    # 2. Add intersection points of the polyhedron's edges with the bounding box planes
    try:
        initial_hull = ConvexHull(input_poly_vertices_coords, tol=tol) # Get edges from hull
        if initial_hull.simplices is not None and initial_hull.simplices.numel() > 0:
            # Simplices are faces (triangles). Extract unique edges.
            unique_edges_indices = set()
            for face_indices in initial_hull.simplices: # face_indices are for original points
                fi_list = face_indices.tolist()
                unique_edges_indices.update([
                    tuple(sorted((fi_list[0],fi_list[1]))), 
                    tuple(sorted((fi_list[1],fi_list[2]))), 
                    tuple(sorted((fi_list[2],fi_list[0])))
                ])
            
            for edge_idx_tuple in unique_edges_indices:
                p1_c = input_poly_vertices_coords[edge_idx_tuple[0]]
                p2_c = input_poly_vertices_coords[edge_idx_tuple[1]]
                
                for pl_norm, pl_d_off in planes_params:
                    intersect_pt_c = _segment_plane_intersection(p1_c, p2_c, pl_norm, pl_d_off, tol)
                    if intersect_pt_c is not None:
                        # Verify this intersection point is itself within the *entire* bounding box
                        # (as segment can intersect a plane outside the actual box face region)
                        is_intersection_valid_for_box = True
                        for chk_pl_norm, chk_pl_d_off in planes_params:
                            if _point_plane_signed_distance(intersect_pt_c, chk_pl_norm, chk_pl_d_off) < -tol:
                                is_intersection_valid_for_box = False; break
                        if is_intersection_valid_for_box:
                             candidate_v_list.append(intersect_pt_c)
    except (ValueError, RuntimeError): 
        # If initial hull computation fails (e.g., degenerate input), proceed with vertices found so far.
        # This might happen if input_poly_vertices_coords are already coplanar/collinear.
        pass 

    if not candidate_v_list: 
        return torch.empty((0,3), dtype=dtype, device=device)
    
    # Create a tensor from the list of candidates and remove duplicates
    stacked_candidates = torch.stack(candidate_v_list)
    unique_final_cand_coords = torch.unique(stacked_candidates, dim=0)
    
    # If fewer than 4 unique points, cannot form a 3D polyhedron. Return these points.
    # (Could be a 2D polygon, line, or single point if fully clipped or input was degenerate).
    if unique_final_cand_coords.shape[0] < 4: 
        return unique_final_cand_coords
    
    # 3. Compute the convex hull of all collected candidate points
    try:
        clipped_hull = ConvexHull(unique_final_cand_coords, tol=tol)
        # Return the vertices that form the *final* clipped hull.
        # These are indices into unique_final_cand_coords.
        if clipped_hull.vertices is not None and clipped_hull.vertices.numel() > 0:
            return unique_final_cand_coords[clipped_hull.vertices]
        else: # Hull computation resulted in no vertices (e.g. all points became collinear/coplanar again after unique)
             return unique_final_cand_coords if unique_final_cand_coords.shape[0]>0 else torch.empty((0,3),dtype=dtype,device=device)

    except (ValueError,RuntimeError): 
        # If final hull fails (e.g., points are coplanar after clipping), return the unique candidates.
        # This might mean the clipped shape is 2D or 1D.
        return unique_final_cand_coords if unique_final_cand_coords.shape[0]>0 else torch.empty((0,3),dtype=dtype,device=device)


def compute_polygon_area(points_coords: torch.Tensor) -> float:
    """
    Computes the area of a 2D polygon, defined by its vertices.
    Assumes vertices are ordered if the direct Shoelace formula is used.
    This implementation uses ConvexHull, so order is handled internally for area.

    Args:
        points_coords (torch.Tensor): A PyTorch tensor of shape (N, 2) representing 
                                      the polygon's N vertices.

    Returns:
        float: The area of the polygon's convex hull. Returns 0.0 if N < 3 or
               if points are collinear.
    """
    if not (isinstance(points_coords,torch.Tensor) and points_coords.ndim==2 and points_coords.shape[1]==2):
        raise ValueError("Input points_coords must be a PyTorch tensor of shape (N, 2).")
    if points_coords.shape[0]<3: return 0.0 # Not enough points for an area
    try: 
        hull = ConvexHull(points_coords,tol=EPSILON)
    except ValueError: # Handles cases where ConvexHull itself might raise error for degenerate inputs
        return 0.0 
    area_val = hull.area.item()
    # Check if area is extremely small (close to zero within tolerance)
    return 0.0 if abs(area_val) < EPSILON**2 else area_val # Use a tighter check for area being zero

def compute_convex_hull_volume(points_coords: torch.Tensor) -> float:
    """
    Computes the volume of the convex hull of a set of 3D points.

    Args:
        points_coords (torch.Tensor): A PyTorch tensor of shape (N, 3) representing 
                                      N points in 3D space.

    Returns:
        float: The volume of the convex hull. Returns 0.0 if N < 4 or if points
               are coplanar or collinear, leading to a degenerate hull.
    """
    if not (isinstance(points_coords,torch.Tensor) and points_coords.ndim==2 and points_coords.shape[1]==3):
        raise ValueError("Input points_coords must be a PyTorch tensor of shape (N, 3).")
    if points_coords.shape[0]<4: return 0.0 # Not enough points for a 3D volume
    try: 
        hull = ConvexHull(points_coords,tol=EPSILON)
    except ValueError:
        return 0.0
    vol_val = hull.volume.item()
    return 0.0 if abs(vol_val) < EPSILON**3 else vol_val # Use a tighter check for volume being zero

def normalize_weights(weights: torch.Tensor, target_sum: float = 1.0, tol: float = EPSILON) -> torch.Tensor:
    """ 
    Normalizes a tensor of weights to sum to a target value (default 1.0).
    Negative weights (within -tol) are clamped to zero.

    Args:
        weights (torch.Tensor): Input tensor of weights (1D, shape (N,)).
        target_sum (float, optional): Desired sum of the normalized weights. Defaults to 1.0.
        tol (float, optional): Tolerance for handling negative weights and sum checks. 
                               Defaults to `EPSILON`.

    Returns:
        torch.Tensor: Normalized weights with the same shape as input, summing to target_sum.
                      Returns uniform weights if sum of clamped weights is less than tolerance.

    Raises:
        TypeError: If weights is not a PyTorch tensor.
        AssertionError: If weights is not a 1D tensor.
        ValueError: If weights contains values less than -tol (more negative than tolerance allows).
    """
    if not isinstance(weights, torch.Tensor): raise TypeError("Input weights must be a PyTorch tensor.")
    if weights.numel()==0: return torch.empty_like(weights) # Handle empty tensor input
    if weights.dim()!=1: raise AssertionError("Weights must be a 1D tensor.")
    
    # Check for significantly negative weights
    if torch.any(weights < -tol): 
         raise ValueError(f"Weights must be non-negative (or within -{tol} tolerance). Found: {weights[weights<-tol]}")
    
    clamped_weights = torch.clamp(weights, min=0.0) # Clamp small negatives (within tol) to zero
    weight_sum = torch.sum(clamped_weights)
    
    if weight_sum < tol: 
        # If sum is effectively zero, return uniform distribution to avoid division by zero
        # and to handle cases where all weights become zero after clamping.
        # print(f"Warning: Sum of weights ({weight_sum.item()}) after clamping is less than tolerance ({tol}). Returning uniform weights.")
        n_weights = weights.shape[0]
        return torch.full_like(weights, (target_sum / n_weights) if n_weights > 0 else 0.0)
        
    return clamped_weights * (target_sum / weight_sum)

# --- Geometric Predicates (Primarily for 3D Delaunay) ---
def _orientation3d_pytorch(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor, tol: float = EPSILON) -> int:
    """
    Computes the orientation of point p4 relative to the plane defined by p1, p2, p3.
    Uses the sign of the determinant of a matrix formed by vectors (p2-p1, p3-p1, p4-p1).

    Args:
        p1, p2, p3, p4 (torch.Tensor): Tensors of shape (3,) representing 3D points.
        tol (float, optional): Tolerance for floating point comparisons to determine coplanarity.
                               Defaults to `EPSILON`.
    Returns:
        int: 
            0 if points are coplanar (within tolerance).
            1 if p4 is on one side of the plane (e.g., "positive" orientation, forming a 
              positively signed volume for tetrahedron p1-p2-p3-p4, assuming p1-p2-p3 is CCW from p4).
           -1 if p4 is on the other side ("negative" orientation).
    """
    mat = torch.stack((p2 - p1, p3 - p1, p4 - p1), dim=0)
    det_val = torch.det(mat.to(dtype=torch.float64)) # Use float64 for precision
    if torch.abs(det_val) < tol: return 0
    return 1 if det_val > 0 else -1

def _in_circumsphere3d_pytorch(
    p_check: torch.Tensor, 
    t1: torch.Tensor, t2: torch.Tensor, t3: torch.Tensor, t4: torch.Tensor, 
    tol: float = EPSILON
) -> bool:
    """
    Checks if point `p_check` is strictly inside the circumsphere of the tetrahedron (t1,t2,t3,t4).

    This predicate is based on the sign of a 5x5 determinant involving the coordinates
    of the five points and their squared magnitudes. The interpretation of the sign
    depends on the orientation of the tetrahedron (t1,t2,t3,t4).

    Args:
        p_check (torch.Tensor): The point to check (shape (3,)).
        t1, t2, t3, t4 (torch.Tensor): Vertices of the tetrahedron (shape (3,)).
        tol (float, optional): Tolerance for determinant calculations to handle floating point
                               inaccuracies. Defaults to `EPSILON`.
    Returns:
        bool: True if `p_check` is strictly inside the circumsphere.
              False if on or outside, or if the tetrahedron is degenerate (coplanar).
    """
    # Use float64 for precision in matrix construction and determinant
    points_for_mat = [t.to(dtype=torch.float64) for t in [t1, t2, t3, t4, p_check]]
    
    mat_rows = []
    for pt_i in points_for_mat:
        sum_sq = torch.sum(pt_i**2) # Squared magnitude ||pt_i||^2
        mat_rows.append(torch.cat((pt_i, sum_sq.unsqueeze(0), torch.tensor([1.0], dtype=torch.float64, device=p_check.device))))
    
    mat_5x5 = torch.stack(mat_rows, dim=0) # Shape (5,5)
    
    # Orientation of the base tetrahedron (t1,t2,t3,t4)
    # This is crucial for interpreting the sign of the 5x5 determinant.
    # A common convention: if Orient(t1,t2,t3,t4) > 0, then p_check is inside if det(mat_5x5) > 0.
    # The _orientation3d_pytorch gives sign of det([t2-t1; t3-t1; t4-t1]).
    orient_val = _orientation3d_pytorch(t1, t2, t3, t4, tol)

    if orient_val == 0: # Degenerate tetrahedron (vertices are coplanar)
        return False # Circumsphere is ill-defined or infinite; point cannot be strictly "inside".
    
    circumsphere_det_val = torch.det(mat_5x5)
    
    # The point p_check is inside if (orientation_determinant * circumsphere_determinant) > tolerance.
    # This test needs to be strictly positive for "strictly inside".
    # If orient_val > 0, then circumsphere_det_val must be > tol.
    # If orient_val < 0, then circumsphere_det_val must be < -tol.
    # This is equivalent to (orient_val * circumsphere_det_val) > some_positive_epsilon_scaled_value.
    # Using `tol` directly here might be too strict if circumsphere_det_val is small but correct sign.
    # For strict insideness, it should be > 0 (or < 0 depending on orientation factor).
    # A common formulation: result > tol for strict inside.
    return (orient_val * circumsphere_det_val) > tol
