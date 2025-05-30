import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon # Alias to avoid confusion if I define a Polygon class later
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Project-specific imports
from .delaunay_2d import delaunay_triangulation_2d
from .delaunay_3d import delaunay_triangulation_3d
from .voronoi_from_delaunay import construct_voronoi_polygons_2d, construct_voronoi_polyhedra_3d
# EPSILON might be needed for small numerical checks if any, though plotting is usually tolerant
# from .geometry_core import EPSILON 

def plot_voronoi_diagram_2d(
    points: torch.Tensor, 
    voronoi_cells_verts_list: list[list[torch.Tensor]] | None = None, 
    unique_voronoi_vertices: torch.Tensor | None = None,
    bounds: torch.Tensor | None = None,
    show_voronoi_vertices: bool = True,
    ax=None,
    title: str = "2D Voronoi Diagram"
):
    """
    Plots a 2D Voronoi diagram.

    Args:
        points (torch.Tensor): Tensor of shape (N, 2) representing the input seed points.
        voronoi_cells_verts_list (list[list[torch.Tensor]] | None, optional): 
            A list where each inner list contains torch.Tensors (shape (2,)) 
            representing the ordered vertices of a Voronoi cell. 
            If None, it will be computed. Defaults to None.
        unique_voronoi_vertices (torch.Tensor | None, optional): 
            Tensor of shape (V, 2) of unique Voronoi vertex coordinates. 
            Needed if `voronoi_cells_verts_list` is provided and contains indices, 
            or if `show_voronoi_vertices` is True and this data is precomputed.
            If None and needed, it will be computed. Defaults to None.
        bounds (torch.Tensor | None, optional): Shape (2,2) [[min_x, min_y], [max_x, max_y]].
                                                If provided, draws a bounding box. Defaults to None.
        show_voronoi_vertices (bool): Whether to plot the Voronoi vertices. Defaults to True.
        ax (matplotlib.axes.Axes | None, optional): Existing axes to plot on. 
                                                   If None, a new figure and axes are created.
        title (str, optional): Title for the plot.
    """
    if points.shape[0] == 0:
        print("Warning: No points provided for Voronoi diagram.")
        return

    if voronoi_cells_verts_list is None:
        delaunay_tris = delaunay_triangulation_2d(points)
        if delaunay_tris.shape[0] == 0 and points.shape[0] >=3 : # Check if points >=3 because delaunay might be empty for <3 points
             print(f"Warning: Delaunay triangulation resulted in 0 triangles for {points.shape[0]} points. Cannot plot Voronoi cells.")
             # Plot just points if Delaunay fails for sufficient points
        elif delaunay_tris.shape[0] == 0 and points.shape[0] < 3:
             print(f"Note: Less than 3 points provided ({points.shape[0]}), no Voronoi cells to plot.")
             # This is expected, don't warn as harshly.
        
        # construct_voronoi_polygons_2d returns list of lists of vertex coordinates
        voronoi_cells_verts_list, unique_voronoi_vertices = construct_voronoi_polygons_2d(points, delaunay_tris)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.plot(points[:, 0].numpy(), points[:, 1].numpy(), 'o', label='Seed Points', color='blue')

    for i, cell_v_list in enumerate(voronoi_cells_verts_list):
        if cell_v_list and len(cell_v_list) >= 2: # Need at least 2 vertices to form a line/polygon edge
            cell_v_np = torch.stack(cell_v_list).numpy()
            polygon = MplPolygon(cell_v_np, edgecolor='black', fill=False, label=f'Cell {i}' if i==0 else None)
            ax.add_patch(polygon)
        elif cell_v_list and len(cell_v_list) == 1 : # Single Voronoi vertex for this cell (e.g. center of square)
             if show_voronoi_vertices: # Plot if it's considered a Voronoi vertex
                ax.plot(cell_v_list[0][0].item(), cell_v_list[0][1].item(), 's', color='lime', markersize=5, alpha=0.7, label='Voronoi Vertices (from single-vertex cells)' if i==0 else None)


    if show_voronoi_vertices and unique_voronoi_vertices is not None and unique_voronoi_vertices.shape[0] > 0:
        # Check if these are already plotted by the single-vertex cell logic
        # This plots all *unique* voronoi vertices.
        ax.plot(unique_voronoi_vertices[:, 0].numpy(), unique_voronoi_vertices[:, 1].numpy(), 's', 
                color='lime', markersize=5, alpha=0.7, label='Unique Voronoi Vertices' if not any(c and len(c)==1 for c in voronoi_cells_verts_list) else None)
    
    if bounds is not None:
        min_coords, max_coords = bounds[0], bounds[1]
        rect = MplPolygon(
            [[min_coords[0], min_coords[1]], [max_coords[0], min_coords[1]], 
             [max_coords[0], max_coords[1]], [min_coords[0], max_coords[1]]],
            edgecolor='gray', linestyle='--', fill=False, label='Bounds'
        )
        ax.add_patch(rect)
        ax.set_xlim(min_coords[0] - 0.1*(max_coords[0]-min_coords[0]), max_coords[0] + 0.1*(max_coords[0]-min_coords[0]))
        ax.set_ylim(min_coords[1] - 0.1*(max_coords[1]-min_coords[1]), max_coords[1] + 0.1*(max_coords[1]-min_coords[1]))
    else:
        ax.autoscale_view()

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')


def plot_voronoi_wireframe_3d(
    points: torch.Tensor,
    voronoi_cells_faces_indices: list[list[list[int]]] | None = None,
    unique_voronoi_vertices: torch.Tensor | None = None,
    ax=None,
    title: str = "3D Voronoi Diagram (Wireframe)"
):
    """
    Plots a 3D Voronoi diagram as a wireframe of cell faces.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) representing input seed points.
        voronoi_cells_faces_indices (list[list[list[int]]] | None, optional):
            A list of cells, where each cell is a list of faces, and each face
            is a list of global indices referring to `unique_voronoi_vertices`.
            If None, it will be computed. Defaults to None.
        unique_voronoi_vertices (torch.Tensor | None, optional):
            Tensor of shape (V, 3) of unique Voronoi vertex coordinates.
            If None and `voronoi_cells_faces_indices` is also None, it will be computed.
            Required if `voronoi_cells_faces_indices` is provided. Defaults to None.
        ax (matplotlib.axes.Axes | None, optional): Existing 3D axes to plot on.
                                                   If None, a new figure and 3D axes are created.
        title (str, optional): Title for the plot.
    """
    if points.shape[0] == 0:
        print("Warning: No points provided for 3D Voronoi diagram.")
        return

    if voronoi_cells_faces_indices is None or unique_voronoi_vertices is None:
        delaunay_tets = delaunay_triangulation_3d(points)
        if delaunay_tets.shape[0] == 0 and points.shape[0] >= 4:
            print(f"Warning: 3D Delaunay triangulation resulted in 0 tetrahedra for {points.shape[0]} points.")
        elif delaunay_tets.shape[0] == 0 and points.shape[0] < 4:
             print(f"Note: Less than 4 points provided ({points.shape[0]}), no Voronoi cells to plot.")

        voronoi_cells_faces_indices, unique_voronoi_vertices = construct_voronoi_polyhedra_3d(points, delaunay_tets)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure() # type: ignore
        if not hasattr(ax, 'plot3D'): #簡易的なチェック
             raise ValueError("Provided axes `ax` is not a 3D projection.")


    ax.scatter(points[:,0].numpy(), points[:,1].numpy(), points[:,2].numpy(), color='blue', label='Seed Points')

    if unique_voronoi_vertices is not None and unique_voronoi_vertices.shape[0] > 0:
        # Plot Voronoi cell edges (faces)
        all_face_collections = []
        for cell_idx, cell_faces in enumerate(voronoi_cells_faces_indices):
            cell_polygons = []
            for face_indices in cell_faces:
                if len(face_indices) >= 3: # A face must have at least 3 vertices
                    face_vertices_coords = unique_voronoi_vertices[torch.tensor(face_indices, dtype=torch.long)].numpy()
                    cell_polygons.append(face_vertices_coords)
            
            if cell_polygons:
                # Create a Poly3DCollection for this cell's faces
                # Use a semi-transparent face color and a distinct edge color
                # Assign a different color per cell for better visualization if many cells
                # Simple color cycling for now
                color_map = plt.cm.get_cmap('viridis', len(voronoi_cells_faces_indices) if len(voronoi_cells_faces_indices) > 0 else 1)
                cell_face_color = color_map(cell_idx / len(voronoi_cells_faces_indices) if len(voronoi_cells_faces_indices) > 0 else 0.5)
                
                poly_collection = Poly3DCollection(cell_polygons, 
                                                   edgecolors='k', 
                                                   linewidths=0.5,
                                                   facecolors=cell_face_color[:3] + (0.10,), # RGBA, low alpha
                                                   alpha=0.10) # Overall alpha for the collection
                ax.add_collection3d(poly_collection)
    else:
        print("Note: No unique Voronoi vertices to plot for 3D cells.")


    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title(title)
    
    # Auto-scaling can be tricky in 3D. Set sensible limits if possible, or rely on autoscale.
    if unique_voronoi_vertices is not None and unique_voronoi_vertices.shape[0] > 0:
        all_plot_points = torch.cat([points, unique_voronoi_vertices], dim=0).numpy()
    else:
        all_plot_points = points.numpy()
    
    if all_plot_points.shape[0] > 0:
        min_coords = np.min(all_plot_points, axis=0)
        max_coords = np.max(all_plot_points, axis=0)
        center = (min_coords + max_coords) / 2
        plot_range = np.max(max_coords - min_coords) * 0.6 # Take 60% of max range for each axis from center
        if plot_range < 1e-1: plot_range = 1.0 # Avoid too small plot range

        ax.set_xlim(center[0] - plot_range, center[0] + plot_range)
        ax.set_ylim(center[1] - plot_range, center[1] + plot_range)
        ax.set_zlim(center[2] - plot_range, center[2] + plot_range)

    ax.legend()

if __name__ == '__main__': # Example Usage
    # 2D Example
    example_points_2d = torch.rand((15, 2)) * 10
    plot_voronoi_diagram_2d(example_points_2d, title="Sample 2D Voronoi Diagram")
    plt.show()

    # 3D Example
    example_points_3d = torch.rand((10, 3)) * 5 
    plot_voronoi_wireframe_3d(example_points_3d, title="Sample 3D Voronoi Wireframe")
    plt.show()
