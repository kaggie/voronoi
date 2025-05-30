import torch
import unittest
import matplotlib.pyplot as plt

# Assuming the main project structure allows this type of relative import
from ..voronoi_plotting import plot_voronoi_diagram_2d, plot_voronoi_wireframe_3d

class TestVoronoiPlotting(unittest.TestCase):
    """
    Smoke tests for Voronoi plotting functions.
    These tests primarily check if the plotting functions execute without raising errors
    for basic inputs. They do not verify the correctness of the visual output.
    """

    @classmethod
    def tearDownClass(cls):
        """Close all Matplotlib figures after all tests in the class have run."""
        plt.close('all')

    def test_plot_2d_smoke(self):
        """Smoke test for plot_voronoi_diagram_2d."""
        points = torch.tensor([[0.,0.], [1.,0.], [0.5,0.866], [1.,1.]], dtype=torch.float32) # A few points
        try:
            plot_voronoi_diagram_2d(points, title="2D Smoke Test")
            # Test with precomputed data (minimal, might not form many cells)
            # delaunay_tris = delaunay_triangulation_2d(points)
            # cells, verts = construct_voronoi_polygons_2d(points, delaunay_tris)
            # plot_voronoi_diagram_2d(points, voronoi_cells_verts_list=cells, unique_voronoi_vertices=verts, title="2D Smoke Test Precomputed")
            
            # Test with bounds
            bounds_2d = torch.tensor([[0.,0.], [1.,1.]])
            plot_voronoi_diagram_2d(points, bounds=bounds_2d, title="2D Smoke Test with Bounds")
            
            # Test with very few points (should not error)
            plot_voronoi_diagram_2d(points[:2,:], title="2D Smoke Test Few Points")
            plot_voronoi_diagram_2d(points[:1,:], title="2D Smoke Test One Point")
            plot_voronoi_diagram_2d(torch.empty((0,2)), title="2D Smoke Test No Points")


        except Exception as e:
            self.fail(f"plot_voronoi_diagram_2d raised an exception: {e}")
        finally:
            plt.close('all') # Close figures created in this test

    def test_plot_3d_smoke(self):
        """Smoke test for plot_voronoi_wireframe_3d."""
        points = torch.tensor([
            [0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.], [0.5,0.5,0.5]
        ], dtype=torch.float32)
        try:
            plot_voronoi_wireframe_3d(points, title="3D Smoke Test")
            
            # Test with very few points
            plot_voronoi_wireframe_3d(points[:3,:], title="3D Smoke Test Few Points")
            plot_voronoi_wireframe_3d(points[:1,:], title="3D Smoke Test One Point")
            plot_voronoi_wireframe_3d(torch.empty((0,3)), title="3D Smoke Test No Points")

        except Exception as e:
            self.fail(f"plot_voronoi_wireframe_3d raised an exception: {e}")
        finally:
            plt.close('all') # Close figures created in this test

if __name__ == '__main__':
    unittest.main()
