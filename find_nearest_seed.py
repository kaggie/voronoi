"""
Finds the nearest seed point for each query point using PyTorch.

This module provides an efficient way to determine which seed point (from a given
set of seed points) is closest to each query point (from a given set of query
points). It supports multi-dimensional points and can return either Euclidean
or squared Euclidean distances.
"""
import torch
import unittest
import math # For torch.sqrt if not directly used, but good for context

# --- findNearestSeed Function ---

def find_nearest_seed(query_points: torch.Tensor, seeds: torch.Tensor, return_squared_distance: bool = False):
    """
    Finds the nearest seed for each query point using batch operations.

    Args:
        query_points (torch.Tensor): A tensor of shape (Q, Dim) representing Q query points
                                     in Dim-dimensional space. Can also be (Dim,) for a 
                                     single query point, which will be unsqueezed.
        seeds (torch.Tensor): A tensor of shape (S, Dim) representing S seed points
                              in Dim-dimensional space. Can also be (Dim,) for a single
                              seed point, which will be unsqueezed.
        return_squared_distance (bool, optional): 
            If False (default), returns Euclidean distances (requires a sqrt operation).
            If True, returns squared Euclidean distances.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - nearest_seed_indices (torch.Tensor): Shape (Q,). Integer indices (dtype long) 
                                                   into the `seeds` tensor, indicating the 
                                                   closest seed for each query point. 
                                                   If no seeds are provided, indices are -1.
            - distances (torch.Tensor): Shape (Q,). Distances (Euclidean by default, or squared)
                                        to the nearest seed for each query point.
                                        If no seeds are provided, distances are `float('inf')`.
    
    Raises:
        ValueError: If the dimensionality of `query_points` and `seeds` does not match.
    """
    was_single_query = False
    if query_points.ndim == 1:
        query_points = query_points.unsqueeze(0) 
        was_single_query = True 

    if seeds.ndim == 1: 
         seeds = seeds.unsqueeze(0)

    if query_points.shape[1] != seeds.shape[1]:
        raise ValueError(f"Query points dimension ({query_points.shape[1]}) "
                         f"must match seeds dimension ({seeds.shape[1]}).")
    
    num_queries = query_points.shape[0]
    num_seeds = seeds.shape[0]

    if num_seeds == 0: 
        return (torch.full((num_queries,), -1, dtype=torch.long, device=query_points.device),
                torch.full((num_queries,), float('inf'), dtype=query_points.dtype, device=query_points.device))
    
    if num_queries == 0: 
        return (torch.empty((0,), dtype=torch.long, device=query_points.device),
                torch.empty((0,), dtype=query_points.dtype, device=query_points.device))

    q_expanded = query_points.unsqueeze(1)
    s_expanded = seeds.unsqueeze(0)
    squared_distances = torch.sum((q_expanded - s_expanded)**2, dim=2)
    min_sq_distances, nearest_seed_indices = torch.min(squared_distances, dim=1)

    if return_squared_distance:
        return nearest_seed_indices, min_sq_distances
    else:
        # Ensure non-negative before sqrt for stability, though squared_distances should be >= 0
        return nearest_seed_indices, torch.sqrt(torch.clamp(min_sq_distances, min=0.0))

# --- Unit Tests ---

class TestFindNearestSeed(unittest.TestCase):
    def test_find_nearest_2d_single_query(self):
        query = torch.tensor([0.0, 0.0], dtype=torch.float32)
        seeds_tensor = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [0.0, 0.5]], dtype=torch.float32)
        # Expected: seed 2 (0.0, 0.5) is nearest. Sq_Dist = 0.25, Dist = 0.5
        
        # Test default (Euclidean distance)
        indices_euc_default, dists_euc_default = find_nearest_seed(query, seeds_tensor)
        self.assertEqual(indices_euc_default.item(), 2)
        self.assertAlmostEqual(dists_euc_default.item(), 0.5, places=6)

        # Test explicit Euclidean distance
        indices_euc, dists_euc = find_nearest_seed(query, seeds_tensor, return_squared_distance=False)
        self.assertEqual(indices_euc.item(), 2)
        self.assertAlmostEqual(dists_euc.item(), 0.5, places=6)

        # Test explicit squared distance
        indices_sq, sq_dists = find_nearest_seed(query, seeds_tensor, return_squared_distance=True)
        self.assertEqual(indices_sq.item(), 2)
        self.assertAlmostEqual(sq_dists.item(), 0.25, places=6)


    def test_find_nearest_2d_multiple_queries(self):
        queries = torch.tensor([[0.0, 0.0], [1.1, 1.1]], dtype=torch.float32)
        seeds_tensor = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [0.0, 0.5]], dtype=torch.float32)
        # Query 1 (0,0): nearest is seed 2 (0,0.5), sq_dist 0.25, dist 0.5
        # Query 2 (1.1,1.1): nearest is seed 0 (1,1), sq_dist 0.02, dist sqrt(0.02) approx 0.141421
        
        expected_indices = torch.tensor([2, 0], dtype=torch.long)
        
        # Test default (Euclidean distance)
        indices_default, dists_default = find_nearest_seed(queries, seeds_tensor)
        expected_dists_euc = torch.tensor([0.5, math.sqrt(0.02)], dtype=torch.float32)
        self.assertTrue(torch.equal(indices_default, expected_indices))
        self.assertTrue(torch.allclose(dists_default, expected_dists_euc, atol=1e-6))

        # Test explicit squared distance
        indices_sq, sq_dists = find_nearest_seed(queries, seeds_tensor, return_squared_distance=True)
        expected_sq_dists = torch.tensor([0.25, 0.02], dtype=torch.float32)
        self.assertTrue(torch.equal(indices_sq, expected_indices))
        self.assertTrue(torch.allclose(sq_dists, expected_sq_dists, atol=1e-6))


    def test_find_nearest_3d(self):
        query = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        seeds_tensor = torch.tensor([[1.,1.,1.], [-1.,-1.,-1.], [0.,0.,0.5]], dtype=torch.float32)
        # Expected: seed 2 (0,0,0.5) is nearest. Sq_Dist = 0.25, Dist = 0.5
        
        # Test default (Euclidean distance)
        indices, dists = find_nearest_seed(query, seeds_tensor)
        self.assertEqual(indices.item(), 2)
        self.assertAlmostEqual(dists.item(), 0.5, places=6)

        # Test explicit squared distance
        indices_sq, sq_dists = find_nearest_seed(query, seeds_tensor, return_squared_distance=True)
        self.assertEqual(indices_sq.item(), 2)
        self.assertAlmostEqual(sq_dists.item(), 0.25, places=6)


    def test_find_nearest_exact_match(self):
        query = torch.tensor([1.0, 1.0], dtype=torch.float32)
        seeds_tensor = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], dtype=torch.float32)
        # Expected: seed 0 (1,1) is nearest. Dist = 0
        
        # Test default (Euclidean distance)
        indices, dists = find_nearest_seed(query, seeds_tensor)
        self.assertEqual(indices.item(), 0)
        self.assertAlmostEqual(dists.item(), 0.0, places=6)

        # Test explicit squared distance
        indices_sq, sq_dists = find_nearest_seed(query, seeds_tensor, return_squared_distance=True)
        self.assertEqual(indices_sq.item(), 0)
        self.assertAlmostEqual(sq_dists.item(), 0.0, places=6)


    def test_find_nearest_single_seed(self):
        query = torch.tensor([0.0, 0.0], dtype=torch.float32)
        single_seed = torch.tensor([1.0, 2.0], dtype=torch.float32) 
        # Expected: seed 0. Sq_Dist = 1^2 + 2^2 = 5. Dist = sqrt(5)
        
        # Test default (Euclidean distance)
        indices, dists = find_nearest_seed(query, single_seed)
        self.assertEqual(indices.item(), 0)
        self.assertAlmostEqual(dists.item(), math.sqrt(5.0), places=6)

        # Test explicit squared distance
        indices_sq, sq_dists = find_nearest_seed(query, single_seed, return_squared_distance=True)
        self.assertEqual(indices_sq.item(), 0)
        self.assertAlmostEqual(sq_dists.item(), 5.0, places=6)


    def test_find_nearest_no_seeds(self):
        query = torch.tensor([0.0, 0.0], dtype=torch.float32)
        no_seeds = torch.empty((0,2), dtype=torch.float32)
        
        # Test default (Euclidean distance)
        indices, dists = find_nearest_seed(query, no_seeds)
        self.assertEqual(indices.item(), -1) 
        self.assertEqual(dists.item(), float('inf'))

        # Test explicit squared distance
        indices_sq, sq_dists = find_nearest_seed(query, no_seeds, return_squared_distance=True)
        self.assertEqual(indices_sq.item(), -1)
        self.assertEqual(sq_dists.item(), float('inf'))


    def test_find_nearest_no_query_points(self):
        queries = torch.empty((0,2), dtype=torch.float32)
        seeds = torch.tensor([[1.,1.]], dtype=torch.float32)
        indices, distances = find_nearest_seed(queries, seeds)
        self.assertEqual(indices.shape[0], 0)
        self.assertEqual(distances.shape[0], 0)


    def test_find_nearest_dim_mismatch(self):
        query_2d = torch.tensor([0.0, 0.0], dtype=torch.float32)
        seeds_3d = torch.tensor([[1.,1.,1.]], dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "Query points dimension .* must match seeds dimension"):
            find_nearest_seed(query_2d, seeds_3d)
            
# if __name__ == '__main__':
#    unittest.main(argv=['first-arg-is-ignored'], exit=False)
