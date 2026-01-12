# Tile coding feature representation
import numpy as np

class TileCoder:
    def __init__(self, num_tilings=8, tiles_per_dim=8, state_bounds=None):
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim
        
        if state_bounds is None:
            state_bounds = [(-2.4, 2.4), (-3.0, 3.0), 
                          (-0.21, 0.21), (-3.0, 3.0)]
        
        self.state_bounds = state_bounds
        self.num_dims = len(state_bounds)
        
        self.tile_widths = [(bounds[1] - bounds[0]) / tiles_per_dim 
                           for bounds in state_bounds]
        
        self.offsets = [[(i / num_tilings) * width 
                        for width in self.tile_widths] 
                       for i in range(num_tilings)]
        
        self.size = num_tilings * (tiles_per_dim ** self.num_dims)
    
    def get_tiles(self, state):
        tiles = []
        
        for tiling_idx in range(self.num_tilings):
            tile_coords = []
            
            for dim in range(self.num_dims):
                val = state[dim] - self.state_bounds[dim][0]
                val -= self.offsets[tiling_idx][dim]
                tile_coord = int(val / self.tile_widths[dim])
                tile_coord = max(0, min(tile_coord, self.tiles_per_dim - 1))
                tile_coords.append(tile_coord)
            
            tile_idx = tiling_idx * (self.tiles_per_dim ** self.num_dims)
            multiplier = 1
            for coord in reversed(tile_coords):
                tile_idx += coord * multiplier
                multiplier *= self.tiles_per_dim
            
            tiles.append(tile_idx)
        
        return tiles