import pytest
import torch
from ai4pdes.grid import Grid

# Test custom initialization
# def test_custom_2Dgrid():
#     grid = Grid(dx=0.2, dy=0.4, nx=32, ny=16, nz=1, halo=3, device='cpu')

#     assert grid.is2D == True
#     assert grid.lx == 2

def test_init_grid():
    grid = Grid()
    assert isinstance(grid, Grid)