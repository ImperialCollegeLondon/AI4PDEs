import torch

def variable_shape(grid):
    if grid.is2D():
        shape = (1, 1, grid.ny , grid.nx )
    else:
        shape = (1, 1, grid.nz , grid.ny , grid.nx )
    return shape

def variable_shape_with_halo(grid):
    if grid.is2D():
        shape = (1, 1, grid.ny + 2*grid.halo, grid.nx + 2*grid.halo)
    else:
        shape = (1, 1, grid.nz + 2*grid.halo, grid.ny + 2*grid.halo, grid.nx + 2*grid.halo)
    return shape

class PrognosticVariables:
    def __init__(self,
        grid,
    ):
        shape = variable_shape(grid)
        shape_with_halo = variable_shape_with_halo(grid)
        self.u = torch.zeros(shape, device=grid.device)
        self.v = torch.zeros(shape, device=grid.device)
        self.w = torch.zeros(shape, device=grid.device) if grid.is3D() else None

        self.uu = torch.zeros(shape_with_halo, device=grid.device)
        self.vv = torch.zeros(shape_with_halo, device=grid.device)
        self.ww = torch.zeros(shape_with_halo, device=grid.device) if grid.is3D() else None

class DiagnosticVariables:
    def __init__(self,
        grid,
    ):
        shape = variable_shape(grid)
        shape_with_halo = variable_shape_with_halo(grid)
        self.p = torch.zeros(shape, device=grid.device)
        self.bu = torch.zeros(shape, device=grid.device)
        self.bv = torch.zeros(shape, device=grid.device)
        self.bw = torch.zeros(shape, device=grid.device) if grid.is3D() else None

        self.pp = torch.zeros(shape_with_halo, device=grid.device)

