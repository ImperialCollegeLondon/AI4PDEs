import torch

# include halo
def variable_shape(grid):
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
        self.u = torch.zeros(shape, device=grid.device)
        self.v = torch.zeros(shape, device=grid.device)
        self.w = torch.zeros(shape, device=grid.device) if grid.is3D() else None

class DiagnosticVariables:
    def __init__(self,
        grid,
    ):
        shape = variable_shape(grid)
        self.p = torch.zeros(shape, device=grid.device)
        self.bu = torch.zeros(shape, device=grid.device)
        self.bv = torch.zeros(shape, device=grid.device)
        self.bw = torch.zeros(shape, device=grid.device) if grid.is3D() else None

        self.w = 

