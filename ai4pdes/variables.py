import torch

# include halo
def variable_shape(grid):
    if grid.is2D():
        shape     = (1, 1, grid.ny + 2*grid.halo, grid.nx + 2*grid.halo)
        shape_pad =  (1, 1, grid.ny + 2*grid.halo + 2, grid.nx + 2*grid.halo + 2)
    else:
        shape     = (1, 1, grid.nz + 2*grid.halo, grid.ny + 2*grid.halo, grid.nx + 2*grid.halo)
        shape_pad = (1, 1, grid.nz + 2*grid.halo + 2, grid.ny + 2*grid.halo + 2, grid.nx + 2*grid.halo + 2)
    return shape, shape_pad

class PrognosticVariables:
    def __init__(self,
        grid,
    ):
        [shape,shape_pad] = variable_shape(grid)
        self.u = torch.zeros(shape, device=grid.device)
        self.v = torch.zeros(shape, device=grid.device)
        self.w = torch.zeros(shape, device=grid.device) if grid.is3D() else None

        self.uu = torch.zeros(shape_pad, device=grid.device)
        self.vv = torch.zeros(shape_pad, device=grid.device)
        self.ww = torch.zeros(shape_pad, device=grid.device) if grid.is3D() else None

class DiagnosticVariables:
    def __init__(self,
        grid,
    ):
        [shape,shape_pad] = variable_shape(grid)
        self.p = torch.zeros(shape, device=grid.device)
        self.bu = torch.zeros(shape, device=grid.device)
        self.bv = torch.zeros(shape, device=grid.device)
        self.bw = torch.zeros(shape, device=grid.device) if grid.is3D() else None

        self.pp = torch.zeros(shape_pad, device=grid.device)
        self.buu = torch.zeros(shape_pad, device=grid.device)
        self.bvv = torch.zeros(shape_pad, device=grid.device)
        self.bww = torch.zeros(shape_pad, device=grid.device) if grid.is3D() else None
        # multigrid array
        self.a = torch.zeros((1,1,1,1), device=grid.device)

