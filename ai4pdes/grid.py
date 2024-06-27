import torch 
class Grid:
    def __init__(self,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        nx=254,
        ny=62,
        nz=1,
        halo=1,
        ub=-1.0,
        device=None,
        ):
    
        # grid spacing
        self.dx = dx
        self.dy = dy
        self.dz = dz

        # number of grid points
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.halo = halo

        # grid domain
        self.lx = dx*nx
        self.ly = dy*ny
        self.lz = dz*nz

        # boundary condition
        self.ub = ub
    
        # computing device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def is2D(self):
        return self.nz == 1
    
    def is3D(self):
        return self.nz > 1