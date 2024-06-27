from ai4pdes.viscosity import Viscosity
from ai4pdes.time_stepping import PredictorCorrector
from ai4pdes.output import Output
from ai4pdes.feedback import Feedback
from ai4pdes.multigrid import FcycleMultigrid
from ai4pdes.variables import PrognosticVariables, DiagnosticVariables
from ai4pdes.operators import get_weights_linear_2D
from ai4pdes.boundary_conditions import boundary_condition_2D_u, boundary_condition_2D_p, boundary_condition_2D_v, boundary_condition_2D_cw
from ai4pdes.models.simulation import Simulation

import math
import torch
import torch.nn as nn


class FlowPastBlock:
    def __init__(self,
        grid,
        block = None,
        viscosity = Viscosity(),
        time_stepping = PredictorCorrector(),
        multigrid = None,
        output = Output(),
        feedback = Feedback(),
        niteration = 5,

    ):
        self.grid = grid
        self.block = Block(grid) if block is None else block
        self.viscosity = viscosity
        self.time_stepping = time_stepping
        self.multigrid = FcycleMultigrid(grid) if multigrid is None else multigrid
        self.output = output
        self.feedback = feedback

        # Temporary
        self.niteration = niteration
        self.nlevel = int(math.log(self.grid.ny, 2)) + 1

        # Define operators
        self.xadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.yadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.A = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.res = nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0)  
        self.prol = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),)
        
        # Set weights
        [w1, w2, w3, wA, w_res, diag] = get_weights_linear_2D(grid.dx)   # Currently weights defined in operators.py
        self.xadv.weight.data = w2
        self.yadv.weight.data = w3
        self.diff.weight.data = w1
        self.A.weight.data = wA
        self.res.weight.data = w_res
        self.diag = diag

        # Set bias
        bias_initializer = torch.tensor([0.0])     # Initial bias is always 0 for NNs 
        self.xadv.bias.data = bias_initializer
        self.yadv.bias.data = bias_initializer
        self.diff.bias.data = bias_initializer
        self.A.bias.data = bias_initializer
        self.res.bias.data = bias_initializer
    
    def solid_body(self, u, v, sigma, dt):
        u = u / (1 + dt * sigma) 
        v = v / (1 + dt * sigma) 
        return u, v   

    def forward(self,
        prognostic_variables,
        diagnostic_variables,
        ):

        # do not execute forward step if simulation should stop (NaNs detected, not converging...)
        if self.feedback.stop_simulation:
            return None

        dt = self.time_stepping.dt
        u = prognostic_variables.u  # unpack u, v, p
        v = prognostic_variables.v
        p = prognostic_variables.p

        boundary_condition_2D_u(u, self.grid.ub) 
        boundary_condition_2D_v(v, self.grid.ub)  
        boundary_condition_2D_p(p)   
        
        Grapx_p  = self.xadv(p) * dt 
        Grapy_p = self.yadv(p) * dt 

        ADx_u = self.xadv(u) 
        ADy_u = self.yadv(u) 
        ADx_v = self.xadv(v)
        ADy_v = self.yadv(v) 
        AD2_u = self.diff(u)
        AD2_v = self.diff(v) 

        # First step for solving uvw
        b_u = diagnostic_variables.b_u  # unpack b_u, b_v
        b_v = diagnostic_variables.b_v

        b_u = u + 0.5 * (self.viscosity.nu * AD2_u * dt - u * ADx_u * dt - v * ADy_u * dt) - Grapx_p 
        b_v = v + 0.5 * (self.viscosity.nu * AD2_v * dt - u * ADx_v * dt - v * ADy_v * dt) - Grapy_p 
        [b_u, b_v] = self.solid_body(b_u, b_v, self.block.sigma, dt)

        # Padding velocity vectors 
        boundary_condition_2D_u(b_u, self.grid.ub) 
        boundary_condition_2D_v(b_v, self.grid.ub) 

        ADx_u = self.xadv(b_u) ; ADy_u = self.yadv(b_u) 
        ADx_v = self.xadv(b_v) ; ADy_v = self.yadv(b_v) 
        AD2_u = self.diff(b_u) ; AD2_v = self.diff(b_v) 

        # Second step for solving uvw   
        u = u + self.viscosity.nu * AD2_u * dt - b_u * ADx_u * dt - b_v * ADy_u * dt - Grapx_p 
        v = v + self.viscosity.nu * AD2_v * dt - b_u * ADx_v * dt - b_v * ADy_v * dt - Grapy_p 
        [u, v] = self.solid_body(u, v, self.block.sigma, dt)

        # pressure
        boundary_condition_2D_u(u, self.grid.ub) 
        boundary_condition_2D_v(v, self.grid.ub)  
        [p, w, r] = self.F_cycle_MG(u, v, p, self.niteration, self.diag, dt, self.nlevel)

        # Pressure gradient correction    
        boundary_condition_2D_p(p)  
        u = u - self.xadv(p) * dt
        v = v - self.yadv(p) * dt 
        [u, v] = self.solid_body(u, v, self.block.sigma, dt)
        return u, v, p, w, r
    
    def initialize(self):
        # initialize model components, e.g.
        # block.initialize()
        
        # allocate all variables
        prognostic_variables = PrognosticVariables(self.grid)
        diagnostic_variables = DiagnosticVariables(self.grid)

        # gather into a simulation object
        simulation = Simulation(prognostic_variables, diagnostic_variables, self)
        return simulation
    
    def F_cycle_MG(self, u, v, p, iteration, diag, dt, nlevel):
        b = -(self.xadv(u) + self.yadv(v)) / dt
        for MG in range(iteration):
            w = torch.zeros((1,1,1,1), device=self.grid.device)
            r = self.A(boundary_condition_2D_p(p)) - b 
            r_s = []
            r_s.append(r)
            for i in range(1,nlevel):
                r = self.res(r)
                r_s.append(r)
            for i in reversed(range(1,nlevel)):
                ww = boundary_condition_2D_cw(w)
                w = w - self.A(ww) / diag + r_s[i] / diag
                w = self.prol(w)         
            p = p - w
            p = p - self.A(boundary_condition_2D_p(p)) / diag + b / diag
        return p, w, r
    
class Block:
    def __init__(
        self,
        grid,
        cor_x = None,
        cor_y = None,
        size_x = None,
        size_y = None,
    ):    
        self.cor_x = int(grid.nx/4) if cor_x is None else cor_x
        self.cor_y = int(grid.ny/4) if cor_y is None else cor_y
        self.size_x = int(grid.nx/4) if size_x is None else size_x
        self.size_y = int(grid.ny/4) if size_y is None else size_y
        self.sigma = self.create_solid_body_2D(grid, self.cor_x, self.cor_y, self.size_x, self.size_y)

    def create_solid_body_2D(self, grid, cor_x, cor_y, size_x, size_y):
        input_shape = (1, 1, grid.ny, grid.nx)
        sigma = torch.zeros(input_shape, device=grid.device)
        sigma[0,0,cor_y-size_y:cor_y+size_y,cor_x-size_x:cor_x+size_x] = 1e08
        print('A bluff body has been created successfully!')
        print('===========================================')
        print('Size of body in x:',size_x*2)
        print('Size of body in y:',size_y*2)
        print('position of body in x:',cor_x)
        print('position of body in y:',cor_y)
        print('===========================================')
        return sigma