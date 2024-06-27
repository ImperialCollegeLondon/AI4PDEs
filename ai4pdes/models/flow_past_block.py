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
    
    def solid_body(self, values_u, values_v, sigma, dt):
        values_u = values_u / (1 + dt * sigma) 
        values_v = values_v / (1 + dt * sigma) 
        return values_u, values_v   

    def forward(self,
                prognostic_variables,
                diagnostic_variables,
                dt):
        #values_u, values_uu, values_v, values_vv, values_p, values_pp, sigma, b_uu, b_vv, dt, iteration):      
        # values_u --> prognostic_variables.u
        values_uu = boundary_condition_2D_u(prognostic_variables.u,prognostic_variables.uu, self.grid.ub) 
        values_vv = boundary_condition_2D_v(prognostic_variables.v,prognostic_variables.vv, self.grid.ub)  
        values_pp = boundary_condition_2D_p(diagnostic_variables.p,diagnostic_variables.pp)   
        
        Grapx_p  = self.xadv(diagnostic_variables.pp) * dt 
        Grapy_p = self.yadv(diagnostic_variables.pp) * dt 

        ADx_u = self.xadv(prognostic_variables.uu) 
        ADy_u = self.yadv(prognostic_variables.uu) 
        ADx_v = self.xadv(prognostic_variables.vv)
        ADy_v = self.yadv(prognostic_variables.vv) 
        AD2_u = self.diff(prognostic_variables.uu)
        AD2_v = self.diff(prognostic_variables.vv) 

        # First step for solving uvw
        diagnostic_variables.b_u = prognostic_variables.u + 0.5 * (self.viscosity.nu * AD2_u * dt - prognostic_variables.u * ADx_u * dt - prognostic_variables.v * ADy_u * dt) - Grapx_p 
        diagnostic_variables.b_v = prognostic_variables.v + 0.5 * (self.viscosity.nu * AD2_v * dt - prognostic_variables.u * ADx_v * dt - prognostic_variables.v * ADy_v * dt) - Grapy_p 
        [diagnostic_variables.b_u, diagnostic_variables.b_v] = self.solid_body(diagnostic_variables.b_u, diagnostic_variables.b_v, self.block.sigma, dt)

        # Padding velocity vectors 
        diagnostic_variables.b_uu = boundary_condition_2D_u(diagnostic_variables.b_u,diagnostic_variables.b_uu, self.grid.ub) 
        diagnostic_variables.b_vv = boundary_condition_2D_v(diagnostic_variables.b_v,diagnostic_variables.b_vv, self.grid.ub) 

        ADx_u = self.xadv(diagnostic_variables.b_uu) ; ADy_u = self.yadv(diagnostic_variables.b_uu) 
        ADx_v = self.xadv(diagnostic_variables.b_vv) ; ADy_v = self.yadv(diagnostic_variables.b_vv) 
        AD2_u = self.diff(diagnostic_variables.b_uu) ; AD2_v = self.diff(diagnostic_variables.b_vv) 

        # Second step for solving uvw   
        prognostic_variables.u = prognostic_variables.u + self.viscosity.nu * AD2_u * dt - diagnostic_variables.b_u * ADx_u * dt - diagnostic_variables.b_v * ADy_u * dt - Grapx_p 
        prognostic_variables.v = prognostic_variables.v + self.viscosity.nu * AD2_v * dt - diagnostic_variables.b_u * ADx_v * dt - diagnostic_variables.b_v * ADy_v * dt - Grapy_p 
        [prognostic_variables.u, prognostic_variables.v] = self.solid_body(prognostic_variables.u, prognostic_variables.v, self.block.sigma, dt)

        # pressure
        prognostic_variables.uu = boundary_condition_2D_u(prognostic_variables.u,prognostic_variables.uu, self.grid.ub) 
        prognostic_variables.vv = boundary_condition_2D_v(prognostic_variables.v,prognostic_variables.vv, self.grid.ub)  
        [diagnostic_variables.p, w ,r] = self.F_cycle_MG(prognostic_variables.uu, prognostic_variables.vv, diagnostic_variables.p, diagnostic_variables.pp, self.niteration, self.diag, dt, self.nlevel)
        # Pressure gradient correction    
        diagnostic_variables.pp = boundary_condition_2D_p(diagnostic_variables.p, diagnostic_variables.pp )  
        prognostic_variables.u = prognostic_variables.u - self.xadv(diagnostic_variables.pp) * dt
        prognostic_variables.v = prognostic_variables.v - self.yadv(diagnostic_variables.pp) * dt 
        [prognostic_variables.u, prognostic_variables.v] = self.solid_body(prognostic_variables.u, prognostic_variables.v, self.block.sigma, dt)
        return prognostic_variables.u, prognostic_variables.v, diagnostic_variables.p, w, r
    
    def initialize(self):
        # initialize model components, e.g.
        # block.initialize()
        
        # allocate all variables
        prognostic_variables = PrognosticVariables(self.grid)
        diagnostic_variables = DiagnosticVariables(self.grid)

        # gather into a simulation object
        simulation = Simulation(prognostic_variables, diagnostic_variables, self)
        return simulation
    
    def F_cycle_MG(self, values_uu, values_vv, values_p, values_pp, iteration, diag, dt, nlevel):
        b = -(self.xadv(values_uu) + self.yadv(values_vv)) / dt
        for MG in range(iteration):
            w = torch.zeros((1,1,1,1), device=self.grid.device)
            r = self.A(boundary_condition_2D_p(values_p, values_pp)) - b 
            r_s = []  
            r_s.append(r)
            for i in range(1,nlevel):
                r = self.res(r)
                r_s.append(r)
            for i in reversed(range(1,nlevel)):
                ww = boundary_condition_2D_cw(w)
                w = w - self.A(ww) / diag + r_s[i] / diag
                w = self.prol(w)         
            values_p = values_p - w
            values_p = values_p - self.A(boundary_condition_2D_p(values_p, values_pp)) / diag + b / diag
        return values_p, w, r
    
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