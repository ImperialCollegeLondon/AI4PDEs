from ai4pdes.viscosity import Viscosity
from ai4pdes.time_stepping import PredictorCorrector
from ai4pdes.output import Output
from ai4pdes.feedback import Feedback
# from ai4pdes.multigrid import FcycleMultigrid
from ai4pdes.variables import PrognosticVariables, DiagnosticVariables
from ai4pdes.operators import get_weights_linear_2D
from ai4pdes.boundary_conditions import boundary_condition_2D_u, boundary_condition_2D_p, boundary_condition_2D_v, boundary_condition_2D_cw
from ai4pdes.models.simulation import Simulation

import math
import torch
import torch.nn as nn

class FlowPastBlock(nn.Module):
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

        super(FlowPastBlock, self).__init__()

        self.grid = grid
        self.block = Block(grid) if block is None else block
        self.viscosity = viscosity
        self.time_stepping = time_stepping
        # self.multigrid = FcycleMultigrid(grid) if multigrid is None else multigrid
        self.output = output
        self.feedback = feedback

        # Temporary
        self.niteration = niteration
        self.nlevel = int(math.log(self.grid.ny, 2)) + 1

        # Define operators

        self.xadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.yadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)

        # Define operators (old version)
        # self.xadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # self.yadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)


        self.A    = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.Ahalo= nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.res  = nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0)  
        self.prol = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),)
        
        # Set weights
        [w1, w2, w3, wA, w_res, diag] = get_weights_linear_2D(grid.dx)   # Currently weights defined in operators.py
        self.xadv.weight.data = w2
        self.yadv.weight.data = w3
        self.diff.weight.data = w1
        self.A.weight.data = wA
        self.Ahalo.weight.data = wA
        self.res.weight.data = w_res
        self.diag = diag

        # Set bias
        bias_initializer = torch.tensor([0.0])     # Initial bias is always 0 for NNs 
        self.xadv.bias.data = bias_initializer
        self.yadv.bias.data = bias_initializer
        self.diff.bias.data = bias_initializer
        self.A.bias.data = bias_initializer
        self.res.bias.data = bias_initializer
    
    def initialize(self):
        # initialize model components, e.g.
        # block.initialize()
        
        # allocate all variables
        prognostic_variables = PrognosticVariables(self.grid)
        diagnostic_variables = DiagnosticVariables(self.grid)

        # gather into a simulation object
        simulation = Simulation(prognostic_variables, diagnostic_variables, self)
        return simulation  
    
    def F_cycle_MG(self, u, v, p, dt, iteration, diag, nlevel):
        b = -(self.xadv(u) + self.yadv(v)) / dt
        for MG in range(iteration):
            a = torch.zeros((1,1,1,1), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            r = self.A(boundary_condition_2D_p(p)) - b  
            r_s = []
            r_s.append(r)
            for i in range(1 ,nlevel):
                r = self.res(r)
                r_s.append(r)
            for i in reversed(range(1, nlevel)):
                aa = boundary_condition_2D_cw(a)
                if i == nlevel-1:
                    a = r_s[i] / diag
                else:
                    a = a - self.A(aa) / diag + r_s[i] / diag
                
                a = self.prol(a)
            p = p - a
            p = p - self.A(boundary_condition_2D_p(p)) / diag + b / diag
        return p
    
    def solid_body(self, u, v, sigma, dt):
        u = u / (1 + dt * sigma) 
        v = v / (1 + dt * sigma) 
        return u,v  #None

    def forward(self,
        prognostic_variables,
        diagnostic_variables,
        ):

        # do not execute forward step if simulation should stop (NaNs detected, not converging...)
        if self.feedback.stop_simulation:
            return None

        dt = self.time_stepping.dt
        # u = prognostic_variables.u  # unpack u, v, p
        # v = prognostic_variables.v
        # p = diagnostic_variables.p

        # print('before making a padding', diagnostic_variables.p.shape)

        prognostic_variables.uu = boundary_condition_2D_u(prognostic_variables.u, self.grid.ub) 
        prognostic_variables.vv = boundary_condition_2D_v(prognostic_variables.v, self.grid.ub)  
        diagnostic_variables.pp = boundary_condition_2D_p(diagnostic_variables.p)   

        # print('after making a padding', diagnostic_variables.p.shape)

        Grapx_p  = self.xadv(diagnostic_variables.pp) * dt 
        Grapy_p = self.yadv(diagnostic_variables.pp) * dt 

        ADx_u = self.xadv(prognostic_variables.uu) 
        ADy_u = self.yadv(prognostic_variables.uu) 
        ADx_v = self.xadv(prognostic_variables.vv)
        ADy_v = self.yadv(prognostic_variables.vv) 
        AD2_u = self.diff(prognostic_variables.uu)
        AD2_v = self.diff(prognostic_variables.vv) 
        # print(ADx_u[ADx_u!=0])
        # print(ADy_u[ADx_u!=0])
        # print(AD2_u[ADx_u!=0])

        # print(u[u!=0])
        # print(v[v!=0])
        # First step for solving uvw
        # bu = diagnostic_variables.bu  # unpack bu, bv
        # bv = diagnostic_variables.bv

        # bu = u + 0.5 * (self.viscosity.nu * AD2_u * dt - u * ADx_u * dt - v * ADy_u * dt) - Grapx_p 
        # bv = v + 0.5 * (self.viscosity.nu * AD2_v * dt - u * ADx_v * dt - v * ADy_v * dt) - Grapy_p 

        diagnostic_variables.bu = prognostic_variables.u + 0.5 * (self.viscosity.nu * AD2_u * dt - prognostic_variables.u * ADx_u * dt - prognostic_variables.v * ADy_u * dt) - Grapx_p 
        diagnostic_variables.bv = prognostic_variables.v + 0.5 * (self.viscosity.nu * AD2_v * dt - prognostic_variables.u * ADx_v * dt - prognostic_variables.v * ADy_v * dt) - Grapy_p 
        # print(bu[bu!=0])
        # print(bv[bv!=0])        
        # self.solid_body(diagnostic_variables.bu, diagnostic_variables.bv, self.block.sigma, dt)
        [diagnostic_variables.bu, diagnostic_variables.bv] = self.solid_body(diagnostic_variables.bu, diagnostic_variables.bv, self.block.sigma, dt)

        # Padding velocity vectors 
        diagnostic_variables.buu = boundary_condition_2D_u(diagnostic_variables.bu, self.grid.ub) 
        diagnostic_variables.bvv = boundary_condition_2D_v(diagnostic_variables.bv, self.grid.ub) 

        # ADx_u = self.xadv(bu) ; ADy_u = self.yadv(bu) 
        # ADx_v = self.xadv(bv) ; ADy_v = self.yadv(bv) 
        # AD2_u = self.diff(bu) ; AD2_v = self.diff(bv) 

        ADx_u = self.xadv(diagnostic_variables.buu) ; ADy_u = self.yadv(diagnostic_variables.buu) 
        ADx_v = self.xadv(diagnostic_variables.bvv) ; ADy_v = self.yadv(diagnostic_variables.bvv) 
        AD2_u = self.diff(diagnostic_variables.buu) ; AD2_v = self.diff(diagnostic_variables.bvv) 

        # Second step for solving uvw   
        prognostic_variables.u = prognostic_variables.u + self.viscosity.nu * AD2_u * dt - diagnostic_variables.bu * ADx_u * dt - diagnostic_variables.bv * ADy_u * dt - Grapx_p 
        prognostic_variables.v = prognostic_variables.v + self.viscosity.nu * AD2_v * dt - diagnostic_variables.bu * ADx_v * dt - diagnostic_variables.bv * ADy_v * dt - Grapy_p 
        # self.solid_body(prognostic_variables.u, prognostic_variables.v, self.block.sigma, dt)
        [prognostic_variables.u, prognostic_variables.v] = self.solid_body(prognostic_variables.u, prognostic_variables.v, self.block.sigma, dt)

        # pressure
        prognostic_variables.uu = boundary_condition_2D_u(prognostic_variables.u, self.grid.ub)
        prognostic_variables.vv = boundary_condition_2D_v(prognostic_variables.v, self.grid.ub)
        # a = diagnostic_variables.a
        diagnostic_variables.p = self.F_cycle_MG(prognostic_variables.uu, prognostic_variables.vv, diagnostic_variables.p, dt, self.niteration, self.diag, self.nlevel)

        # Pressure gradient correction    
        diagnostic_variables.pp = boundary_condition_2D_p(diagnostic_variables.p)   
        prognostic_variables.u = prognostic_variables.u - self.xadv(diagnostic_variables.pp) * dt
        prognostic_variables.v = prognostic_variables.v - self.yadv(diagnostic_variables.pp) * dt 
        # self.solid_body(prognostic_variables.u, prognostic_variables.v, self.block.sigma, dt)
        [prognostic_variables.u, prognostic_variables.v] = self.solid_body(prognostic_variables.u, prognostic_variables.v, self.block.sigma, dt)
        return None
    
class Block:
    def __init__(
        self,
        grid,
        timescale = 1e-8,   # in seconds
        cor_x = None,
        cor_y = None,
        size_x = None,
        size_y = None,
    ):    
        self.cor_x = int(grid.nx/4) if cor_x is None else cor_x
        self.cor_y = int(grid.ny/2) if cor_y is None else cor_y
        self.size_x = int(grid.nx/32) if size_x is None else size_x
        self.size_y = int(grid.ny/8) if size_y is None else size_y
        self.timescale = timescale
        self.sigma = self.create_solid_body_2D(grid)

    def create_solid_body_2D(self, grid):
        shape = (1, 1, grid.ny+2*grid.halo, grid.nx+2*grid.halo)
        sigma = torch.zeros(shape, device=grid.device)

        cor_x = self.cor_x      # unpack
        cor_y = self.cor_y
        size_x = self.size_x
        size_y = self.size_y

        sigma[0, 0 ,cor_y-size_y:cor_y+size_y, cor_x-size_x:cor_x+size_x] = 1/self.timescale
        print('A bluff body has been created successfully!')
        print('===========================================')
        print('Size of body in x:',size_x*2)
        print('Size of body in y:',size_y*2)
        print('position of body in x:',cor_x)
        print('position of body in y:',cor_y)
        print('===========================================')
        return sigma
