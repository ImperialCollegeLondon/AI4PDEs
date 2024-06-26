class FlowPastBlock:
    def __init__(self,
        grid,
        block = None,
        viscosity = Viscosity(),
        time_stepping = PredictorCorrector(),
        multigrid = None,
        output = Output(),
        feedback = Feedback(),
    ):
        self.block = Block(grid) if block is None else block
        self.viscosity = viscosity
        self.time_stepping = time_stepping
        self.multigrid = FcycleMultigrid(grid) if multigrid is None else multigrid
        self.output = output
        self.feedback = feedback

    def forward(self, values_u, values_uu, values_v, values_vv, values_p, values_pp, sigma, b_uu, b_vv, dt, iteration):      
        values_uu = boundary_condition_2D_u(values_u,values_uu,ub) 
        values_vv = boundary_condition_2D_v(values_v,values_vv,ub)  
        values_pp = boundary_condition_2D_p(values_p,values_pp)   
        Grapx_p  = self.xadv(values_pp) * dt  ; Grapy_p = self.yadv(values_pp) * dt 
        ADx_u = self.xadv(values_uu) ; ADy_u = self.yadv(values_uu) 
        ADx_v = self.xadv(values_vv) ; ADy_v = self.yadv(values_vv) 
        AD2_u = self.diff(values_uu) ; AD2_v = self.diff(values_vv) 
    # First step for solving uvw
        b_u = values_u + 0.5 * (nu * AD2_u * dt - values_u * ADx_u * dt - values_v * ADy_u * dt) - Grapx_p 
        b_v = values_v + 0.5 * (nu * AD2_v * dt - values_u * ADx_v * dt - values_v * ADy_v * dt) - Grapy_p 
        [b_u, b_v] = self.solid_body(b_u, b_v, sigma, dt)
    # Padding velocity vectors 
        b_uu = boundary_condition_2D_u(b_u,b_uu,ub) 
        b_vv = boundary_condition_2D_v(b_v,b_vv,ub) 
        ADx_u = self.xadv(b_uu) ; ADy_u = self.yadv(b_uu) 
        ADx_v = self.xadv(b_vv) ; ADy_v = self.yadv(b_vv) 
        AD2_u = self.diff(b_uu) ; AD2_v = self.diff(b_vv) 
    # Second step for solving uvw   
        values_u = values_u + nu * AD2_u * dt - b_u * ADx_u * dt - b_v * ADy_u * dt - Grapx_p 
        values_v = values_v + nu * AD2_v * dt - b_u * ADx_v * dt - b_v * ADy_v * dt - Grapy_p 
        [values_u, values_v] = self.solid_body(values_u, values_v, sigma, dt)
    # pressure
        values_uu = boundary_condition_2D_u(values_u,values_uu,ub) 
        values_vv = boundary_condition_2D_v(values_v,values_vv,ub)  
        [values_p, w ,r] = self.F_cycle_MG(values_uu, values_vv, values_p, values_pp, iteration, diag, dt, nlevel)
    # Pressure gradient correction    
        values_pp = boundary_condition_2D_p(values_p, values_pp)  
        values_u = values_u - self.xadv(values_pp) * dt ; values_v = values_v - self.yadv(values_pp) * dt 
        [values_u, values_v] = self.solid_body(values_u, values_v, sigma, dt)
        return values_u, values_v, values_p, w, r
    
    def initialize():

        # initialize model components, e.g.
        # block.initialize()
        
        # allocate all variables
        prognostic_variables = PrognosticVariables(grid)
        diagnostic_variables = DiagnosticVariables(grid)

        # gather into a simulation object
        simulation = Simulation(prognostic_variables, diagnostic_variables, model)
        return simulation
    
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

    def create_solid_body_2D(grid, cor_x, cor_y, size_x, size_y):
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