class FcycleMultigrid:
    def __init__(self,
        grid,
        niteration = 5,
    ):
        self.nlevel = int(math.log(grid.ny, 2)) + 1
        self.niteration = niteration

    def cycle(self, values_uu, values_vv, values_p, values_pp, iteration, diag, dt, nlevel):
        b = -(self.xadv(values_uu) + self.yadv(values_vv)) / dt
        for MG in range(iteration):
            w = torch.zeros((1,1,1,1), device=device)
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