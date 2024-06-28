import numpy as np

class Feedback:
    def __init__(self,
        ncheck = 100,               # Time step to check residual
        residual_max = 80000.0,     # maximum residual to abort simulation
    ):
        self.ncheck = ncheck
        self.residual_max = residual_max
        self.nan_detected = False
        self.stop_simulation = False

    def check(self,
        itime,
        prognostic_variables,
        diagnostic_variables,
    ):
        if itime % self.ncheck == 0:
            a = diagnostic_variables.a
            residual = np.max(np.abs(a.cpu().detach().numpy()))
            # print('Time step:', itime, 'Pressure residual:',"{:.5f}".format(residual))

            if residual > self.residual_max:
                self.stop_simulation = True
                print('Not converged !!!!!!')