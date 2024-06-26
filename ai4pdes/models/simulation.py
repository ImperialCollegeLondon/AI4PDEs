import time
import torch
from tqdm import tqdm as progressbar

class Simulation:
    def __init__(self,
        prognostic_variables,
        diagnostic_variables,
        model,
    ):
        self.prognostic_variables = prognostic_variables
        self.diagnostic_variables = diagnostic_variables
        self.model = model

    def run(self,
        ntimesteps = 100,
    ):
        with torch.no_grad():
            for itime in progressbar(range(1, ntimesteps+1)):
                self.model.forward(self.prognostic_variables, self.diagnostic_variables)
                self.model.feedback.check(itime, self.prognostic_variables, self.diagnostic_variables)
                self.model.output.store(itime, self.prognostic_variables)