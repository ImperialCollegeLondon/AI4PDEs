import ai4pdes
from ai4pdes.models import FlowPastBlock
grid = ai4pdes.grid.Grid()
prognostic_variables = ai4pdes.variables.PrognosticVariables(grid)
diagnostic_variables = ai4pdes.variables.DiagnosticVariables(grid)
model = FlowPastBlock(grid)
model.forward(prognostic_variables=prognostic_variables, 
              diagnostic_variables=diagnostic_variables, 
              dt=0.05)

