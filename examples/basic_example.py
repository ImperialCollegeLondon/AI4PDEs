import ai4pdes
from ai4pdes.models import FlowPastBlock
grid = ai4pdes.grid.Grid()
prognostic_variables = ai4pdes.variables.PrognosticVariables(grid)
diagnostic_variables = ai4pdes.variables.DiagnosticVariables(grid)
model = FlowPastBlock(grid)
