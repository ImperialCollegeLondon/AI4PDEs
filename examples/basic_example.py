import ai4pdes
from ai4pdes.models import FlowPastBlock

import matplotlib.pyplot as plt

grid = ai4pdes.grid.Grid(nx=1024, ny=256)
prognostic_variables = ai4pdes.variables.PrognosticVariables(grid)
diagnostic_variables = ai4pdes.variables.DiagnosticVariables(grid)
model = FlowPastBlock(grid)
model.forward(prognostic_variables=prognostic_variables, 
              diagnostic_variables=diagnostic_variables, 
              dt=0.05)

plt.figure(figsize=(15, 6))
plt.imshow(-prognostic_variables.u[0,0,:,:].detach().numpy())
plt.colorbar()
plt.title('u component velocity (m/s)')
plt.show()
# plt.axis('off')