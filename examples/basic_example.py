import ai4pdes
from ai4pdes.models import FlowPastBlock, Block

import matplotlib.pyplot as plt

# Create dims same as notebook
grid = ai4pdes.grid.Grid(nx=1024, ny=256)
prognostic_variables = ai4pdes.variables.PrognosticVariables(grid)
diagnostic_variables = ai4pdes.variables.DiagnosticVariables(grid)
# Create same block as notebook
block = Block(grid=grid, cor_x=145, cor_y=int(grid.ny/2), size_x=25, size_y=25)
model = FlowPastBlock(grid, block=block)
for i in range(100):
    model.forward(prognostic_variables=prognostic_variables, 
              diagnostic_variables=diagnostic_variables, 
              dt=0.05)

plt.figure(figsize=(15, 6))
plt.imshow(-prognostic_variables.u[0,0,:,:].detach().numpy())
plt.colorbar()
plt.title('u component velocity (m/s)')
plt.show()
# plt.axis('off')