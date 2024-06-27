import ai4pdes
from ai4pdes.models import FlowPastBlock, Block
from ai4pdes.plot_state import plot_u, plot_v, plot_speed
import matplotlib.pyplot as plt

# Create dims same as notebook
grid = ai4pdes.grid.Grid(nx=1024, ny=256)
prognostic_variables = ai4pdes.variables.PrognosticVariables(grid)
diagnostic_variables = ai4pdes.variables.DiagnosticVariables(grid)
# Create same block as notebook
block = Block(grid=grid, cor_x=145, cor_y=int(grid.ny/2), size_x=25, size_y=25)
model = FlowPastBlock(grid, block=block)
for i in range(1):
    model.forward(prognostic_variables=prognostic_variables, 
              diagnostic_variables=diagnostic_variables, 
              dt=0.05)

plot_u(prognostic_variables)
plt.show()
plot_v(prognostic_variables)
plt.show()
plot_speed(prognostic_variables)
plt.show()