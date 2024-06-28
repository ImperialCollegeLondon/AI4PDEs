# AI4PDEs

[![Tests](https://github.com/ImperialCollegeLondon/AI4PDEs/actions/workflows/ci.yml/badge.svg)](https://github.com/ImperialCollegeLondon/AI4PDEs/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/AI4PDEs/badge/?version=latest)](https://ai4pdes.readthedocs.io/en/latest/?badge=latest)

AI4PDEs is a package that solves partial differential equations (PDEs) using functionality from PyTorch.
It currently solves the 2D Navier-Stokes equations, 2D advection-diffusion equations and 3D Navier-Stokes equations.

## Installation

1. **Clone the repo**
   ```sh
   git clone https://github.com/ImperialCollegeLondon/AI4PDEs.git
   ```
2. **Install**

   Currently requires python>=3.10.

   ```python -m pip install . ``` or 
   ```python -m pip install -e ."[dev]"```

3. **Run one of the example files**
   ```
   cd ai4pdes/structured/
   jupyter notebook flow_past_block.ipynb
   jupyter notebook advection_diffusion.ipynb
   jupyter notebook flow_past_buildings.ipynb
   ```

## Usage

Run your first model of flow past a block!

```python
import ai4pdes
from ai4pdes.models import FlowPastBlock, Block
from ai4pdes.grid import Grid

grid = Grid(nx=254, ny=62)
block = Block(grid)
model = FlowPastBlock(grid, block)
simulation = model.initialize()
simulation.run(ntimesteps=100)
import matplotlib.pyplot as plt
plt.imshow(-simulation.prognostic_variables.u.cpu()[0,0,:,:])
plt.colorbar()
```

## Contributing

Contributions are welcome! If you have a suggestion that would make this better, please fork the repo and create a pull request.
You can also simply open an issue.

## Citing

If you use AI4PDEs in research, teaching, or other activities, please cite the following publications if you use this code.

> Chen, B, CE Heaney, CC Pain (2024). Using AI libraries for Incompressible Computational Fluid Dynamics. [arXiv:2402.17913](https://arxiv.org/abs/2402.17913), DOI:[10.48550/arXiv.2402.17913](https://doi.org/10.48550/arXiv.2402.17913)


The bibtex entry for the paper is:

```bibtex
@misc{chen2024,
      title={Using AI libraries for Incompressible Computational Fluid Dynamics}, 
      author={Boyang Chen and Claire E. Heaney and Christopher C. Pain},
      year={2024},
      eprint={2402.17913},
      archivePrefix={arXiv},
}
```

Related papers using the AI4PDEs approach are

> T. R. F. Phillips, C. E. Heaney, B. Chen, A. G. Buchan and C. C. Pain. Solving the discretised neutron diffusion equations using neural networks. Int J Numer Methods Eng (2023) 124(21):4659--4686 <https://doi.org/10.1002/nme.7321>

> B. Chen, C. E. Heaney, J. L. M. A. Gomes, O. K. Matar and C. C. Pain. Solving the discretised multiphase flow equations with interface capturing on structured grids using machine learning libraries. Computer Methods in Applied Mechanics and Engineering (2024) 426:116974 <https://doi.org/10.1016/j.cma.2024.116974>
