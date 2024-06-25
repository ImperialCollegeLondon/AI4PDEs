# AI4PDEs

This repository contains code devloped by the authors of Using AI libraries for Incompressible Computational Fluid Dynamics (2024) <https://arxiv.org/abs/2402.17913> Boyang Chen, Claire E. Heaney and Christopher C. Pain. 

Please cite this work if you use this code.

Related papers using the AI4PDEs approach are: 
T. R. F. Phillips, C. E. Heaney, B. Chen, A. G. Buchan and C. C. Pain. Solving the discretised neutron diffusion equations using neural networks. Int J Numer Methods Eng (2023) 124(21):4659--4686 <https://doi.org/10.1002/nme.7321>

B. Chen, C. E. Heaney, J. L. M. A. Gomes, O. K. Matar and C. C. Pain. Solving the discretised multiphase flow equations with interface capturing on structured grids using machine learning libraries. Computer Methods in Applied Mechanics and Engineering (2024) 426:116974 <https://doi.org/10.1016/j.cma.2024.116974> 


## Installation

1. **Clone the repo**
   ```sh
   git clone https://github.com/ImperialCollegeLondon/AI4PDEs.git
   ```
2. **Install**
   ```pip install . ``` or 
   ```pip install -e ."[dev]"```

3. **Run one of the example files**
   ```
   cd ai4pdes/structured/
   jupyter notebook flow_past_block.ipynb
   jupyter notebook advection_diffusion.ipynb
   jupyter notebook flow_past_buildings.ipynb
   ```
<!--   
*** ## Usage
*** 
*** ## Contributing
***
*** Contributions are welcome! If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star! Thanks again!

*** 1. Fork the Project
*** 2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
*** 3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
*** 4. Push to the Branch (`git push origin feature/AmazingFeature`)
*** 5. Open a Pull Request
-->

## License

Distributed under the *** License. See `LICENSE` for more information.

