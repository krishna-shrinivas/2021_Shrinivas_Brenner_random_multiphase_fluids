## Phase-field simulations of multiphase behavior in fluids with randomly interacting components
This is a readme for the documentation of the phase-field simulation model. Running this code requires 
pre-requisites (below) and  recommended to be run in a separate conda environment

### Requirements for Code
- python (>=3.7)
- numpy
- numexpr
- moviepy
- matplotlib

### How to run code
The code can be run by running run_pfsim.py with input and output flags

`` python run_pfsim --flag_name flag_params ``

#### Required flags

`` --i /path/to/input_params.file ``
Input parameters (example provided along with the code under /Input/input_params.txt)

`` --o /path/to/output_folder/ ``
Path to output folder


#### Optional flags

`` --p /path/to/parameter_sweep.file ``
Values of single parameter to sweep over (example provided along with the code under /Input/params_file.txt)

Note when using this - if only the effect of the parameter is to studied, best to fix a seed instead of using
a random seed. 

`` --pN x``
Run a simulation with param in line (x) from param file

`` --test 0/1``
If 1, runs a test setup - initialization occurs but no phase-field simulation is run and outputs are not generated to be stored.

