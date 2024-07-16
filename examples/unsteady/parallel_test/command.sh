#!/bin/bash -i

### Shell scripting
### ---------------
### Loading conda environment thanks to interactive shell
### > See: 'dd-nm-rom/conda/README.md' file
# load_conda_env_toss
load_conda_env_coral
### Launch program
python -u main.py --inpfile inputs.json
