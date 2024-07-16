#!/bin/bash -i

### LSF syntax
### ---------------
#BSUB -nnodes 1                  #number of nodes
#BSUB -W 12:00                   #walltime in hours:minutes
#BSUB -e pod2_err.txt  #stderr
#BSUB -o pod2_out.txt  #stdout
#BSUB -J pod2          #name of job
#BSUB -q pbatch                  #queue to use
#BSUB -G sosu                    #account

### Shell scripting
### ---------------
### Loading conda environment thanks to interactive shell
### > See: 'dd-nm-rom/conda/README.md' file
load_conda_env_coral
### Launch program
python -u ./../../steady/scripts/perform_pod.py --inpfile ./../inputs/perform_pod2.json
