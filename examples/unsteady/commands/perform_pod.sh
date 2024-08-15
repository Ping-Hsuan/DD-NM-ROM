#!/bin/bash -i

### Slurm syntax
### ---------------
#SBATCH -N 1                     #number of nodes
#SBATCH -t 24:00:00              #walltime in hours:minutes:seconds
#SBATCH -e pod_err.txt      #stderr
#SBATCH -o pod_out.txt      #stdout
#SBATCH -J pod              #name of job
#SBATCH -p pbatch                #queue to use
#SBATCH -A sosu                  #account

### Shell scripting
### ---------------
### Loading conda environment thanks to interactive shell
### > See: 'dd-nm-rom/conda/README.md' file
load_conda_env_toss
### Launch program
python -u ./../../steady/scripts/perform_pod.py --inpfile ./../inputs/perform_pod.json
