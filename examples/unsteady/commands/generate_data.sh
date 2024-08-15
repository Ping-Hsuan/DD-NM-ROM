#!/bin/bash -i

### Slurm syntax
### ---------------
#SBATCH -N 1                     #number of nodes
#SBATCH -t 24:00:00              #walltime in hours:minutes:seconds
#SBATCH -e gen_data_err.txt      #stderr
#SBATCH -o gen_data_out.txt      #stdout
#SBATCH -J gen_data              #name of job
#SBATCH -p pbatch                #queue to use
#SBATCH -A sosu                  #account

### Shell scripting
### ---------------
### Loading conda environment thanks to interactive shell
### > See: 'dd-nm-rom/conda/README.md' file
load_conda_env_toss
### Launch program
python -u ./../scripts/generate_data.py --inpfile ./../inputs/generate_data.json
