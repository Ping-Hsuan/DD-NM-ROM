"""
Launch multiple testing jobs for DD-NM-ROM.
"""

import os
import sys
import json
import argparse

# Inputs
# =====================================
parser = argparse.ArgumentParser()
parser.add_argument("--inpfile", type=str, help="path to JSON input file")
args = parser.parse_args()

with open(args.inpfile) as file:
  inputs = json.load(file)

# Import 'dd_nm_rom' package
# =====================================
with open(inputs["pathfile"]) as file:
  paths = file.read().splitlines()
sys.path.extend(paths)

# Libraries
# =====================================
import subprocess
import numpy as np

from dd_nm_rom import ops

# Initialization
# =====================================
# Directories
for path in (
  inputs["paths"]["inp_dir"],
  inputs["paths"]["cmd_dir"]
):
  os.makedirs(path, exist_ok=True)

# Running
# =====================================
# Batch script function
# -------------------------------------
def generate_batch_script_toss(tag, inpfile):
  return f"\
#!/bin/bash -i                                                        \n\
                                                                      \n\
### Slurm syntax                                                      \n\
### ---------------                                                   \n\
#SBATCH -N 1                             #number of nodes             \n\
#SBATCH -t 24:00:00                      #walltime in hours:minutes   \n\
#SBATCH -e test_dd_nmrom_{tag}_err.txt   #stderr                      \n\
#SBATCH -o test_dd_nmrom_{tag}_out.txt   #stdout                      \n\
#SBATCH -J test_dd_nmrom_{tag}           #name of job                 \n\
#SBATCH -q pbatch                        #queue to use                \n\
#SBATCH -A sosu                          #account                     \n\
                                                                      \n\
### Shell scripting                                                   \n\
### ---------------                                                   \n\
### Loading conda environment thanks to interactive shell             \n\
### > See: 'dd-nm-rom/conda/README.md' file                           \n\
load_conda_env_toss                                                   \n\
### Launch program                                                    \n\
python -u ./../scripts/test_dd_nmrom.py --inpfile {inpfile}           \n\
"

def generate_batch_script_coral(tag, inpfile):
  return f"\
#!/bin/bash -i                                                        \n\
                                                                      \n\
### LSF syntax                                                        \n\
### ---------------                                                   \n\
#BSUB -nnodes 1                        #number of nodes               \n\
#BSUB -W 12:00                         #walltime in hours:minutes     \n\
#BSUB -e test_dd_nmrom_{tag}_err.txt   #stderr                        \n\
#BSUB -o test_dd_nmrom_{tag}_out.txt   #stdout                        \n\
#BSUB -J test_dd_nmrom_{tag}           #name of job                   \n\
#BSUB -q pbatch                        #queue to use                  \n\
#BSUB -G sosu                          #account                       \n\
                                                                      \n\
### Shell scripting                                                   \n\
### ---------------                                                   \n\
### Loading conda environment thanks to interactive shell             \n\
### > See: 'dd-nm-rom/conda/README.md' file                           \n\
load_conda_env_coral                                                  \n\
### Launch program                                                    \n\
python -u ./../scripts/test_dd_nmrom.py --inpfile {inpfile}           \n\
"

if (inputs["system"] == "coral"):
  generate_batch_script = generate_batch_script_coral
  batch_cmd = lambda cmdfile: f"bsub < {cmdfile}"
elif (inputs["system"] == "toss"):
  generate_batch_script = generate_batch_script_toss
  batch_cmd = lambda cmdfile: f"sbatch {cmdfile}"
else:
  raise ValueError("System not valid.")

# Generate all configurations
# -------------------------------------
dims, cfgs = {}, []
for element in inputs["elements"]:
  dims[element] = ops.generate_combs([
    np.arange(**inputs["dim"]["ranges"][element]["latent_dim"]),
    np.arange(**inputs["dim"]["ranges"][element]["row_nonzero"])
  ])
  cfgs.append(np.arange(len(dims[element])))
cfgs = ops.generate_combs(cfgs)

# Loop over configurations
# -------------------------------------
n_jobs = 0
for cfg in cfgs:
  # > Set tag
  tag_i = {}
  text = "\nLaunching testing for elements:"
  for (e, element) in enumerate(inputs["elements"]):
    ld, rnz = dims[element][cfg[e]]
    tag_i[element] = f"{element}_ld_{ld}_rnz_{rnz}"
    text += f"\n- '{element}' with (ld, rnz) = ({ld}, {rnz})"
  fulltag_i = "_".join(tag_i.values())
  print(text)
  # > Update input file
  with open(inputs["paths"]["inpfile"]) as file:
    inp_i = json.load(file)
  for (element, etag) in tag_i.items():
    inp_i["paths"]["nets_tag"][element] = etag
  # > Save input file
  inpfile_i = inputs["paths"]["inp_dir"] + f'/test_dd_nmrom_{fulltag_i}.json'
  with open(inpfile_i, 'w') as file:
    json.dump(inp_i, file, indent=2)
  # Batch script
  # -------------
  cmdfile_i = inputs["paths"]["cmd_dir"] + f'/test_dd_nmrom_{fulltag_i}.sh'
  with open(cmdfile_i, 'w') as file:
    file.write(generate_batch_script(fulltag_i, inpfile_i))
  # Launch program
  # -------------
  subprocess.run(
    batch_cmd(cmdfile_i),
    shell=True,
    timeout=1e2,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT
  )
  n_jobs += 1

print(f"\nTotal number of jobs: {n_jobs}\n")
