"""
Compute port SVD for DD-LS-ROM.

Supports multiple datasets (steady and unsteady) via 'data_loads' list in config.
Steady snapshots (nxy, 2) are reshaped to (1, 2*nxy) per sample.
Unsteady snapshots (nt+1, 2*nxy) are used as-is (each row = one time step).
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

# Environment
# =====================================
from dd_nm_rom import env
env.set(**inputs["env"])

# Libraries
# =====================================
import shutil
import numpy as np
import dill as pickle

from dd_nm_rom import ops
from dd_nm_rom import utils
from dd_nm_rom import fom as fom_mod
from dd_nm_rom import field as field_mod
from dd_nm_rom.elements import mesh as mesh_mod
from dd_nm_rom.rom.utils import pod as pod_mod

# Initialization
# =====================================
print("\nInitialization ...")
# Mesh
mesh = utils.get_class(modules=[mesh_mod], **inputs["mesh"])
mesh.build()
# Field (used only to build FOM/DD-FOM structure)
field = utils.get_class(
  modules=[field_mod],
  name=inputs["field"]["name"]
)(mesh=mesh, **inputs["field"]["kwargs"])
field.set_params(mu=field.sample_design_space())
# FOM
fom = utils.get_class(
  modules=[fom_mod],
  name=inputs["fom"]["name"]
)(mesh=mesh, **inputs["fom"]["kwargs"])
fom.build(field)
# DD-FOM
dd_fom = utils.get_class(
  modules=[fom_mod],
  name=inputs["dd_fom"]["name"]
)(monolithic=fom, **inputs["dd_fom"]["kwargs"])
dd_fom.build()

# Data Loading
# =====================================
print("\nLoading data ...")
all_snaps = []
for cfg in inputs["data_loads"]:
  is_steady = cfg.get("steady", True)
  load_cfg = {k: v for k, v in cfg.items() if k != "steady"}
  data = utils.load_case_parallel(**load_cfg)
  n_loaded = 0
  for x in data:
    if x is None:
      continue
    if is_steady:
      # (nxy, 2) -> (1, 2*nxy): first col = u, second col = v
      row = np.concatenate([x[:, 0], x[:, 1]]).reshape(1, -1)
      all_snaps.append(row)
    else:
      # (nt+1, 2*nxy): each row is already a full state snapshot
      all_snaps.append(x)
    n_loaded += 1
  label = "steady" if is_steady else "unsteady"
  print(f"  [{label}] {load_cfg['path'].split('LLM_data/')[-1].split('/dset')[0]}: {n_loaded} samples")

dataset = np.vstack(all_snaps)
print(f"\nTotal snapshots: {dataset.shape[0]}  (state dim: {dataset.shape[1]})")

print("> Map dataset on DD elements")
dataset = dd_fom.map_sol_on_elements(dataset, map_on_ports=True)
print(f"  port[0]: {dataset['port'][0].shape}")

# Linear subspace
# =====================================
print("\nBuilding linear subspace...")
# Sweep specification: ("emin", value) selects bases by energy threshold
# (as before); ("n_bases", value) fixes the exact port dimension directly,
# bypassing energy_min -- useful to sweep port dimension independently while
# interior stays fixed at whatever emin/dimension it was already trained at
# (interior and port are trained by separate scripts into separately-tagged
# directories, so there's no need to retrain interior to hold it fixed).
sweep_specs = (
  [("emin", e) for e in inputs.get("svd_emins", [])] +
  [("n_bases", n) for n in inputs.get("svd_n_bases", [])]
)
if not sweep_specs:
  sweep_specs = [("emin", e) for e in [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]]

for (kind, val) in sweep_specs:
  ls_models = []
  path = inputs["model"]["path"]
  path += "/merged/" if inputs["trainable"]["merged"] else "/multi/"
  tag = f"emin_{val}" if kind == "emin" else f"nbases_{val}"
  path += f"/port_{tag}/"

  for (orient, size_to_ports) in dd_fom.dd_indices.orientsize_to_ports.items():
    for (size, ports) in size_to_ports.items():
      if inputs["trainable"]["merged"]:
        ls_models.append((orient, size, ports))
        if kind == "emin":
          inputs["svd"]["energy_min"]["port"] = val
          inputs["svd"]["n_bases"]["port"] = -1
        else:
          inputs["svd"]["n_bases"]["port"] = val
      else:
        for p in ports:
          ls_models.append((orient, size, [p]))

  for model in ls_models:
    orient, size, ports = model
    print(f"\n> Training ports {ports} of '{orient}' orientation and size {size} ...")
    path_i = path + f"/orient_{orient}_size_{size}_ports"
    for p in ports:
      path_i += f"_{p}"
    path_i = path_i + "/scratch/"

    dataset_merged = {"port": [np.vstack([dataset["port"][p] for p in ports])]}
    print(f"  port data shape: {dataset_merged['port'][0].shape}")

    # Perform SVD
    svd, bases = pod_mod.compute_svd(
      data=dataset_merged,
      **inputs["svd"]
    )
    # Saving
    os.makedirs(path_i, exist_ok=True)
    pickle.dump(svd, open(path_i + "/svd.p", "wb"))
    if bases is not None:
      pickle.dump(bases, open(path_i + "/bases.p", "wb"))
      n_bases = ops.map_nested_dict(bases, lambda x: x.shape[1])
      with open(path_i + "/n_bases.json", "w") as file:
        json.dump(n_bases, file, indent=2)

    shutil.copyfile(args.inpfile, path_i + "/inputs.json")

  print("\nDone!\n")
