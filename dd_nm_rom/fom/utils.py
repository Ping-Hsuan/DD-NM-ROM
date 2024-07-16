from . import MeshMono, MeshDD


def get_mesh(config):
  if ("mono" in config):
    mesh_mn = MeshMono(**config["mono"])
    mesh_mn.build()
    mesh_dd_cfg = mesh_mn.get_config_dd(**config["dd"])
  else:
    mesh_dd_cfg = config
  mesh_dd = MeshDD(**mesh_dd_cfg)
  mesh_dd.build()
  return mesh_dd
