SIDES = {
  "x": ("left", "right"),
  "y": ("bottom", "top")
}
SIDES["all"] = SIDES["x"] + SIDES["y"]
TYPES = ("dirichlet", "neumann")

def check_bc_type(bc_type):
  if (bc_type not in TYPES):
    raise ValueError(
      f"Could not interpret b.c. type: '{bc_type}'." \
        f"Valid options are: {TYPES}"
    )

def check_side(side):
  if (side not in SIDES["all"]):
    raise ValueError(
      f"Could not interpret b.c. side: '{side}'." \
        f"Valid options are: {SIDES['all']}"
    )

def get_axis_method(side):
  check_side(side)
  axis = "x" if (side in SIDES["x"]) else "y"
  method = "fwd" if (side in ("left", "bottom")) else "bwd"
  return axis, method
