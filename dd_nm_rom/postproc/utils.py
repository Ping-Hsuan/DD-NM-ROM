# def set_style(pyplot):
#   pyplot.rc('font', size=20)
#   pyplot.rcParams['lines.linewidth'] = 4
#   pyplot.rcParams['text.usetex'] = True
#   return pyplot

def set_style(pyplot):
  pyplot.rc("font", size=20)
  pyplot.rcParams["text.usetex"] = False
  pyplot.rcParams["figure.max_open_warning"] = int(1e3)
  return pyplot
