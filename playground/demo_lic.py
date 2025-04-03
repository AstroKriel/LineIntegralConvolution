## This file is part of the "Vegtamr" project.
## Copyright (c) 2024 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.


## ###############################################################
## DEPENDENCIES
## ###############################################################
import sys
import matplotlib.pyplot as mplplot
from vegtamr import vfields, lic, utils


## ###############################################################
## MAIN PROGRAM
## ###############################################################
@utils.time_func
def main(
    func_vfield     ,
    size            : int,
    num_iterations  : int = 1,
    num_repetitions : int = 1,
    use_filter      : bool = True,
    filter_sigma    : float = 3.0,
    use_equalize    : bool = True,
    debug_mode      : bool = False,
  ):
  print("Started running demo script...")
  vfield_dict  = func_vfield(size)
  vfield       = vfield_dict["vfield"]
  streamlength = vfield_dict["streamlength"]
  bounds_rows  = vfield_dict["bounds_rows"]
  bounds_cols  = vfield_dict["bounds_cols"]
  vfield_name  = vfield_dict["name"]
  ## apply the LIC a few times: equivelant to painting over with a few brush strokes
  print("Computing LIC...")
  sfield = lic.compute_lic_with_postprocessing(
    vfield          = vfield,
    streamlength    = streamlength,
    num_iterations  = num_iterations,
    num_repetitions = num_repetitions,
    use_filter      = use_filter,
    filter_sigma    = filter_sigma,
    use_equalize    = use_equalize,
  )
  ## visualise the LIC
  print("Plotting data...")
  fig, _ = utils.plot_lic(
    sfield      = sfield,
    vfield      = vfield,
    bounds_rows = bounds_rows,
    bounds_cols = bounds_cols,
    debug_mode  = debug_mode,
  )
  ## save and close the figure
  print("Saving figure...")
  fig_name = f"lic_{vfield_name}.png"
  fig.savefig(fig_name, dpi=300, bbox_inches="tight")
  mplplot.close(fig)
  print("Saved:", fig_name)


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main(
    func_vfield     = vfields.vfield_swirls, # pass function reference (not a function call): without brackets
    size            = 100,
    num_iterations  = 3,
    num_repetitions = 3,
    use_filter      = True,
    filter_sigma    = 3.0, # lower values produce thinner LIC tubes
    use_equalize    = True,
    debug_mode      = False,
  )
  sys.exit(0)


## END OF SCRIPT
