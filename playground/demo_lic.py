## This file is part of the "Vegtamr" project.
## Copyright (c) 2024 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.


## ###############################################################
## DEPENDENCIES
## ###############################################################
import sys
import matplotlib.pyplot as mpl_plot
from vegtamr import vfields, lic, utils


## ###############################################################
## MAIN PROGRAM
## ###############################################################
@utils.time_func
def main(
    func_vfield,
    size                   : int,
    num_lic_passes         : int = 1,
    num_postprocess_cycles : int = 1,
    use_filter             : bool = True,
    filter_sigma           : float = 3.0,
    use_equalize           : bool = True,
    overlay_streamlines    : bool = False,
    backend                : str = "rust",
  ):
  print("Started running demo script...")
  vfield_dict  = func_vfield(size)
  vfield       = vfield_dict["vfield"]
  streamlength = vfield_dict["streamlength"]
  bounds_rows  = vfield_dict["bounds_rows"]
  bounds_cols  = vfield_dict["bounds_cols"]
  vfield_name  = vfield_dict["name"]
  ## apply the LIC multiple times: equivelant to applying several passes with a paint brush
  print("Computing LIC...")
  sfield = lic.compute_lic_with_postprocessing(
    vfield                 = vfield,
    streamlength           = streamlength,
    num_lic_passes         = num_lic_passes,
    num_postprocess_cycles = num_postprocess_cycles,
    use_filter             = use_filter,
    filter_sigma           = filter_sigma,
    use_equalize           = use_equalize,
    backend                = backend,
  )
  ## visualise the LIC
  print("Plotting data...")
  fig, _ = utils.plot_lic(
    sfield              = sfield,
    vfield              = vfield,
    bounds_rows         = bounds_rows,
    bounds_cols         = bounds_cols,
    overlay_streamlines = overlay_streamlines,
  )
  ## save and close the figure
  print("Saving figure...")
  fig_name = f"lic_{vfield_name}.png"
  fig.savefig(fig_name, dpi=300, bbox_inches="tight")
  mpl_plot.close(fig)
  print("Saved:", fig_name)


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main(
    func_vfield            = vfields.vfield_flowers, # pass function reference (not a function call; without round-brackets)
    size                   = 500,
    num_lic_passes         = 1,
    num_postprocess_cycles = 1,
    use_filter             = False,
    filter_sigma           = 2.0, # lower values produce thinner LIC tubes
    use_equalize           = False,
    backend                = "rust", # options: "python" (implemented in this project) or "rust" (100x faster; https://github.com/tlorach/rLIC)
  )
  sys.exit(0)


## END OF SCRIPT
