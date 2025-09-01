## { MODULE

## This file is part of the "vegtamr" project.
## Copyright (c) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.


##
## === DEPENDENCIES ===
##

import sys
import matplotlib.pyplot as mpl_plot
from pathlib import Path
from vegtamr.lic import compute_lic_with_postprocessing
from vegtamr.utils import vfields, plots


##
## === HELPER FUNCTIONS ===
##

def format_text_for_latex(string):
  moified_string = string.replace(" ", " \;")
  return rf"$\mathrm{{{moified_string}}}$"


##
## === MAIN PROGRAM ===
##

def main():
  print("Running demo script...")
  num_pixels   = 500
  vfield_dict  = vfields.vfield_swirls(size=num_pixels, num_swirls=4)
  vfield       = vfield_dict["vfield"]
  bounds_rows  = vfield_dict["bounds_rows"]
  bounds_cols  = vfield_dict["bounds_cols"]
  ideal_streamlength = vfield_dict["streamlength"]
  streamlengths = [
    ideal_streamlength / 2,
    ideal_streamlength,
    ideal_streamlength * 4,
  ]
  num_cols = len(streamlengths)
  axis_length = 2.5
  fig, axs = mpl_plot.subplots(
    nrows = 3,
    ncols = num_cols,
    figsize = (num_cols*axis_length, 3*axis_length),
  )
  fig.subplots_adjust(wspace=0.05, hspace=0.05)
  num_rows = axs.shape[0] 
  print("Computing LIC...")
  for row_index in range(num_rows):
    use_filter = row_index > 0
    use_equalize = row_index > 1
    for col_index, streamlength in enumerate(streamlengths):
      sfield = compute_lic_with_postprocessing(
        vfield       = vfield,
        streamlength = int(streamlength),
        filter_sigma = 5e-2 * num_pixels,
        use_filter   = use_filter,
        use_equalize = use_equalize,
        backend      = "rust",
        verbose      = False,
      )
      print(f"Plotting axs[{row_index},{col_index}]")
      ax = axs[row_index, col_index]
      im = plots.plot_lic(
        ax          = ax,
        sfield      = sfield,
        vfield      = vfield,
        bounds_rows = bounds_rows,
        bounds_cols = bounds_cols,
        cmap_name   = "twilight_shifted" if (row_index < num_rows-1) else "pink",
      )
      if col_index == num_rows-1:
        if row_index < num_rows-1:
          label = r"a diverging cmap works best"
        else: label = r"a sequential cmap works best"
        plots.add_cbar(
          ax,
          mappable = im,
          label    = format_text_for_latex(label),
        )
  for col_index, streamlength in enumerate(streamlengths):
    axs[0, col_index].set_title(
      rf"$L_\mathrm{{stream}} = {int(streamlength)} \;\mathrm{{pixels}}$",
      fontsize=10
    )
  white_transparent_box = dict(
    facecolor = "white",
    edgecolor = "white",
    boxstyle  = "round,pad=0.3",
    alpha     = 0.75,
  )
  axs[0,0].text(
    0.05, 0.95,
    r"$N_\mathrm{pixels} = %d$" % num_pixels,
    ha = "left",
    va = "top",
    transform = axs[0,0].transAxes,
    bbox = white_transparent_box
  )
  axs[0,0].set_ylabel(format_text_for_latex("no post-processing"), fontsize=10)
  axs[1,0].set_ylabel(format_text_for_latex("highpass filter enabled"), fontsize=10)
  axs[2,0].set_ylabel(format_text_for_latex("highpass filter enabled") + r" $\newline$ " + format_text_for_latex("histogram equalisation enabled"), fontsize=10)
  print("Saving figure...")
  script_dir = Path(__file__).parent
  fig_path = script_dir / f"effect_of_params.png"
  fig.savefig(fig_path, dpi=300, bbox_inches="tight")
  mpl_plot.close(fig)
  print("Saved:", fig_path)


##
## === ENTRY POINT ===
##

if __name__ == "__main__":
  main()
  sys.exit(0)


## } SCRIPT
