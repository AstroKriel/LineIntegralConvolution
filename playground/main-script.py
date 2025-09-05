## { MODULE

## This file is part of the "LineIntegralConvolution" project.
## Copyright (c) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.

##
## === DEPENDENCIES ===
##

import sys
import time
import matplotlib.pyplot as mpl_plot
from pathlib import Path
from vegtamr.lic import compute_lic_with_postprocessing
from vegtamr.utils import vfields, plots

##
## === MAIN PROGRAM ===
##


def main():
    print("Running demo script...")
    num_pixels = 1000
    vfield_dict = vfields.vfield_lotka_volterra(size=num_pixels)
    vfield = vfield_dict["vfield"]
    streamlength = vfield_dict["streamlength"]
    bounds_rows = vfield_dict["bounds_rows"]
    bounds_cols = vfield_dict["bounds_cols"]
    vfield_name = vfield_dict["name"]
    ## apply the LIC multiple times: equivelant to applying several passes with a paint brush.
    ## note: `backend` options include "python" (this project) or "rust" (10x faster; https://github.com/tlorach/rLIC)
    print("Computing LIC...")
    start_time = time.perf_counter()
    sfield = compute_lic_with_postprocessing(
      vfield         = vfield,
      streamlength   = streamlength,
      num_lic_passes = 3,
      filter_sigma   = 5e-2 * num_pixels, # approx width of LIC tubes
      backend        = "rust",
    )
    elapsed_time = time.perf_counter() - start_time
    print(f"LIC execution took {elapsed_time:.3f} seconds.")
    print("Plotting data...")
    fig, ax = mpl_plot.subplots()
    plots.plot_lic(
        ax=ax,
        sfield=sfield,
        vfield=vfield,
        cmap_name="pink",
        bounds_rows=bounds_rows,
        bounds_cols=bounds_cols,
        overlay_streamlines=False,
        streamline_colour="royalblue",
        streamline_alpha=0.75,
    )
    print("Saving figure...")
    script_dir = Path(__file__).parent
    fig_path = script_dir / f"lic_{vfield_name}.png"
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
