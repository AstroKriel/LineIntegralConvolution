## { SCRIPT

## This file is part of the "vegtamr" project.
## Copyright (c) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.

##
## === DEPENDENCIES
##

## stdlib
import time
from pathlib import Path

## third-party
import matplotlib.pyplot as mpl_plot

## local
from vegtamr import lic
from vegtamr.utils import vfields, plots

##
## === PROGRAM MAIN
##


def main() -> None:
    print("Running demo script...")
    num_cells = 1000
    vfield_config = vfields.vfield_swirls(num_cells=num_cells)
    vfield = vfield_config.vfield
    streamlength = vfield_config.streamlength
    bounds_rows = vfield_config.bounds_rows
    bounds_cols = vfield_config.bounds_cols
    vfield_name = vfield_config.name
    ## apply the LIC multiple times: equivelant to applying several passes with a paint brush.
    ## note: `backend` options include "python" (this project) or "rust" (10x faster; https://github.com/tlorach/rLIC)
    print("Computing LIC...")
    start_time = time.perf_counter()
    sfield = lic.compute_lic_with_postprocessing(
      vfield         = vfield,
      streamlength   = streamlength,
      num_lic_passes = 3,
      use_filter     = True,
      filter_sigma   = 5e-2 * num_cells, # approx width of LIC tubes
      use_equalize   = True,
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
        cmap_range=(0.0, 0.75),
        bounds_rows=bounds_rows,
        bounds_cols=bounds_cols,
        overlay_streamlines=False,
        streamline_colour="royalblue",
        streamline_alpha=0.75,
    )
    print("Saving figure...")
    script_dir = Path(__file__).parent
    fig_path = script_dir / f"lic_{vfield_name}.png"
    fig.savefig(
        fig_path,
        dpi=300,
        bbox_inches="tight",
    )
    mpl_plot.close(fig)
    print("Saved:", fig_path)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
