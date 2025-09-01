## { MODULE

## This file is part of the "LineIntegralConvolution" project.
## Copyright (c) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.

##
## === DEPENDENCIES ===
##

import sys
import time
import numpy
import matplotlib.pyplot as mpl_plot
from pathlib import Path
from vegtamr.lic import compute_lic_with_postprocessing
from vegtamr.utils import plots

##
## === MAIN PROGRAM ===
##


def main():
    script_dir = Path(__file__).parent
    file_path = script_dir / "quantities_planez_100_rotated_projected_ccom.npz"
    quantities = numpy.load(file_path, allow_pickle=True)
    _, _, _, _, slice_bx, slice_by, slice_b_magn, _, _, _, _, _, _ = quantities["arr_0"]
    slice_log10_b_magn = numpy.where(slice_b_magn > 0, numpy.log10(slice_b_magn), -9)
    vfield = numpy.array([slice_bx, slice_by])
    num_pixels = numpy.max(slice_by.shape)
    print("Computing LIC...")
    start_time = time.perf_counter()
    sfield = compute_lic_with_postprocessing(
      vfield         = vfield,
      streamlength   = int(num_pixels // 20),
      num_lic_passes = 3,
      use_equalize   = False,
      use_filter     = False,
      filter_sigma   = 2e-2 * num_pixels, # approx width of LIC tubes
      backend        = "python",
    )
    elapsed_time = time.perf_counter() - start_time
    print(f"LIC execution took {elapsed_time:.3f} seconds.")
    print("Plotting data...")
    scaled_sfield = sfield * slice_log10_b_magn
    fig, ax = mpl_plot.subplots()
    im = plots.plot_lic(
        ax=ax,
        sfield=scaled_sfield,
        vfield=vfield,
        bounds_rows=(0, 1),
        bounds_cols=(0, 1),
        overlay_streamlines=False,
        streamline_colour="white",
        streamline_alpha=0.25,
    )
    im.set_clim(vmin=-3.5, vmax=-1)
    cbar = plots.add_cbar(
        ax=ax,
        mappable=im,
        side="top",
        label=r"$\log_{10}(E_\mathrm{mag} / \mu G)$",
        percentage=0.05,
        fontsize=20,
    )
    cbar.set_ticks([-3.5, -2.25, -1])
    cbar.set_ticklabels([r"$-1$", r"$0$", r"$1$"])
    print("Saving figure...")
    fig_path = script_dir / f"lic_lmc.png"
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
