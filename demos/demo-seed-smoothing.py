## { SCRIPT

## This file is part of the "vegtamr" project.
## Copyright (c) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.

##
## === DEPENDENCIES
##

## stdlib
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
    num_cells = 500
    vfield_config = vfields.vfield_swirls(
        num_cells=num_cells,
        num_swirls=4,
    )
    vfield = vfield_config.vfield
    bounds_rows = vfield_config.bounds_rows
    bounds_cols = vfield_config.bounds_cols
    streamlength = vfield_config.streamlength
    seed_smoothing_sigmas = [0.0, 1.0, 3.0]
    num_cols = len(seed_smoothing_sigmas)
    axis_length = 2.5
    fig, axs_row = mpl_plot.subplots(
        nrows=1,
        ncols=num_cols,
        figsize=(num_cols * axis_length, axis_length),
    )
    fig.subplots_adjust(wspace=0.05)
    print("Computing LIC...")
    for col_index, seed_smoothing_sigma in enumerate(seed_smoothing_sigmas):
        sfield = lic.compute_lic_with_postprocessing(
            vfield=vfield,
            streamlength=streamlength,
            filter_sigma=5e-2 * num_cells,
            use_filter=True,
            use_equalize=True,
            seed_smoothing_sigma=seed_smoothing_sigma,
            backend="rust",
            verbose=False,
        )
        print(f"Plotting axs_row[{col_index}]")
        ax = axs_row[col_index]
        plots.plot_lic(
            ax=ax,
            sfield=sfield,
            vfield=vfield,
            bounds_rows=bounds_rows,
            bounds_cols=bounds_cols,
            cmap_name="pink",
            cmap_range=(0.0, 0.75),
        )
        ax.set_title(
            rf"$\sigma_\mathrm{{seed}} = {seed_smoothing_sigma:.1f} \;\mathrm{{pixels}}$",
            fontsize=10,
        )
    white_transparent_box = dict(
        facecolor="white",
        edgecolor="white",
        boxstyle="round,pad=0.3",
        alpha=0.75,
    )
    axs_row[0].text(
        0.05,
        0.95,
        r"$N_\mathrm{pixels} = %d$" % num_cells,
        ha="left",
        va="top",
        transform=axs_row[0].transAxes,
        bbox=white_transparent_box,
    )
    print("Saving figure...")
    script_dir = Path(__file__).parent
    fig_path = script_dir / "effect_of_seed_smoothing.png"
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
