## { MODULE

## This file is part of the "vegtamr" project.
## Copyright (c) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.

##
## === DEPENDENCIES
##

## third-party
import numpy
import matplotlib.colors as mpl_colors
from matplotlib.axes import Axes as mpl_axes
from matplotlib import rcParams

rcParams["text.usetex"] = True

##
## === HELPER FUNCTIONS
##


def plot_lic(
    ax: mpl_axes,
    sfield: numpy.ndarray,
    vfield: numpy.ndarray,
    cmap_name: str = "pink",
    bounds_rows: tuple[float, float] | None = None,
    bounds_cols: tuple[float, float] | None = None,
    overlay_streamlines: bool = False,
    streamline_colour: str = "royalblue",
    streamline_alpha: float = 0.5,
):
    if bounds_rows is None: bounds_rows = (0.0, sfield.shape[0])
    if bounds_cols is None: bounds_cols = (0.0, sfield.shape[1])
    im = ax.imshow(
        sfield,
        cmap=cmap_name,
        origin="lower",
        extent=(
            bounds_cols[0],
            bounds_cols[1],
            bounds_rows[0],
            bounds_rows[1],
        ),
    )
    if overlay_streamlines:
        coords_row = numpy.linspace(bounds_rows[0], bounds_rows[1], sfield.shape[0])
        coords_col = numpy.linspace(bounds_cols[0], bounds_cols[1], sfield.shape[1])
        mg_x, mg_y = numpy.meshgrid(coords_col, coords_row, indexing="xy")
        ax.streamplot(
            mg_x,
            mg_y,
            vfield[0],
            vfield[1],
            color=mpl_colors.to_rgba(streamline_colour, alpha=streamline_alpha),
            arrowstyle="->",
            linewidth=1.5,
            density=0.5,
            arrowsize=1.0,
            broken_streamlines=False,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(bounds_cols)
    ax.set_ylim(bounds_rows)
    return im


def add_cbar(
    ax,
    mappable,
    label: str | None = "",
    label_size: float = 10,
    cbar_thickness: float = 0.1,
    cbar_padding: float = 0.02,
):
    fig = ax.figure
    box = ax.get_position()
    cbar_bounds = [
        box.x1 + cbar_padding,
        box.y0,
        box.width * cbar_thickness,
        box.height,
    ]
    ax_cbar = fig.add_axes(cbar_bounds)
    cbar = fig.colorbar(
        mappable=mappable,
        cax=ax_cbar,
        orientation="vertical",
    )
    cbar.set_label(
        label,
        fontsize=label_size,
        rotation=-90,
        va="bottom",
    )
    cbar.ax.yaxis.set_ticks_position("right")
    return cbar


## } MODULE
