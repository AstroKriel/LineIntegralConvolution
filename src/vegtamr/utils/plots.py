## { MODULE

## This file is part of the "vegtamr" project.
## Copyright (c) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.

##
## === DEPENDENCIES ===
##

import numpy
from matplotlib.colors import to_rgba
from matplotlib.axes import Axes as mpl_axes

from matplotlib import rcParams

rcParams["text.usetex"] = True

##
## === HELPER FUNCTIONS ===
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
            color=to_rgba(streamline_colour, alpha=streamline_alpha),
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
    side: str = "right",
    percentage: float = 0.1,
    cbar_padding: float = 0.02,
    label_padding: float = 10,
    fontsize: float = 10,
):
    fig = ax.figure
    box = ax.get_position()
    if side in ["left", "right"]:
        orientation = "vertical"
        cbar_size = box.width * percentage
        if side == "right":
            cbar_bounds = [box.x1 + cbar_padding, box.y0, cbar_size, box.height]
        else:
            cbar_bounds = [box.x0 - cbar_size - cbar_padding, box.y0, cbar_size, box.height]
    elif side in ["top", "bottom"]:
        orientation = "horizontal"
        cbar_size = box.height * percentage
        if side == "top":
            cbar_bounds = [box.x0, box.y1 + cbar_padding, box.width, cbar_size]
        else:
            cbar_bounds = [box.x0, box.y0 - cbar_size - cbar_padding, box.width, cbar_size]
    else:
        raise ValueError(f"Unsupported side: {side}")
    ax_cbar = fig.add_axes(cbar_bounds)
    cbar = fig.colorbar(mappable=mappable, cax=ax_cbar, orientation=orientation)
    if orientation == "horizontal":
        cbar.ax.set_title(label, fontsize=fontsize, pad=label_padding)
        cbar.ax.xaxis.set_ticks_position(side)
    else:
        cbar.set_label(label, fontsize=fontsize, rotation=-90, va="bottom")
        cbar.ax.yaxis.set_ticks_position(side)
    return cbar


## } MODULE
