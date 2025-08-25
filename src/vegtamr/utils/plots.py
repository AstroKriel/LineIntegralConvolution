## This file is part of the "vegtamr" project.
## Copyright (c) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.


## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
from matplotlib.colors import to_rgba
from matplotlib.axes import Axes as mpl_axes


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################

def plot_lic(
    ax                  : mpl_axes,
    sfield              : numpy.ndarray,
    vfield              : numpy.ndarray,
    bounds_rows         : tuple[float, float] | None = None,
    bounds_cols         : tuple[float, float] | None = None,
    overlay_streamlines : bool = False,
    streamline_colour   : str = "orange",
    streamline_alpha    : float = 0.25,
  ):
  if bounds_rows is None: bounds_rows = (0.0, sfield.shape[0])
  if bounds_cols is None: bounds_cols = (0.0, sfield.shape[1]) 
  ax.imshow(
    sfield,
    cmap   = "bone",
    origin = "lower",
    extent = (
      bounds_rows[0], bounds_rows[1],
      bounds_cols[0], bounds_cols[1]
    ),
  )
  if overlay_streamlines:
    coords_row = numpy.linspace(bounds_rows[0], bounds_rows[1], sfield.shape[0])
    coords_col = numpy.linspace(bounds_cols[0], bounds_cols[1], sfield.shape[1])
    mg_x, mg_y = numpy.meshgrid(coords_col, coords_row, indexing="xy")
    ax.streamplot(
      mg_x, mg_y,
      vfield[0], vfield[1],
      color              = to_rgba(streamline_colour, alpha=streamline_alpha),
      arrowstyle         = "->",
      linewidth          = 1.0,
      density            = 0.5,
      arrowsize          = 0.5,
      broken_streamlines = False,
    )
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_xlim(bounds_rows)
  ax.set_ylim(bounds_cols)


## END OF MODULE