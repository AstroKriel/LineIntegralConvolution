## This file is part of the "Vegtamr" project.
## Copyright (c) 2024 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.


## ###############################################################
## DEPENDENCIES
## ###############################################################
import time
import numpy
import matplotlib.pyplot as mplplot
from scipy import ndimage
from matplotlib.colors import to_rgba
from skimage.exposure import equalize_adapthist


## ###############################################################
## START OF UTILITY FUNCTIONS
## ###############################################################
def time_func(func):
    def wrapper(*args, **kwargs):
        time_start = time.time()
        result = func(*args, **kwargs)
        time_elapsed = time.time() - time_start
        print(f"{func.__name__}() took {time_elapsed:.3f} seconds to execute.")
        return result
    return wrapper

def filter_highpass(sfield: numpy.ndarray, sigma: float = 3.0):
    lowpass = ndimage.gaussian_filter(sfield, sigma)
    gauss_highpass = sfield - lowpass
    return gauss_highpass

def rescaled_equalize(
    sfield: numpy.ndarray,
    num_subregions_rows: int = 8,
    num_subregions_cols: int = 8,
    clip_intensity_gradient: float = 0.01,
    num_intensity_bins: int = 150,
):
    min_val = sfield.min()
    max_val = sfield.max()
    bool_rescale_needed = (max_val > 1.0) or (min_val < 0.0)
    ## rescale values to enhance local contrast
    ## note, output values are bound by [0, 1]
    sfield = equalize_adapthist(
        image=sfield,
        kernel_size=(num_subregions_rows, num_subregions_cols),
        clip_limit=clip_intensity_gradient,
        nbins=num_intensity_bins,
    )
    ## rescale field back to its original value range
    if bool_rescale_needed:
        sfield = sfield * (max_val - min_val) + min_val
    return sfield

def plot_lic(
    sfield: numpy.ndarray,
    vfield: numpy.ndarray,
    bounds_rows=None,
    bounds_cols=None,
    bool_debug=False,
):
    fig, ax = mplplot.subplots(figsize=(6, 6))
    ax.imshow(
        sfield,
        cmap="bone",
        origin="lower",
        extent=[bounds_rows[0], bounds_rows[1], bounds_cols[0], bounds_cols[1]],
    )
    if bool_debug:
        coords_row = numpy.linspace(bounds_rows[0], bounds_rows[1], sfield.shape[0])
        coords_col = numpy.linspace(bounds_cols[0], bounds_cols[1], sfield.shape[1])
        mg_x, mg_y = numpy.meshgrid(coords_col, coords_row, indexing="xy")
        color = to_rgba("green", alpha=0.75)
        ax.streamplot(
            mg_x,
            mg_y,
            vfield[0],
            vfield[1],
            color=color,
            arrowstyle="->",
            linewidth=1.0,
            density=0.5,
            arrowsize=0.5,
            broken_streamlines=False,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(bounds_rows)
    ax.set_ylim(bounds_cols)
    return fig, ax


## END OF UTILITY FUNCTIONS
