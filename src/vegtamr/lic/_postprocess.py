## { MODULE

## This file is part of the "vegtamr" project.
## Copyright (c) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.

##
## === DEPENDENCIES
##

## third-party
import numpy
from scipy import ndimage as scipy_ndimage
from skimage import exposure as skimage_exposure

##
## === FUNCTIONS
##


def filter_highpass(
    sfield: numpy.ndarray,
    sigma: float = 3.0,
) -> numpy.ndarray:
    lowpass = scipy_ndimage.gaussian_filter(sfield, sigma)
    gauss_highpass = sfield - lowpass
    return gauss_highpass


def rescaled_equalize(
    sfield: numpy.ndarray,
    num_subregions_rows: int = 8,
    num_subregions_cols: int = 8,
    clip_intensity_gradient: float = 0.01,
    num_intensity_bins: int = 150,
) -> numpy.ndarray:
    min_val = sfield.min()
    max_val = sfield.max()
    is_rescale_needed = (max_val > 1.0) or (min_val < 0.0)
    ## `equalize_adapthist` expects input already normalised to [0, 1]; it clips negative values instead of rescaling them
    if is_rescale_needed: sfield = (sfield - min_val) / (max_val - min_val)
    ## rescale values to enhance local contrast
    ## note, output values are bound by [0, 1]
    sfield = skimage_exposure.equalize_adapthist(
        image=sfield,
        kernel_size=(num_subregions_rows, num_subregions_cols),
        clip_limit=clip_intensity_gradient,
        nbins=num_intensity_bins,
    )
    ## rescale field back to its original value range
    if is_rescale_needed: sfield = sfield * (max_val - min_val) + min_val
    return sfield


## } MODULE
