## { MODULE

## This file is part of the "vegtamr" project.
## Copyright (c) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.

##
## === DEPENDENCIES
##

## third-party
import ahe
import numpy
from scipy import ndimage as scipy_ndimage

##
## === FUNCTIONS
##

def filter_lowpass(
    sfield: numpy.ndarray,
    sigma: float = 3.0,
) -> numpy.ndarray:
    return scipy_ndimage.gaussian_filter(sfield, sigma)

def filter_highpass(
    sfield: numpy.ndarray,
    sigma: float = 3.0,
) -> numpy.ndarray:
    return sfield - filter_lowpass(sfield, sigma)

def equalize_histogram(
    sfield: numpy.ndarray,
    num_subregions_rows: int = 9,
    num_subregions_cols: int = 9,
    max_normalized_bincount: float = 0.01,
) -> numpy.ndarray:
    return ahe.equalize_histogram(
        sfield.astype(numpy.float64),
        adaptive_strategy={
            "kind": "sliding-tile",
            "tile-size": (num_subregions_rows, num_subregions_cols),
        },
        max_normalized_bincount=max_normalized_bincount,
    )


## } MODULE
