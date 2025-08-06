## This file is part of the "LineIntegralConvolution" project.
## Copyright (c) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.


## ###############################################################
## DEPENDENCIES
## ###############################################################

import numpy
from scipy import ndimage
from skimage.exposure import equalize_adapthist


## ###############################################################
## FUNCTIONS
## ###############################################################

def filter_highpass(
    sfield : numpy.ndarray,
    sigma  : float = 3.0
  ) -> numpy.ndarray:
  lowpass = ndimage.gaussian_filter(sfield, sigma)
  gauss_highpass = sfield - lowpass
  return gauss_highpass

def rescaled_equalize(
    sfield                  : numpy.ndarray,
    num_subregions_rows     : int = 8,
    num_subregions_cols     : int = 8,
    clip_intensity_gradient : float = 0.01,
    num_intensity_bins      : int = 150,
  ) -> numpy.ndarray:
  min_val = sfield.min()
  max_val = sfield.max()
  is_rescale_needed = (max_val > 1.0) or (min_val < 0.0)
  ## rescale values to enhance local contrast
  ## note, output values are bound by [0, 1]
  sfield = equalize_adapthist(
    image       = sfield,
    kernel_size = (num_subregions_rows, num_subregions_cols),
    clip_limit  = clip_intensity_gradient,
    nbins       = num_intensity_bins,
  )
  ## rescale field back to its original value range
  if is_rescale_needed: sfield = sfield * (max_val - min_val) + min_val
  return sfield


## END OF MODULE
