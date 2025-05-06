## This file is part of the "LineIntegralConvolution" project.
## Copyright (c) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from vegtamr.lic import _core


## ###############################################################
## LOOP THROUGH THE DOMAIN SERIALLY
## ###############################################################
def compute_lic(
    vfield           : numpy.ndarray,
    sfield_in        : numpy.ndarray,
    sfield_out       : numpy.ndarray,
    streamlength     : int,
    num_rows         : int,
    num_cols         : int,
    use_periodic_BCs : bool,
  ) -> numpy.ndarray:
  """
  Perform a Line Integral Convolution (LIC) over the entire domain by tracing streamlines from each pixel in both
  forward and backward directions along the vector field.
  """
  for row_index in range(num_rows):
    for col_index in range(num_cols):
      forward_sum, forward_total = _core.advect_streamline(
        vfield           = vfield,
        sfield_in        = sfield_in,
        start_row        = row_index,
        start_col        = col_index,
        dir_sgn          = +1,
        streamlength     = streamlength,
        use_periodic_BCs = use_periodic_BCs,
      )
      backward_sum, backward_total = _core.advect_streamline(
        vfield           = vfield,
        sfield_in        = sfield_in,
        start_row        = row_index,
        start_col        = col_index,
        dir_sgn          = -1,
        streamlength     = streamlength,
        use_periodic_BCs = use_periodic_BCs,
      )
      total_sum = forward_sum + backward_sum
      total_weight = forward_total + backward_total
      if total_weight > 0.0:
        sfield_out[row_index, col_index] = total_sum / total_weight
      else: sfield_out[row_index, col_index] = 0.0
  return sfield_out


## END OF MODULE