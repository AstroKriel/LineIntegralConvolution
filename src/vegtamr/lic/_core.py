## This file is part of the "Vegtamr" project.
## Copyright (c) 2024 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy


## ###############################################################
## LIC IMPLEMENTATION
## ###############################################################
def taper_pixel_contribution(
    streamlength : int,
    step_index   : int,
  ) -> float:
  """
  Computes a weight bound between 0 and 1 for the decreasing contribution of a pixel based on its distance along a streamline.
  """
  return 0.5 * (1 + numpy.cos(numpy.pi * step_index / streamlength))

def interpolate_bilinear(
    vfield : numpy.ndarray,
    row    : float,
    col    : float,
  ) -> tuple[float, float]:
  """
  Bilinear interpolation on the vector field at a non-integer position (row, col).
  """
  row_low = int(numpy.floor(row))
  col_low = int(numpy.floor(col))
  row_high = min(row_low + 1, vfield.shape[1] - 1)
  col_high = min(col_low + 1, vfield.shape[2] - 1)
  ## weight based on distance from the pixel edge
  weight_row_high = row - row_low
  weight_col_high = col - col_low
  weight_row_low = 1 - weight_row_high
  weight_col_low = 1 - weight_col_high
  interpolated_vfield_comp_col = (
      vfield[0, row_low, col_low]   * weight_row_low  * weight_col_low
    + vfield[0, row_low, col_high]  * weight_row_low  * weight_col_high
    + vfield[0, row_high, col_low]  * weight_row_high * weight_col_low
    + vfield[0, row_high, col_high] * weight_row_high * weight_col_high
  )
  interpolated_vfield_comp_row = (
      vfield[1, row_low, col_low]   * weight_row_low  * weight_col_low
    + vfield[1, row_low, col_high]  * weight_row_low  * weight_col_high
    + vfield[1, row_high, col_low]  * weight_row_high * weight_col_low
    + vfield[1, row_high, col_high] * weight_row_high * weight_col_high
  )
  ## remember (x,y) -> (col, row)
  return interpolated_vfield_comp_col, interpolated_vfield_comp_row

def advect_streamline(
    vfield           : numpy.ndarray,
    sfield_in        : numpy.ndarray,
    start_row        : int,
    start_col        : int,
    dir_sgn          : int,
    streamlength     : int,
    use_periodic_BCs : bool,
  ) -> tuple[float, float]:
  """
  Computes the intensity of a given pixel (start_row, start_col) by summing the weighted contributions of pixels along
  a streamline originating from that pixel, integrating along the vector field.
  """
  weighted_sum = 0.0
  total_weight = 0.0
  row_float, col_float = start_row, start_col
  num_rows, num_cols = vfield.shape[1], vfield.shape[2]
  for step in range(streamlength):
    row_int = int(numpy.floor(row_float))
    col_int = int(numpy.floor(col_float))
    # ## nearest neighbor interpolation
    # vfield_comp_col = dir_sgn * vfield[0, row_int, col_int]  # x
    # vfield_comp_row = dir_sgn * vfield[1, row_int, col_int]  # y
    ## bilinear interpolation (negligble performance hit compared to nearest neighbor)
    vfield_comp_col, vfield_comp_row = interpolate_bilinear(
      vfield = vfield,
      row    = row_float,
      col    = col_float,
    )
    vfield_comp_col *= dir_sgn
    vfield_comp_row *= dir_sgn
    ## skip if the field magnitude is zero: advection has halted
    if abs(vfield_comp_row) == 0.0 and abs(vfield_comp_col) == 0.0: break
    ## compute how long the streamline advects before it leaves the current cell region (divided by cell-centers)
    if   vfield_comp_row > 0.0: delta_time_row = (numpy.floor(row_float) + 1 - row_float) / vfield_comp_row
    elif vfield_comp_row < 0.0: delta_time_row = (numpy.ceil(row_float)  - 1 - row_float) / vfield_comp_row
    else:                       delta_time_row = numpy.inf
    if   vfield_comp_col > 0.0: delta_time_col = (numpy.floor(col_float) + 1 - col_float) / vfield_comp_col
    elif vfield_comp_col < 0.0: delta_time_col = (numpy.ceil(col_float)  - 1 - col_float) / vfield_comp_col
    else:                       delta_time_col = numpy.inf
    ## equivelant to a CFL condition
    time_step = min(delta_time_col, delta_time_row)
    ## advect the streamline to the next cell region
    col_float += vfield_comp_col * time_step
    row_float += vfield_comp_row * time_step
    if use_periodic_BCs:
      row_float = (row_float + num_rows) % num_rows
      col_float = (col_float + num_cols) % num_cols
    ## open boundaries: terminate if streamline leaves the domain
    elif not ((0 <= row_float < num_rows) and (0 <= col_float < num_cols)): break
    ## weight the contribution of the current pixel based on its distance from the start of the streamline
    contribution_weight = taper_pixel_contribution(streamlength, step)
    ## ensure indices are integers before accessing the array
    row_int = int(row_int)
    col_int = int(col_int)
    weighted_sum += contribution_weight * sfield_in[row_int, col_int]
    total_weight += contribution_weight
  return weighted_sum, total_weight


## END OF LIC IMPLEMENTATION
