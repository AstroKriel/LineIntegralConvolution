## This file is part of the "LineIntegralConvolution" project.
## Copyright (c) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy


## ###############################################################
## FUNCTIONS
## ###############################################################
def _estimate_L1_cache_capacity(num_values_per_cell):
  L1_bytes_capacity        = 32 * 1024  # L1-cache size (32 KB)
  float32_byte_size        = 4
  bytes_per_cell           = float32_byte_size * num_values_per_cell
  max_cells_per_L1_cache   = L1_bytes_capacity // bytes_per_cell
  max_cells_per_block_axis = int(numpy.sqrt(max_cells_per_L1_cache))
  return max_cells_per_block_axis

def _generate_block_ranges(num_cells_along_axis, streamlength, iter_cells_per_block_axis, use_periodic):
  iter_range = []
  data_range = []
  cell_start_index = 0
  while cell_start_index < num_cells_along_axis:
    iter_lo = cell_start_index
    iter_hi = min(iter_lo + iter_cells_per_block_axis, num_cells_along_axis)
    data_lo = iter_lo - streamlength
    data_hi = iter_hi + streamlength
    if not use_periodic: # TODO: check this (and the alternative) is correct
      data_lo = max(data_lo, 0)
      data_hi = min(data_hi, num_cells_along_axis)
    iter_range.append((iter_lo, iter_hi))
    data_range.append((data_lo, data_hi))
    cell_start_index += iter_cells_per_block_axis
  return iter_range, data_range

def _generate_blocks(num_rows, num_cols, streamlength, use_periodic=True):
  max_cells_per_block_axis  = _estimate_L1_cache_capacity(1)
  min_cells_per_block_axis  = 10 # still to determine. arbitrary for now.
  data_cells_per_block_axis = 2 * streamlength
  iter_cells_per_block_axis = max(min_cells_per_block_axis, max_cells_per_block_axis - data_cells_per_block_axis)
  iter_row_ranges, data_row_ranges = _generate_block_ranges(num_rows, streamlength, iter_cells_per_block_axis, use_periodic)
  iter_col_ranges, data_col_ranges = _generate_block_ranges(num_cols, streamlength, iter_cells_per_block_axis, use_periodic)
  iter_ranges = [
    (row_range[0], row_range[1], col_range[0], col_range[1])
    for row_range in iter_row_ranges
    for col_range in iter_col_ranges
  ]
  data_ranges = [
    (row_range[0], row_range[1], col_range[0], col_range[1])
    for row_range in data_row_ranges
    for col_range in data_col_ranges
  ]
  return {
    "iter_ranges": iter_ranges,
    "data_ranges": data_ranges
  }


## END OF MODULE