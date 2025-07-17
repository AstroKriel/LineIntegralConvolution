## This file is part of the "LineIntegralConvolution" project.
## Copyright (c) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.

## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from multiprocessing import Pool, cpu_count
from vegtamr.lic import _core

## ###############################################################
## CACHE-AWARE BLOCK GENERATION (KEEP THIS)
## ###############################################################
def _estimate_L1_cache_capacity():
  L1_bytes_capacity = 32 * 1024  # 32 KB L1 cache
  float32_byte_size = 4
  bytes_per_cell = 4 * float32_byte_size  # 4 components per cell
  max_cells_per_L1_cache = L1_bytes_capacity // bytes_per_cell
  return int(numpy.sqrt(max_cells_per_L1_cache))

def _generate_block_ranges(num_cells_along_axis, streamlength, iter_cells_per_block_axis):
  iter_range = []
  data_range = []
  cell_start_index = 0
  while cell_start_index < num_cells_along_axis:
    iter_lo = cell_start_index
    iter_hi = min(iter_lo + iter_cells_per_block_axis, num_cells_along_axis)
    data_lo = iter_lo - streamlength
    data_hi = iter_hi + streamlength
    ## TODO: implement perioidc BCs
    data_lo = max(data_lo, 0)
    data_hi = min(data_hi, num_cells_along_axis)
    iter_range.append((iter_lo, iter_hi))
    data_range.append((data_lo, data_hi))
    cell_start_index += iter_cells_per_block_axis
  return iter_range, data_range

def _generate_blocks(num_rows, num_cols, streamlength):
  max_cells_per_block_axis  = _estimate_L1_cache_capacity()
  min_cells_per_block_axis  = max(10, 2 + 2*streamlength)
  iter_cells_per_block_axis = max(min_cells_per_block_axis, max_cells_per_block_axis - 2 * streamlength)
  iter_row_ranges, data_row_ranges = _generate_block_ranges(num_rows, streamlength, iter_cells_per_block_axis)
  iter_col_ranges, data_col_ranges = _generate_block_ranges(num_cols, streamlength, iter_cells_per_block_axis)
  return {
    "iter_ranges": [
      (row[0], row[1], col[0], col[1])
      for row in iter_row_ranges
      for col in iter_col_ranges
    ],
    "data_ranges": [
      (row[0], row[1], col[0], col[1])
      for row in data_row_ranges
      for col in data_col_ranges
    ]
  }

def _extract_blocks(data_ranges, vfield, sfield):
  vfield_blocks = []
  sfield_blocks = []
  for (row_lo, row_hi, col_lo, col_hi) in data_ranges:
    vfield_block = vfield[:, row_lo:row_hi, col_lo:col_hi]
    sfield_block = sfield[row_lo:row_hi, col_lo:col_hi]
    vfield_blocks.append(vfield_block)
    sfield_blocks.append(sfield_block)
  return {
    "vfield_blocks": vfield_blocks,
    "sfield_blocks": sfield_blocks
  }

