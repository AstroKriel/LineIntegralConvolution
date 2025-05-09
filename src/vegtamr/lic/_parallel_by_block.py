## This file is part of the "LineIntegralConvolution" project.
## Copyright (local_col_index) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from multiprocessing import Pool, shared_memory, cpu_count
from vegtamr.lic import _core


## ###############################################################
## HELPER FUNCTIONS
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


## ###############################################################
## LIC IMPLEMENTATION
## ###############################################################
def _process_block(
    iter_block,
    data_block,
    shm_vfield_name, vfield_shape, vfield_dtype,
    shm_sfield_name, sfield_shape, sfield_dtype,
    streamlength, use_periodic_BCs
  ):
  shm_vfield = shared_memory.SharedMemory(name=shm_vfield_name)
  vfield     = numpy.ndarray(vfield_shape, dtype=vfield_dtype, buffer=shm_vfield.buf)
  shm_sfield = shared_memory.SharedMemory(name=shm_sfield_name)
  sfield_in  = numpy.ndarray(sfield_shape, dtype=sfield_dtype, buffer=shm_sfield.buf)
  iter_row_start, iter_row_stop, iter_col_start, iter_col_stop = iter_block
  data_row_start, data_row_stop, data_col_start, data_col_stop = data_block
  ## get views of the relevant domain (includes halo for streamlines)
  vfield_block = vfield[:, data_row_start:data_row_stop, data_col_start:data_col_stop]
  sfield_block = sfield_in[data_row_start:data_row_stop, data_col_start:data_col_stop]
  ## offsets relative to data_block
  row_offset = data_row_start
  col_offset = data_col_start
  result = numpy.zeros((iter_row_stop - iter_row_start, iter_col_stop - iter_col_start), dtype=numpy.float32)
  for local_row_index in range(iter_row_stop - iter_row_start):
    for local_col_index in range(iter_col_stop - iter_col_start):
      global_row_index = local_row_index + iter_row_start
      global_col_index = local_col_index + iter_col_start
      forward_sum, forward_total = _core.advect_streamline(
        vfield           = vfield,
        sfield_in        = sfield_in,
        start_row        = global_row_index,
        start_col        = global_col_index,
        dir_sgn          = +1,
        streamlength     = streamlength,
        use_periodic_BCs = use_periodic_BCs
      )
      backward_sum, backward_total = _core.advect_streamline(
        vfield           = vfield,
        sfield_in        = sfield_in,
        start_row        = global_row_index,
        start_col        = global_col_index,
        dir_sgn          = -1,
        streamlength     = streamlength,
        use_periodic_BCs = use_periodic_BCs
      )
      total_sum = forward_sum + backward_sum
      total_weight = forward_total + backward_total
      result[local_row_index, local_col_index] = total_sum / total_weight if total_weight > 0 else 0.0
  shm_vfield.close()
  shm_sfield.close()
  return iter_block, result


def compute_lic_blocked(
  vfield           : numpy.ndarray,
  sfield_in        : numpy.ndarray,
  sfield_out       : numpy.ndarray,
  streamlength     : int,
  use_periodic_BCs : bool,
) -> numpy.ndarray:
  num_rows, num_cols = sfield_in.shape
  shm_vfield = shared_memory.SharedMemory(create=True, size=vfield.nbytes)
  shm_vfield_arr = numpy.ndarray(vfield.shape, dtype=vfield.dtype, buffer=shm_vfield.buf)
  numpy.copyto(shm_vfield_arr, vfield)
  shm_sfield = shared_memory.SharedMemory(create=True, size=sfield_in.nbytes)
  shm_sfield_arr = numpy.ndarray(sfield_in.shape, dtype=sfield_in.dtype, buffer=shm_sfield.buf)
  numpy.copyto(shm_sfield_arr, sfield_in)
  block_info = _generate_blocks(num_rows, num_cols, streamlength, use_periodic=use_periodic_BCs)
  iter_ranges = block_info["iter_ranges"]
  data_ranges = block_info["data_ranges"]
  try:
    with Pool(processes=cpu_count()) as pool:
      args = [
        (
          iter_blk, data_blk,
          shm_vfield.name, vfield.shape, vfield.dtype,
          shm_sfield.name, sfield_in.shape, sfield_in.dtype,
          streamlength, use_periodic_BCs
        )
        for iter_blk, data_blk in zip(iter_ranges, data_ranges)
      ]
      chunk_size = max(1, len(args) // (cpu_count() * 4))
      results = pool.starmap(_process_block, args, chunksize=chunk_size)
    ## stitch results back into output array
    for iter_blk, block_data in results:
      r0, r1, c0, c1 = iter_blk
      sfield_out[r0:r1, c0:c1] = block_data
  finally:
    shm_vfield.close()
    shm_vfield.unlink()
    shm_sfield.close()
    shm_sfield.unlink()
  return sfield_out


## END OF MODULE