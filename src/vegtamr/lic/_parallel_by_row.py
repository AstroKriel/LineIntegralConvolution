## { MODULE

## This file is part of the "LineIntegralConvolution" project.
## Copyright (c) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.


##
## === DEPENDENCIES ===
##

import numpy
from vegtamr.lic import _core
from multiprocessing import Pool, shared_memory, cpu_count


##
## === LOOP THROUGH THE DOMAIN IN PARALLEL ===
##

def _process_row(
    row_index,
    shm_vfield_name, vfield_shape, vfield_dtype,
    shm_sfield_name, sfield_shape, sfield_dtype,
    streamlength, use_periodic_BCs
  ):
  shm_vfield = shared_memory.SharedMemory(name=shm_vfield_name)
  vfield     = numpy.ndarray(vfield_shape, dtype=vfield_dtype, buffer=shm_vfield.buf)
  shm_sfield = shared_memory.SharedMemory(name=shm_sfield_name)
  sfield_in  = numpy.ndarray(sfield_shape, dtype=sfield_dtype, buffer=shm_sfield.buf)
  _, num_cols_total = vfield_shape[1], vfield_shape[2]
  row_results = numpy.zeros(num_cols_total, dtype=numpy.float32)
  for col_index in range(num_cols_total):
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
    row_results[col_index] = total_sum / total_weight if total_weight > 0 else 0.0
  shm_vfield.close()
  shm_sfield.close()
  return row_index, row_results

def compute_lic(
  vfield           : numpy.ndarray,
  sfield_in        : numpy.ndarray,
  sfield_out       : numpy.ndarray,
  streamlength     : int,
  use_periodic_BCs : bool,
) -> numpy.ndarray:
  _, num_rows, _ = vfield.shape
  shm_vfield = shared_memory.SharedMemory(create=True, size=vfield.nbytes)
  shm_vfield_arr = numpy.ndarray(vfield.shape, dtype=vfield.dtype, buffer=shm_vfield.buf)
  numpy.copyto(shm_vfield_arr, vfield)
  shm_sfield = shared_memory.SharedMemory(create=True, size=sfield_in.nbytes)
  shm_sfield_arr = numpy.ndarray(sfield_in.shape, dtype=sfield_in.dtype, buffer=shm_sfield.buf)
  numpy.copyto(shm_sfield_arr, sfield_in)
  try:
    with Pool(processes=cpu_count()) as pool:
      args = [
        (
          row_index,
          shm_vfield.name, vfield.shape, vfield.dtype,
          shm_sfield.name, sfield_in.shape, sfield_in.dtype,
          streamlength, use_periodic_BCs
        )
        for row_index in range(num_rows)
      ]
      chunk_size = max(1, num_rows // (cpu_count() * 8))
      results = pool.starmap(_process_row, args, chunksize=chunk_size)
      for row_index, row_data in results:
        sfield_out[row_index] = row_data
  finally:
    ## ensure cleanup even if errors occur
    shm_vfield.close()
    shm_vfield.unlink()
    shm_sfield.close()
    shm_sfield.unlink()
  return sfield_out


## } MODULE