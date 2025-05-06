## This file is part of the "LineIntegralConvolution" project.
## Copyright (c) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import logging
from multiprocessing import Pool, shared_memory, cpu_count
from vegtamr.lic import _core

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CACHE_SIZE = 8 * 1024**2  # 8 MB per core
MIN_BLOCK_SIZE     = 64           # minimum viable block size for efficiency
ELEMENT_SIZE       = 4            # float32 size in bytes

def _calculate_block_size(array_shape, streamlength):
  """
  Determine optimal block size considering cache and streamlength.
  Returns None if row chunking is better.
  """
  try:
    max_dim = numpy.max(array_shape)
    max_block_padded = int(numpy.sqrt(DEFAULT_CACHE_SIZE / ELEMENT_SIZE))
    block_size = max_block_padded - 2 * streamlength
    block_size = numpy.max(block_size, MIN_BLOCK_SIZE)
    block_size = numpy.min(block_size, max_dim - 2 * streamlength)
    ## check if block size is still viable
    if (block_size < MIN_BLOCK_SIZE) or (streamlength > max_dim // 4):
      return None
    return block_size
  except Exception as error:
    logger.warning(f"Block size calculation failed: {error}")
    return None

def _generate_chunks(array_shape, streamlength):
  """Generate chunk descriptors with automatic strategy selection"""
  block_size = _calculate_block_size(array_shape, streamlength)
  if (block_size is None) or (array_shape[0] < 2*block_size):
    logger.info(f"Using row chunking (streamlength={streamlength})")
    return [
      ("row", row_index)
      for row_index in range(array_shape[0])
    ]
  logger.info(f"Using block chunking {block_size}x{block_size} (streamlength={streamlength})")
  chunks = []
  for row_index in range(0, array_shape[0], block_size):
    for col_index in range(0, array_shape[1], block_size):
      chunks.append(("block", row_index, col_index, block_size))
  return chunks

def _process_row(row_index, shm_vfield_name, vfield_shape, vfield_dtype,
        shm_sfield_name, sfield_shape, sfield_dtype,
        streamlength, use_periodic_BCs):
  """Process a single row"""
  try:
    shm_vfield = shared_memory.SharedMemory(name=shm_vfield_name)
    vfield     = numpy.ndarray(vfield_shape, dtype=vfield_dtype, buffer=shm_vfield.buf)
    shm_sfield = shared_memory.SharedMemory(name=shm_sfield_name)
    sfield_in  = numpy.ndarray(sfield_shape, dtype=sfield_dtype, buffer=shm_sfield.buf)
    num_cols = vfield_shape[1]
    row_results = numpy.zeros(num_cols, dtype=numpy.float32)
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
      row_results[col_index] = total_sum / total_weight if total_weight > 0 else 0.0
    return row_index, row_results
  finally:
    shm_vfield.close()
    shm_sfield.close()

def _process_block(
    block_row, block_col, block_size,
    shm_vfield_name, vfield_shape, vfield_dtype,
    shm_sfield_name, sfield_shape, sfield_dtype,
    streamlength, use_periodic_BCs
  ):
  """Process a block with padding"""
  try:
    shm_vfield = shared_memory.SharedMemory(name=shm_vfield_name)
    vfield     = numpy.ndarray(vfield_shape, dtype=vfield_dtype, buffer=shm_vfield.buf)
    shm_sfield = shared_memory.SharedMemory(name=shm_sfield_name)
    sfield_in  = numpy.ndarray(sfield_shape, dtype=sfield_dtype, buffer=shm_sfield.buf)
    pad = streamlength
    results = {}
    # Calculate padded region bounds
    row_start = numpy.max(0, block_row - pad)
    row_end   = numpy.min(vfield_shape[0], block_row + block_size + pad)
    col_start = numpy.max(0, block_col - pad)
    col_end   = numpy.min(vfield_shape[1], block_col + block_size + pad)
    # Extract padded subarray
    padded_vfield = vfield[row_start:row_end, col_start:col_end]
    padded_sfield = sfield_in[row_start:row_end, col_start:col_end]
    # Calculate local coordinates within block
    local_row_start = block_row - row_start
    local_col_start = block_col - col_start
    # Process core block (excluding padding)
    for row_index in range(block_row, min(block_row + block_size, vfield_shape[0])):
      for col_index in range(block_col, min(block_col + block_size, vfield_shape[1])):
        # Convert to local coordinates
        local_row_index = row_index - row_start
        local_col_index = col_index - col_start
        # Perform streamline calculation on padded subarray
        forward_sum, forward_total = _core.advect_streamline(
            vfield           = vfield,
            sfield_in        = sfield_in,
            start_row        = local_row_index,
            start_col        = local_col_index,
            dir_sgn          = +1,
            streamlength     = streamlength,
            use_periodic_BCs = use_periodic_BCs,
        )
        backward_sum, backward_total = _core.advect_streamline(
            vfield           = vfield,
            sfield_in        = sfield_in,
            start_row        = local_row_index,
            start_col        = local_col_index,
            dir_sgn          = -1,
            streamlength     = streamlength,
            use_periodic_BCs = use_periodic_BCs,
        )
        total_sum = forward_sum + backward_sum
        total_weight = forward_total + backward_total
        results[(row_index, col_index)] = total_sum / total_weight if total_weight > 0 else 0.0
    return results
  finally:
    shm_vfield.close()
    shm_sfield.close()

def compute_lic(vfield, sfield_in, sfield_out, streamlength, use_periodic_BCs):
  """
  Compute Line Integral Convolution with automatic parallelization strategy.
  
  Args:
    vfield: 2D numpy array of vector field
    sfield_in: 2D numpy input scalar field
    sfield_out: 2D numpy output scalar field (pre-allocated)
    streamlength: Maximum advection steps
    use_periodic_BCs: Boolean for boundary handling
    
  Returns:
    sfield_out: Computed scalar field
  """
  # Validate inputs
  assert vfield.ndim == 2, "vfield must be 2D"
  assert sfield_in.shape == vfield.shape, "Field shapes must match"
  # Setup shared memory
  shm_vfield = shared_memory.SharedMemory(create=True, size=vfield.nbytes)
  shm_vfield_arr = numpy.ndarray(vfield.shape, dtype=vfield.dtype, buffer=shm_vfield.buf)
  numpy.copyto(shm_vfield_arr, vfield)
  shm_sfield = shared_memory.SharedMemory(create=True, size=sfield_in.nbytes)
  shm_sfield_arr = numpy.ndarray(sfield_in.shape, dtype=sfield_in.dtype, buffer=shm_sfield.buf)
  numpy.copyto(shm_sfield_arr, sfield_in)
  try:
    # Generate chunking strategy
    chunks = _generate_chunks(vfield.shape, streamlength)
    with Pool(processes=cpu_count()) as pool:
      tasks = []
      for chunk in chunks:
        chunk_type = chunk[0]
        if chunk_type == "row":
          row_index = chunk[1]
          args = (
            row_index,
            shm_vfield.name, vfield.shape, vfield.dtype,
            shm_sfield.name, sfield_in.shape, sfield_in.dtype,
            streamlength, use_periodic_BCs
          )
          tasks.append((_process_row, args))
        elif chunk_type == "block":
          _, block_row, block_col, block_size = chunk
          args = (
            block_row, block_col, block_size,
            shm_vfield.name, vfield.shape, vfield.dtype,
            shm_sfield.name, sfield_in.shape, sfield_in.dtype,
            streamlength, use_periodic_BCs
          )
          tasks.append((_process_block, args))
      # Process tasks with load balancing
      chunk_size = max(1, len(tasks) // (cpu_count() * 4))
      results = []
      for func, args in tasks:
        results.append(pool.apply_async(func, args))
      # Collect results
      for async_result in results:
        result = async_result.get()
        if isinstance(result, tuple):  # Row result
          row_index, row_data = result
          sfield_out[row_index] = row_data
        elif isinstance(result, dict):  # Block result
          for (row_index, col_index), val in result.items():
            sfield_out[row_index, col_index] = val
  finally:
    # Cleanup shared memory
    shm_vfield.close()
    shm_vfield.unlink()
    shm_sfield.close()
    shm_sfield.unlink()
  return sfield_out
