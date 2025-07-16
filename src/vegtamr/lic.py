## This file is part of the "Vegtamr" project.
## Copyright (c) 2024 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
import rlic
from . import utils


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

def _compute_lic(
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
      forward_sum, forward_total = advect_streamline(
        vfield           = vfield,
        sfield_in        = sfield_in,
        start_row        = row_index,
        start_col        = col_index,
        dir_sgn          = +1,
        streamlength     = streamlength,
        use_periodic_BCs = use_periodic_BCs,
      )
      backward_sum, backward_total = advect_streamline(
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

@utils.time_func
def compute_lic(
    vfield           : numpy.ndarray,
    sfield_in        : numpy.ndarray = None,
    streamlength     : int = None,
    seed_sfield      : int = 42,
    use_periodic_BCs : bool = True,
  ) -> numpy.ndarray:
  """
  Computes the Line Integral Convolution (LIC) for a given vector field.

  This function generates a LIC image using the input vector field (`vfield`) and an optional background scalar field (`sfield_in`).
  If no scalar field is provided, a random scalar field is generated, visualising the vector field on its own. If a background scalar
  field is provided, the LIC is computed over it.

  The `streamlength` parameter controls the length of the LIC streamlines. For best results, set it close to the correlation length of
  the vector field (often known a priori). If not specified, it defaults to 1/4 of the smallest domain dimension.

  Parameters:
  -----------
  vfield : numpy.ndarray
    3D array storing a 2D vector field with shape (num_vcomps=2, num_rows, num_cols). The first dimension holds the vector components (x,y),
    and the remaining two dimensions define the domain size. For 3D fields, provide a 2D slice.

  sfield_in : numpy.ndarray, optional, default=None
    2D scalar field to be used for the LIC. If None, a random scalar field is generated.

  streamlength : int, optional, default=None
    Length of LIC streamlines. If None, it defaults to 1/4 the smallest domain dimension.

  seed_sfield : int, optional, default=42
    The random seed for generating the scalar field.

  use_periodic_BCs : bool, optional, default=True
    If True, periodic boundary conditions are applied; otherwise, uses open boundary conditions.

  Returns:
  --------
  numpy.ndarray
    A 2D array storing the output LIC image with shape (num_rows, num_cols).
  """
  assert vfield.ndim == 3, f"vfield must have 3 dimensions, but got {vfield.ndim}."
  num_vcomps, num_rows, num_cols = vfield.shape
  assert num_vcomps == 2, f"vfield must have 2 components (in the first dimension), but got {num_vcomps}."
  sfield_out = numpy.zeros((num_rows, num_cols), dtype=numpy.float32)
  if sfield_in is None:
    if seed_sfield is not None: numpy.random.seed(seed_sfield)
    sfield_in = numpy.random.rand(num_rows, num_cols).astype(numpy.float32)
  else:
    assert sfield_in.shape == (num_rows, num_cols), (
      f"sfield_in must have dimensions ({num_rows}, {num_cols}), "
      f"but received it with dimensions {sfield_in.shape}."
    )
  if streamlength is None: streamlength = min(num_rows, num_cols) // 4
  return _compute_lic(
    vfield           = vfield,
    sfield_in        = sfield_in,
    sfield_out       = sfield_out,
    streamlength     = streamlength,
    num_rows         = num_rows,
    num_cols         = num_cols,
    use_periodic_BCs = use_periodic_BCs,
  )

def compute_lic_with_postprocessing(
    vfield                 : numpy.ndarray,
    sfield_in              : numpy.ndarray = None,
    streamlength           : int = None,
    seed_sfield            : int = 42,
    use_periodic_BCs       : bool = True,
    num_lic_passes         : int = 3,
    num_postprocess_cycles : int = 3,
    use_filter             : bool = True,
    filter_sigma           : float = 3.0,
    use_equalize           : bool = True,
    backend                : str = "rust",
  ) -> numpy.ndarray:
  """
  Iteratively compute a Line Integral Convolution (LIC) for a given vector field with optional post-processing steps,
  including filtering and intensity equalisation. This supports both a native Python backend and a pre-compiled, Rust-accelerated
  backend, which can be up to 100 times faster. The Rust backend is powered by `rLIC`, a minimal and optimised LIC implementation
  authored by @neutrinoceros (https://github.com/neutrinoceros/rLIC), and is used by default for performance.

  Parameters:
  -----------
  vfield : numpy.ndarray
    3D array storing a 2D vector field with shape (num_vcomps=2, num_rows, num_cols).
    For 3D fields, provide a 2D slice.

  sfield_in : numpy.ndarray, optional, default=None
    2D scalar field to be used for the LIC. If None, a random scalar field is generated.

  streamlength : int, optional, default=None
    Length of LIC streamlines. If None, defaults to 1/4 of the smallest domain dimension.

  seed_sfield : int, optional, default=42
    Random seed for generating the scalar field (only used if sfield_in is None).

  use_periodic_BCs : bool, optional, default=True
    If True, applies periodic boundary conditions; otherwise, uses open boundary conditions.

  num_lic_passes : int, optional, default=3
    Number of LIC passes to perform.

  num_postprocess_cycles : int, optional, default=3
    Number of full LIC + post-processing cycles to apply.

  use_filter : bool, optional, default=True
    If True, applies a high-pass filter after each LIC cycle.

  filter_sigma : float, optional, default=3.0
    Standard deviation for the Gaussian high-pass filter. Lower values produce finer structure.

  use_equalize : bool, optional, default=True
    If True, applies histogram equalisation at the end of the routine.

  backend : str, optional, default="rust"
    Selects the LIC backend implementation. Options are:
      - "rust": Use the fast Rust-based implementation via `rLIC`
      - "python": Use the slower, native Python implementation

  Returns:
  --------
  numpy.ndarray
    The post-processed LIC image.
  """
  dtype = vfield.dtype
  shape = vfield.shape[1:]
  if sfield_in is None:
    if seed_sfield is not None: numpy.random.seed(seed_sfield)
    sfield_in = numpy.random.rand(*shape).astype(dtype)
  if streamlength is None: streamlength = min(shape) // 4
  if backend.lower() == "python":
    print("Using the native `python` backend. This is slower compared to the `rust` backend, which can be up to 100x faster.")
    for _ in range(num_postprocess_cycles):
      for _ in range(num_lic_passes):
        sfield = compute_lic(
          vfield           = vfield,
          sfield_in        = sfield_in,
          streamlength     = streamlength,
          seed_sfield      = seed_sfield,
          use_periodic_BCs = use_periodic_BCs,
        )
        sfield_in = sfield
      if use_filter: sfield = utils.filter_highpass(sfield, sigma=filter_sigma)
    if use_equalize: sfield = utils.rescaled_equalize(sfield)
    return sfield
  elif backend.lower() == "rust":
    kernel = 0.5 * (1 + numpy.cos(numpy.pi * numpy.arange(1-streamlength, streamlength) / streamlength, dtype=dtype))
    for _ in range(num_postprocess_cycles):
      sfield  = rlic.convolve(
        sfield_in,
        vfield[0],
        vfield[1],
        kernel=kernel,
        boundaries="periodic" if use_periodic_BCs else "closed",
        iterations=num_lic_passes,
      )
      sfield /= numpy.max(numpy.abs(sfield))
      sfield_in = sfield
      if use_filter: sfield = utils.filter_highpass(sfield, sigma=filter_sigma)
    if use_equalize: sfield = utils.rescaled_equalize(sfield)
    return sfield
  else: raise ValueError(f"Unsupported backend: `{backend}`.")


## END OF LIC IMPLEMENTATION
