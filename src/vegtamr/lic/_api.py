## { MODULE

## This file is part of the "LineIntegralConvolution" project.
## Copyright (c) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.

##
## === DEPENDENCIES ===
##

import rlic
import numpy
from vegtamr.lic import _serial, _parallel_by_row
from vegtamr.utils import _postprocess

##
## === PERFORM LIC ON ITS OWN ===
##


def compute_lic(
    vfield: numpy.ndarray,
    sfield_in: numpy.ndarray | None = None,
    streamlength: int | None = None,
    seed_sfield: int = 42,
    use_periodic_BCs: bool = True,
    run_in_parallel: bool = True,
) -> numpy.ndarray:
    """
  Computes the Line Integral Convolution (LIC) for a given vector field.

  This function generates a LIC image using the input vector field (`vfield`) and an optional background scalar field (`sfield_in`).
  If no scalar field is provided, a random scalar field is generated. If a background scalar field is provided, the LIC is computed on top of it.

  The `streamlength` parameter controls the length of the LIC streamlines. For better results, set it close to the correlation length of
  the vector field (often known a priori). If not specified, it defaults to 1/4 of the smallest domain dimension.

  Parameters:
  -----------
  vfield : numpy.ndarray
    3D array storing a 2D vector field with shape (num_vcomps=2, num_rows, num_cols).
    The first dimension holds the vector components (x,y), and the remaining two dimensions define the domain size.
    For 3D vector fields, provide a 2D slice.

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
    assert vfield.ndim == 3, f"`vfield` must have 3 dimensions, but got {vfield.ndim}."
    num_vcomps, num_rows, num_cols = vfield.shape
    assert num_vcomps == 2, f"`vfield` must have 2 components (in the first dimension), but got {num_vcomps}."
    sfield_out = numpy.zeros((num_rows, num_cols), dtype=numpy.float32)
    if sfield_in is None:
        if seed_sfield is not None: numpy.random.seed(seed_sfield)
        sfield_in = numpy.random.rand(num_rows, num_cols).astype(numpy.float32)
    else:
        assert sfield_in.shape == (num_rows, num_cols), (
            f"`sfield_in` must have dimensions ({num_rows}, {num_cols}), "
            f"but it has dimensions {sfield_in.shape}."
        )
    if streamlength is None: streamlength = int(min(num_rows, num_cols) // 4)
    assert isinstance(
        streamlength,
        int,
    ), print(f"Error: `streamlength = {streamlength}` is not an int.")
    if run_in_parallel:
        return _parallel_by_row.compute_lic(
            vfield=vfield,
            sfield_in=sfield_in,
            sfield_out=sfield_out,
            streamlength=streamlength,
            use_periodic_BCs=use_periodic_BCs,
        )
    else:
        return _serial.compute_lic(
            vfield=vfield,
            sfield_in=sfield_in,
            sfield_out=sfield_out,
            streamlength=streamlength,
            use_periodic_BCs=use_periodic_BCs,
        )


##
## === PERFORM LIC + POSTPROCESSING ===
##


def compute_lic_with_postprocessing(
    vfield: numpy.ndarray,
    sfield_in: numpy.ndarray | None = None,
    streamlength: int | None = None,
    *,
    seed_sfield: int = 42,
    use_periodic_BCs: bool = True,
    num_lic_passes: int = 2,
    use_filter: bool = True,
    filter_sigma: float = 3.0,
    use_equalize: bool = True,
    backend: str = "rust",
    run_in_parallel: bool = True,
    verbose: bool = True,
) -> numpy.ndarray:
    """
  Computes LIC with optional iterative post-processing, including high-pass filtering and histogram equalisation.
  
  This routine supports both a native Python backend (more accurate, but slower), and a Rust-accelerated backend
  (less accurate, but significantly faster). By default, the Rust backend is used for performance and is powered
  by the excellent `rLIC` implementation developed by @tlorach: https://github.com/tlorach/rLIC

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

  num_lic_passes : int, optional, default=2
    Number of LIC passes to perform.

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
    if streamlength is None: streamlength = int(min(shape) // 4)
    elif streamlength < 5: raise ValueError("`streamlength` should be at least 5 pixels.")
    sfield = numpy.array(sfield_in, copy=True)
    if backend.lower() == "python":
        if verbose:
            print(
                "Using the native `python` backend. This is slower but more accurate than to the `rust` backend.",
            )
        if not run_in_parallel:
            ## always print this hint
            print(
                "The serial Python backend is deprecated, but retained for completeness. "
                "Consider using the parallel backend (`run_in_parallel = True`) for better performance.",
            )
        for _ in range(num_lic_passes):
            sfield = compute_lic(
                vfield=vfield,
                sfield_in=sfield_in,
                streamlength=streamlength,
                seed_sfield=seed_sfield,
                use_periodic_BCs=False,
                run_in_parallel=run_in_parallel,
            )
            sfield_in = sfield
        if use_filter: sfield = _postprocess.filter_highpass(sfield, sigma=filter_sigma)
        if use_equalize: sfield = _postprocess.rescaled_equalize(sfield)
        return sfield
    elif backend.lower() == "rust":
        if verbose:
            print(
                "Using the `rust` backend. This is much faster but also less accurate than the `python` backend.",
            )
        kernel = 0.5 * (
            1 + numpy.cos(
                numpy.pi * numpy.arange(1 - streamlength, streamlength) / streamlength,
                dtype=dtype,
            )
        )
        sfield  = rlic.convolve(
          sfield_in, # type: ignore
          vfield[0],
          vfield[1],
          kernel     = kernel,
          boundaries = "periodic" if use_periodic_BCs else "closed",
          iterations = num_lic_passes,
        )
        sfield /= numpy.max(numpy.abs(sfield))
        sfield_in = sfield
        if use_filter: sfield = _postprocess.filter_highpass(sfield, sigma=filter_sigma)
        if use_equalize: sfield = _postprocess.rescaled_equalize(sfield)
        return sfield
    else:
        raise ValueError(f"Unsupported backend: `{backend}`.")


## } MODULE
