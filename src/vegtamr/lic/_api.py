## { MODULE

## This file is part of the "vegtamr" project.
## Copyright (c) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.

##
## === DEPENDENCIES
##

## third-party
import numpy
import rlic

##
## === PERFORM LIC ON ITS OWN
##


def _ensure_valid_lic_inputs(
    *,
    vfield: numpy.ndarray,
    sfield_in: numpy.ndarray | None,
    streamlength: int | None,
) -> None:
    if vfield.ndim != 3:
        raise ValueError(f"`vfield` must have 3 dimensions; got `{vfield.ndim}`.")
    num_vcomps, num_rows, num_cols = vfield.shape
    if num_vcomps != 2:
        raise ValueError(f"`vfield` must have 2 components (in the first dimension); got `{num_vcomps}`.")
    if (sfield_in is not None) and (sfield_in.shape != (num_rows, num_cols)):
        raise ValueError(
            f"`sfield_in` must have shape `({num_rows}, {num_cols})`; got `{sfield_in.shape}`."
        )
    if (streamlength is not None) and (not isinstance(streamlength, int)):
        raise TypeError(f"`streamlength` must be an int; got `{type(streamlength).__name__}`.")


def compute_lic(
    vfield: numpy.ndarray,
    sfield_in: numpy.ndarray | None = None,
    streamlength: int | None = None,
    *,
    seed_sfield: int = 42,
    use_periodic_BCs: bool = True,
    run_in_parallel: bool = True,
) -> numpy.ndarray:
    """
    Compute the Line Integral Convolution (LIC) for `vfield`.

    `streamlength` should be close to the correlation length of `vfield` for the best results; defaults to 1/4 of the smallest domain dimension.

    Parameters
    ---
    - `vfield`:
        3D array with shape `(2, num_rows, num_cols)`; the first axis holds the vector components. Provide a 2D slice for 3D fields.
    - `sfield_in`:
        2D scalar field with shape `(num_rows, num_cols)` to seed the LIC; a random field is generated if `None`.
    """
    from vegtamr.lic import _serial, _parallel_by_row
    _ensure_valid_lic_inputs(
        vfield=vfield,
        sfield_in=sfield_in,
        streamlength=streamlength,
    )
    num_vcomps, num_rows, num_cols = vfield.shape
    sfield_out = numpy.zeros((num_rows, num_cols), dtype=numpy.float32)
    if sfield_in is None:
        if seed_sfield is not None: numpy.random.seed(seed_sfield)
        sfield_in = numpy.random.rand(num_rows, num_cols).astype(numpy.float32)
    if streamlength is None: streamlength = int(min(num_rows, num_cols) // 4)
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
## === PERFORM LIC + POSTPROCESSING
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
    Compute LIC for `vfield` with optional iterative high-pass filtering and histogram equalisation.

    Supports a native Python backend (slower, more accurate) and a Rust-accelerated backend via `rLIC`
    (default; faster, less accurate): https://github.com/tlorach/rLIC

    Parameters
    ---
    - `vfield`:
        3D array with shape `(2, num_rows, num_cols)`; provide a 2D slice for 3D fields.
    - `sfield_in`:
        2D scalar field with shape `(num_rows, num_cols)` to seed the LIC; a random field is generated if `None`.
    - `backend`:
        `"rust"` or `"python"`; see above for the tradeoff.
    """
    from vegtamr.lic import _postprocess
    dtype = vfield.dtype
    shape = vfield.shape[1:]
    if sfield_in is None:
        if seed_sfield is not None: numpy.random.seed(seed_sfield)
        sfield_in = numpy.random.rand(*shape).astype(dtype)
    if streamlength is None: streamlength = int(min(shape) // 4)
    elif streamlength < 5: raise ValueError(f"`streamlength` must be at least 5 pixels; got `{streamlength}`.")
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
        sfield = rlic.convolve(
            sfield_in,  # pyright: ignore[reportArgumentType]
            vfield[0],
            vfield[1],
            kernel=kernel,
            boundaries="periodic" if use_periodic_BCs else "closed",
            iterations=num_lic_passes,
        )
        sfield /= numpy.max(numpy.abs(sfield))
        sfield_in = sfield
        if use_filter: sfield = _postprocess.filter_highpass(sfield, sigma=filter_sigma)
        if use_equalize: sfield = _postprocess.rescaled_equalize(sfield)
        return sfield
    else:
        raise ValueError(f"`backend` must be one of {{'python', 'rust'}}; got `{backend}`.")


## } MODULE
