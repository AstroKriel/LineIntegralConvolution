## This file is part of the "line-integral-convolutions" project.
## Copyright (c) 2024 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from numba import njit, prange
from . import utils


## ###############################################################
## LIC IMPLEMENTATION
## ###############################################################
@njit
def taper_pixel_contribution(streamlength: int, step_index: int) -> float:
    """
    Computes a weight for the decreasing contribution of a pixel based on its distance along a streamline.

    Parameters:
    -----------
    streamlength : int
        Maximum length of a streamline.

    step_index : int
        Index of the current step along the streamline.

    Returns:
    --------
    float
        Weighting value bound between 0 and 1.
    """
    return 0.5 * (1 + numpy.cos(numpy.pi * step_index / streamlength))

@njit
def interpolate_bilinear(
    vfield: numpy.ndarray, row: float, col: float
) -> tuple[float, float]:
    """
    Bilinear interpolation on the vector field at a non-integer position (row, col).
    """
    row_low = int(numpy.floor(row))
    col_low = int(numpy.floor(col))
    row_high = min(row_low + 1, vfield.shape[1] - 1)
    col_high = min(col_low + 1, vfield.shape[2] - 1)
    ## weight based on distance from pixel edge
    weight_row_high = row - row_low
    weight_col_high = col - col_low
    weight_row_low = 1 - weight_row_high
    weight_col_low = 1 - weight_col_high
    interpolated_vfield_comp_col = (
        vfield[0, row_low, col_low] * weight_row_low * weight_col_low
        + vfield[0, row_low, col_high] * weight_row_low * weight_col_high
        + vfield[0, row_high, col_low] * weight_row_high * weight_col_low
        + vfield[0, row_high, col_high] * weight_row_high * weight_col_high
    )
    interpolated_vfield_comp_row = (
        vfield[1, row_low, col_low] * weight_row_low * weight_col_low
        + vfield[1, row_low, col_high] * weight_row_low * weight_col_high
        + vfield[1, row_high, col_low] * weight_row_high * weight_col_low
        + vfield[1, row_high, col_high] * weight_row_high * weight_col_high
    )
    ## remember (x,y) -> (col, row)
    return interpolated_vfield_comp_col, interpolated_vfield_comp_row

@njit
def advect_streamline(
    vfield: numpy.ndarray,
    sfield_in: numpy.ndarray,
    start_row: int,
    start_col: int,
    dir_sgn: int,
    streamlength: int,
    bool_periodic_BCs: bool,
) -> tuple[float, float]:
    """
    Computes the intensity of a pixel (start_row, start_col) by summing the weighted contributions of pixels along a streamline stemming from it.

    Returns:
    --------
    tuple
        A tuple containing:
        - weighted_sum : float
            The weighted sum of intensity values along the streamline.
        - total_weight : float
            The total weight accumulated from the taper function along the streamline.
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
            vfield=vfield,
            row=row_float,
            col=col_float,
        )
        vfield_comp_col *= dir_sgn
        vfield_comp_row *= dir_sgn
        ## skip if the field magnitude is zero: advection has halted
        if abs(vfield_comp_row) == 0.0 and abs(vfield_comp_col) == 0.0:
            break
        ## compute how long the streamline advects before it leaves the current cell region (divided by cell-centers)
        if vfield_comp_row > 0.0:
            delta_time_row = (numpy.floor(row_float) + 1 - row_float) / vfield_comp_row
        elif vfield_comp_row < 0.0:
            delta_time_row = (numpy.ceil(row_float) - 1 - row_float) / vfield_comp_row
        else:
            delta_time_row = numpy.inf
        if vfield_comp_col > 0.0:
            delta_time_col = (numpy.floor(col_float) + 1 - col_float) / vfield_comp_col
        elif vfield_comp_col < 0.0:
            delta_time_col = (numpy.ceil(col_float) - 1 - col_float) / vfield_comp_col
        else:
            delta_time_col = numpy.inf
        ## equivelant to a CFL condition
        time_step = min(delta_time_col, delta_time_row)
        ## advect the streamline to the next cell region
        col_float += vfield_comp_col * time_step
        row_float += vfield_comp_row * time_step
        if bool_periodic_BCs:
            row_float = (row_float + num_rows) % num_rows
            col_float = (col_float + num_cols) % num_cols
        else:
            ## open boundaries: terminate if streamline leaves the domain
            if not ((0 <= row_float < num_rows) and (0 <= col_float < num_cols)):
                break
        ## weight the contribution of the current pixel based on its distance from the start of the streamline
        contribution_weight = taper_pixel_contribution(streamlength, step)
        weighted_sum += contribution_weight * sfield_in[row_int, col_int]
        total_weight += contribution_weight
    return weighted_sum, total_weight

@njit(parallel=True)
def _compute_lic(
    vfield: numpy.ndarray,
    sfield_in: numpy.ndarray,
    sfield_out: numpy.ndarray,
    streamlength: int,
    num_rows: int,
    num_cols: int,
    bool_periodic_BCs: bool,
) -> numpy.ndarray:
    """
    Computes the Line Integral Convolution (LIC) over the entire domain by advecting streamlines from each pixel in both forward and backward directions along the vector field.
    """
    for row in prange(num_rows):
        for col in range(num_cols):
            forward_sum, forward_total = advect_streamline(
                vfield=vfield,
                sfield_in=sfield_in,
                start_row=row,
                start_col=col,
                dir_sgn=+1,
                streamlength=streamlength,
                bool_periodic_BCs=bool_periodic_BCs,
            )
            backward_sum, backward_total = advect_streamline(
                vfield=vfield,
                sfield_in=sfield_in,
                start_row=row,
                start_col=col,
                dir_sgn=-1,
                streamlength=streamlength,
                bool_periodic_BCs=bool_periodic_BCs,
            )
            total_sum = forward_sum + backward_sum
            total_weight = forward_total + backward_total
            if total_weight > 0.0:
                sfield_out[row, col] = total_sum / total_weight
            else:
                sfield_out[row, col] = 0.0
    return sfield_out

@utils.time_func
def compute_lic(
    vfield: numpy.ndarray,
    sfield_in: numpy.ndarray = None,
    streamlength: int = None,
    seed_sfield: int = 42,
    bool_periodic_BCs: bool = True,
) -> numpy.ndarray:
    """
    Computes the Line Integral Convolution (LIC) for a given vector field.

    This function generates a LIC image using the input vector field (`vfield`) and an optional background scalar field (`sfield_in`). If no scalar field is provided, a random scalar field is generated, visualising the vector field on its own. If a background scalar field is provided, the LIC is computed over it.

    The `streamlength` parameter controls the length of the LIC streamlines. For best results, set it close to the correlation length of the vector field (often known a priori). If not specified, it defaults to 1/4 of the smallest domain dimension.

    Parameters:
    -----------
    vfield : numpy.ndarray
        3D array storing a 2D vector field with shape (num_vcomps=2, num_rows, num_cols). The first dimension holds the vector components (x,y), and the remaining two dimensions define the domain size. For 3D fields, provide a 2D slice.

    sfield_in : numpy.ndarray, optional, default=None
        2D scalar field to be used for the LIC. If None, a random scalar field is generated.

    streamlength : int, optional, default=None
        Length of LIC streamlines. If None, it defaults to 1/4 the smallest domain dimension.

    seed_sfield : int, optional, default=42
        The random seed for generating the scalar field.

    bool_periodic_BCs : bool, optional, default=True
        If True, periodic boundary conditions are applied; otherwise, uses open boundary conditions.

    Returns:
    --------
    numpy.ndarray
        A 2D array storing the output LIC image with shape (num_rows, num_cols).
    """
    assert vfield.ndim == 3, f"vfield must have 3 dimensions, but got {vfield.ndim}."
    num_vcomps, num_rows, num_cols = vfield.shape
    assert (
        num_vcomps == 2
    ), f"vfield must have 2 components (in the first dimension), but got {num_vcomps}."
    sfield_out = numpy.zeros((num_rows, num_cols), dtype=numpy.float32)
    if sfield_in is None:
        if seed_sfield is not None:
            numpy.random.seed(seed_sfield)
        sfield_in = numpy.random.rand(num_rows, num_cols).astype(numpy.float32)
    else:
        assert sfield_in.shape == (num_rows, num_cols), (
            f"sfield_in must have dimensions ({num_rows}, {num_cols}), "
            f"but received it with dimensions {sfield_in.shape}."
        )
    if streamlength is None:
        streamlength = min(num_rows, num_cols) // 4
    return _compute_lic(
        vfield=vfield,
        sfield_in=sfield_in,
        sfield_out=sfield_out,
        streamlength=streamlength,
        num_rows=num_rows,
        num_cols=num_cols,
        bool_periodic_BCs=bool_periodic_BCs,
    )

def compute_lic_with_postprocessing(
    vfield: numpy.ndarray,
    sfield_in: numpy.ndarray = None,
    streamlength: int = None,
    seed_sfield: int = 42,
    bool_periodic_BCs: bool = True,
    num_iterations: int = 3,
    num_repetitions: int = 3,
    bool_filter: bool = True,
    filter_sigma: float = 3.0,
    bool_equalize: bool = True,
) -> numpy.ndarray:
    """
    Iteratively computes the Line Integral Convolutions (LICs) for a given vector field with optional postprocessing steps (i.e., filtering and intensity binning). See the `compute_lic` function for more details on the core LIC computation.

    Parameters:
    -----------
    vfield : numpy.ndarray
        3D array storing a 2D vector field with shape (num_vcomps=2, num_rows, num_cols). For 3D fields, provide a 2D slice.

    sfield_in : numpy.ndarray, optional, default=None
        2D scalar field to be used for the LIC. If None, a random scalar field is generated.

    streamlength : int, optional, default=None
        Length of LIC streamlines. If None, it defaults to 1/4 the smallest domain dimension.

    seed_sfield : int, optional, default=42
        Random seed for generating the scalar field.

    bool_periodic_BCs : bool, optional, default=True
        If True, periodic boundary conditions are applied; otherwise, uses open boundary conditions.

    num_iterations : int, optional, default=3
        Number of times to repeat the LIC computation.

    num_repetitions : int, optional, default=3
        Number of times to repeat the entire routine: LIC + highpass filter.

    bool_filter : bool, optional, default=True
        If True, applies a high-pass filter after the LIC computation.

    filter_sigma : float, optional, default=3.0
        The standard deviation of the intensity values to Gaussian filter. Lower values tend to produce thinner tubes.

    bool_equalize : bool, optional, default=True
        If True, applies an intensity binning equalization at the end of the routine.

    Returns:
    --------
    numpy.ndarray
        The post-processed LIC image.
    """
    for _ in range(num_repetitions):
        for _ in range(num_iterations):
            sfield = compute_lic(
                vfield=vfield,
                sfield_in=sfield_in,
                streamlength=streamlength,
                seed_sfield=seed_sfield,
                bool_periodic_BCs=bool_periodic_BCs,
            )
            sfield_in = sfield
        if bool_filter:
            sfield = utils.filter_highpass(sfield, sigma=filter_sigma)
    if bool_equalize:
        sfield = utils.rescaled_equalize(sfield)
    return sfield


## END OF LIC IMPLEMENTATION
