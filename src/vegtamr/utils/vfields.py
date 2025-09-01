## { MODULE

## This file is part of the "LineIntegralConvolution" project.
## Copyright (c) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.

##
## === DEPENDENCIES ===
##

import numpy

##
## === EXAMPLE VECTOR FIELDS ===
##


def vfield_lotka_volterra(size: int) -> dict:
    bounds_rows = (-3, 11)
    bounds_cols = (-5, 10)
    coords_row = numpy.linspace(bounds_rows[0], bounds_rows[1], size)
    coords_col = numpy.linspace(bounds_cols[0], bounds_cols[1], size)
    mg_x, mg_y = numpy.meshgrid(coords_col, coords_row, indexing="xy")
    x_capacity = 8
    y_growth = 3
    y_decay = 2
    vcomp_rows = mg_x * (1 - mg_x / x_capacity) - mg_y * mg_x / (1 + mg_x)
    vcomp_cols = y_growth * mg_y * mg_x / (1 + mg_x) - y_decay * mg_y
    vfield = numpy.array([vcomp_rows, vcomp_cols])
    return {
        "name": "lotka_volterra",
        "vfield": vfield,
        "streamlength": size // 4,
        "num_rows": size,
        "num_cols": size,
        "bounds_rows": bounds_rows,
        "bounds_cols": bounds_cols,
    }


def vfield_circles(size: int) -> dict:
    bounds_rows = (-10, 10)
    bounds_cols = (-10, 10)
    coords_row = numpy.linspace(bounds_rows[0], bounds_rows[1], size)
    coords_col = numpy.linspace(bounds_cols[0], bounds_cols[1], size)
    mg_x, mg_y = numpy.meshgrid(coords_col, coords_row, indexing="xy")
    radius = numpy.hypot(mg_x, mg_y)
    vcomp_rows = numpy.where(
        radius > 2.5 * numpy.pi, numpy.cos(mg_y / numpy.pi), numpy.cos(mg_y * numpy.pi / 2)
    )
    vcomp_cols = numpy.where(
        radius > 2.5 * numpy.pi, numpy.cos(mg_x / numpy.pi), numpy.cos(mg_x * numpy.pi / 2)
    )
    vfield = numpy.array([vcomp_rows, vcomp_cols])
    return {
        "name": "circles",
        "vfield": vfield,
        "streamlength": size // 4,
        "num_rows": size,
        "num_cols": size,
        "bounds_rows": bounds_rows,
        "bounds_cols": bounds_cols,
    }


def vfield_swirls(
    size: int,
    num_swirls: float = 1,
) -> dict:
    bounds_rows = (-10, 10)
    bounds_cols = (-10, 10)
    coords_row = numpy.linspace(bounds_rows[0], bounds_rows[1], size)
    coords_col = numpy.linspace(bounds_cols[0], bounds_cols[1], size)
    mg_x, mg_y = numpy.meshgrid(coords_col, coords_row, indexing="xy")
    vcomp_rows = numpy.sin(num_swirls * (mg_y + mg_x) / (2 * numpy.pi))
    vcomp_cols = numpy.cos(num_swirls * (mg_x - mg_y) / (2 * numpy.pi))
    vfield = numpy.array([vcomp_rows, vcomp_cols])
    return {
        "name": "swirls",
        "vfield": vfield,
        "streamlength": size // (4 * num_swirls),
        "num_rows": size,
        "num_cols": size,
        "bounds_rows": bounds_rows,
        "bounds_cols": bounds_cols,
    }


def gen_random_field(size, correlation_length):
    ki_values = numpy.fft.fftfreq(size)
    kx, ky = numpy.meshgrid(*(ki_values for _ in range(2)), indexing="ij")
    k_magn = numpy.hypot(kx, ky)
    fft_filter = numpy.exp(-2.0 * numpy.square(k_magn * correlation_length))
    white_noise = numpy.random.normal(0.0, 1.0, (size, size))
    sfield_fft = fft_filter * numpy.fft.fftn(white_noise)
    return numpy.real(numpy.fft.ifftn(sfield_fft))


def vfield_squiggles(size: int) -> dict:
    correlation_length = size / 7
    vfield = numpy.array(
        [
            gen_random_field(size, correlation_length),
            gen_random_field(size, correlation_length),
        ]
    )
    return {
        "name": "squiggles",
        "vfield": vfield,
        "streamlength": correlation_length // 2,
        "num_rows": size,
        "num_cols": size,
        "bounds_rows": (-10, 10),
        "bounds_cols": (-10, 10),
    }


## } MODULE
