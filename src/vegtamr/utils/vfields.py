## { MODULE

## This file is part of the "vegtamr" project.
## Copyright (c) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.

##
## === DEPENDENCIES ===
##

import numpy

##
## === EXAMPLE VECTOR FIELDS ===
##


def vfield_lotka_volterra(num_cells: int) -> dict:
    bounds_rows = (-3, 11)
    bounds_cols = (-5, 10)
    coords_row = numpy.linspace(bounds_rows[0], bounds_rows[1], num_cells)
    coords_col = numpy.linspace(bounds_cols[0], bounds_cols[1], num_cells)
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
        "streamlength": num_cells // 4,
        "num_rows": num_cells,
        "num_cols": num_cells,
        "bounds_rows": bounds_rows,
        "bounds_cols": bounds_cols,
    }


def vfield_flowers(num_cells: int) -> dict:
    bounds_rows = (-10, 10)
    bounds_cols = (-10, 10)
    coords_row = numpy.linspace(bounds_rows[0], bounds_rows[1], num_cells)
    coords_col = numpy.linspace(bounds_cols[0], bounds_cols[1], num_cells)
    mg_x, mg_y = numpy.meshgrid(coords_col, coords_row, indexing="xy")
    vcomp_rows = numpy.cos(0.5 * mg_x)
    vcomp_cols = numpy.cos(0.5 * mg_y)
    vfield = numpy.array([vcomp_rows, vcomp_cols])
    return {
        "name": "flowers",
        "vfield": vfield,
        "streamlength": num_cells // 4,
        "num_rows": num_cells,
        "num_cols": num_cells,
        "bounds_rows": bounds_rows,
        "bounds_cols": bounds_cols,
    }


def vfield_swirls(
    num_cells: int,
    num_swirls: float = 1,
) -> dict:
    bounds_rows = (-10, 10)
    bounds_cols = (-10, 10)
    coords_row = numpy.linspace(bounds_rows[0], bounds_rows[1], num_cells)
    coords_col = numpy.linspace(bounds_cols[0], bounds_cols[1], num_cells)
    mg_x, mg_y = numpy.meshgrid(coords_col, coords_row, indexing="xy")
    vcomp_rows = numpy.sin(num_swirls * (mg_y + mg_x) / (2 * numpy.pi))
    vcomp_cols = numpy.cos(num_swirls * (mg_x - mg_y) / (2 * numpy.pi))
    vfield = numpy.array([vcomp_rows, vcomp_cols])
    return {
        "name": "swirls",
        "vfield": vfield,
        "streamlength": num_cells // (4 * num_swirls),
        "num_rows": num_cells,
        "num_cols": num_cells,
        "bounds_rows": bounds_rows,
        "bounds_cols": bounds_cols,
    }


def vfield_orszag_tang(num_cells: int) -> dict:
    bounds_rows = (0.0, 2 * numpy.pi)
    bounds_cols = (0.0, 2 * numpy.pi)
    y = numpy.linspace(bounds_rows[0], bounds_rows[1], num_cells)
    x = numpy.linspace(bounds_cols[0], bounds_cols[1], num_cells)
    mg_x, mg_y = numpy.meshgrid(x, y, indexing="xy")
    v_rows = -numpy.sin(mg_y)
    v_cols = numpy.sin(mg_x)
    vfield = numpy.array([v_rows, v_cols])
    return {
        "name": "orszag_tang",
        "vfield": vfield,
        "streamlength": num_cells // 4,
        "num_rows": num_cells,
        "num_cols": num_cells,
        "bounds_rows": bounds_rows,
        "bounds_cols": bounds_cols,
    }


## } MODULE
