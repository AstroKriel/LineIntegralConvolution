## This file is part of the "line-integral-convolutions" project.
## Copyright (c) 2024 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.


## ###############################################################
## DEPENDENCIES
## ###############################################################
import sys
import matplotlib.pyplot as mplplot
from line_integral_convolutions import fields, lic, utils


## ###############################################################
## MAIN PROGRAM
## ###############################################################
@utils.time_func
def main(
    size: int,
    func_vfield,
    num_iterations: int = 1,
    num_repetitions: int = 1,
    bool_filter: bool = True,
    filter_sigma: float = 3.0,
    bool_equalize: bool = True,
    bool_debug: bool = False,
):
    print("Started running demo script...")
    dict_field = func_vfield(size)
    vfield = dict_field["vfield"]
    streamlength = dict_field["streamlength"]
    bounds_rows = dict_field["bounds_rows"]
    bounds_cols = dict_field["bounds_cols"]
    vfield_name = dict_field["name"]
    ## apply the LIC a few times: equivelant to painting over with a few brush strokes
    print("Computing LIC...")
    sfield = lic.compute_lic_with_postprocessing(
        vfield=vfield,
        streamlength=streamlength,
        num_iterations=num_iterations,
        num_repetitions=num_repetitions,
        bool_filter=bool_filter,
        filter_sigma=filter_sigma,
        bool_equalize=bool_equalize,
    )
    ## visualise the LIC
    print("Plotting data...")
    fig, _ = utils.plot_lic(
        sfield=sfield,
        vfield=vfield,
        bounds_rows=bounds_rows,
        bounds_cols=bounds_cols,
        bool_debug=bool_debug,
    )
    ## save and close the figure
    print("Saving figure...")
    fig_name = f"lic_{vfield_name}.png"
    fig.savefig(fig_name, dpi=300, bbox_inches="tight")
    mplplot.close(fig)
    print("Saved:", fig_name)


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
    main(
        size=500,
        func_vfield=fields.vfield_swirls,  # pass function reference (not a function call): without brackets
        num_iterations=3,
        num_repetitions=3,
        bool_filter=True,
        filter_sigma=3.0,  # lower values produce thinner LIC tubes
        bool_equalize=True,
        bool_debug=False,
    )
    sys.exit(0)


## END OF SCRIPT
