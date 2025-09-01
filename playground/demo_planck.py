## { MODULE

## This file is part of the "LineIntegralConvolution" project.
## Copyright (c) 2025 Neco Kriel.
## Licensed under the MIT License. See LICENSE for details.


##
## === DEPENDENCIES ===
##

import sys
import time
import numpy
import matplotlib.pyplot as mpl_plot
from pathlib import Path
from astropy.io import fits
from vegtamr.lic import compute_lic_with_postprocessing
from vegtamr.utils import plots


##
## === MAIN PROGRAM ===
##

def main():
  script_dir = Path(__file__).parent
  log10_n_map_fits = fits.open(script_dir / "Taurusfwhm5_logNHmap.fits")
  q_map_fits = fits.open(script_dir / "Taurusfwhm10_Qmap.fits")
  u_map_fits = fits.open(script_dir / "Taurusfwhm10_Umap.fits")
  log10_n_map_data = log10_n_map_fits[0].data # type: ignore
  q_map_data = q_map_fits[0].data # type: ignore
  u_map_data = u_map_fits[0].data # type: ignore
  psi        = 0.5 * numpy.arctan2(-u_map_data, q_map_data)
  ex         = -numpy.sin(psi)
  ey         =  numpy.cos(psi)
  vfield     = numpy.array([ey, -ex])
  num_pixels = numpy.max(ey.shape)
  print("Computing LIC...")
  start_time = time.perf_counter()
  sfield = compute_lic_with_postprocessing(
    vfield         = vfield,
    streamlength   = int(num_pixels // 20),
    num_lic_passes = 3,
    filter_sigma   = 2e-2 * num_pixels, # approx width of LIC tubes
    backend        = "python",
  )
  elapsed_time = time.perf_counter() - start_time
  print(f"LIC execution took {elapsed_time:.3f} seconds.")
  print("Plotting data...")
  scaled_sfield = numpy.log10(numpy.abs(sfield * 10**log10_n_map_data))
  fig, ax = mpl_plot.subplots()
  im = plots.plot_lic(
    ax                  = ax,
    sfield              = scaled_sfield,
    vfield              = vfield,
    cmap_name           = "twilight_shifted",
    bounds_rows         = (0, 1),
    bounds_cols         = (0, 1),
    overlay_streamlines = False,
    streamline_colour   = "white",
    streamline_alpha    = 0.25,
  )
  im.set_clim(vmin=20, vmax=22)
  plots.add_cbar(
    ax         = ax,
    mappable   = im,
    side       = "top",
    label      = r"$\log_{10}(N_H / \mathrm{cm}^3)$",
    percentage = 0.05,
    fontsize   = 20,
  )
  print("Saving figure...")
  fig_path = script_dir / f"lic_planck.png"
  fig.savefig(fig_path, dpi=300, bbox_inches="tight")
  mpl_plot.close(fig)
  print("Saved:", fig_path)


##
## === ENTRY POINT ===
##

if __name__ == "__main__":
  main()
  sys.exit(0)


## } SCRIPT
