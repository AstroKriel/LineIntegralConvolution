# A library for generating Line Integral Convolutions

| Platform | Name |
|---|---|
| [GitHub](https://github.com/AstroKriel/LineIntegralConvolutions) | `LineIntegralConvolutions` |
| [PyPI](https://pypi.org/project/line-integral-convolutions/) | `line-integral-convolutions` |
| Python (import) | `vegtamr` (Odin's alias while wandering Hel) |

Line Integral Convolutions (LICs) are an amazing way to visualise 2D vector fields, and are widely used in many different fields (e.g., weather modelling, plasma physics, etc.), however I couldn't find a simple, up-to-date implementation, so I wrote my own. I hope it can now also help you on your own vector field fueled journey!

Here is the LIC code applied to a couple of example vector fields:
- Left: modified version of the Lotka-Volterra equations
- Right: a swirling pattern

<div style="display: flex; justify-content: space-between;">
  <img src="https://raw.githubusercontent.com/AstroKriel/LineIntegralConvolutions/refs/heads/main/gallery/lic_lotka_volterra.png" width="49%" />
  <img src="https://raw.githubusercontent.com/AstroKriel/LineIntegralConvolutions/refs/heads/main/gallery/lic_swirls.png" width="49%" />
</div>


## Getting setup

You can now install the LIC package directly from [PyPI](https://pypi.org/project/line-integral-convolutions/) or clone the [Github](https://github.com/AstroKriel/LineIntegralConvolutions/) repository if you'd like to play around with the source code.

### Option 1: Install from PyPI (for general use)

If you only need to use the package, you can install it via `pip`:

```bash
pip install line-integral-convolutions
```

After installing, import the library as follows:

```python
from vegtamr import lic
```

Inside this module, you will want to use the `lic.compute_lic_with_postprocessing` function. See below for details on how to get the most out of it.

### Option 2: Clone the GitHub repository (for development)

#### 1. Clone the repo:

```bash
git clone git@github.com:AstroKriel/LineIntegralConvolutions.git
cd LineIntegralConvolutions
```

#### 2. Create a development environment with uv:

```bash
uv sync
```

This will install dependencies listed in `pyproject.toml` into a virtual environment managed by `uv`.

With `uv` you get clean package management and reproducibility, where the only trade-off is a few extra keystrokes when running scripts:

```bash
uv run demos/demo-lic.py
```

A small price to pay for sanity! Alternatively, you can activate the environment with source `.venv/bin/activate` and run `python3 demos/demo-lic.py`.

#### 3. Use your local checkout from another project (optional):

`uv sync` (step 2) already gives you an editable install inside this repo's own `.venv`, so edits are picked up immediately when you work from here. If you want to use this edited clone in a different project, you will need to install it as an editable dependency from that project:

```bash
uv add --editable /path/to/vegtamr
```

## Quick start

`compute_lic_with_postprocessing` is the main entry point for generating LICs. It manages all the internal calls and offers optional postprocessing: filtering and intensity equalisation. In practice, this is the only function you’ll need to call!

Here’s a quick example:


```python
import matplotlib.pyplot as mpl_plot
from vegtamr import lic
from vegtamr.utils import vfields, plots

## generate a sample vector field
num_cells = 500
vfield_config = vfields.vfield_swirls(num_cells)
vfield = vfield_config.vfield
streamlength = vfield_config.streamlength

## apply the lic
sfield = lic.compute_lic_with_postprocessing(
    vfield=vfield,
    streamlength=streamlength,  # brush stroke length
    num_lic_passes=3,  # number of brush strokes
    use_filter=True,
    filter_sigma=5e-2 * num_cells,  # tube thickness
    use_equalize=True,
    backend="rust",
)

## and now plot!
fig, ax = mpl_plot.subplots()
plots.plot_lic(
    ax=ax,
    sfield=sfield,
    vfield=vfield,
    cmap_name="pink",
)
mpl_plot.show()
```

There are a number of parameters for you to experiment with; the effect of some choices is demonstrated by `demos/demo-params.py`, which produces the following image:

<img src="https://raw.githubusercontent.com/AstroKriel/LineIntegralConvolutions/refs/heads/main/gallery/effect_of_params.png" width="100%" />

In practice you will want to choose a `streamlength` close to the correlation length (in cells) of the structures you are trying to highlight. Depending on the effect you're aiming for, you can also play around with turning on the highpass filter (`use_filter`), changing its size (`filter_sigma`; controls the thickness of tubes), and turning on intensity equalization (`use_equalize`).

If you want a darker or lighter look without touching the underlying data, `plot_lic` also accepts `cmap_range`, e.g. `cmap_range=(0.0, 0.75)`; this restricts which portion of the colormap gets used, so the full value range still maps smoothly without any clipping.

## File structure

```text
LineIntegralConvolutions/  # project root
├── src/
│   └── vegtamr/  # package root (named after Odin's alias, "Wanderer")
│       ├── __init__.py
│       ├── py.typed  # marker for type checkers (PEP 561)
│       ├── lic/
│       │   ├── __init__.py
│       │   ├── _api.py  # public-facing API
│       │   ├── _core.py  # core algorithms
│       │   ├── _parallel_by_row.py  # parallel implementation
│       │   ├── _postprocess.py  # filtering + equalisation
│       │   └── _serial.py  # serial implementation
│       └── utils/
│           ├── __init__.py
│           ├── plots.py  # plotting helpers
│           └── vfields.py  # example vector fields
├── demos/  # example scripts
│   ├── demo-lic.py  # simple demo
│   └── demo-params.py  # demo of how parameters affect LIC output
├── gallery/  # reference images
├── pyproject.toml  # project metadata and dependencies
├── uv.lock  # lock file (used by uv to pin dependencies)
├── LICENSE  # terms of use and distribution
└── README.md  # this file
```

## Acknowledgements

The fast (pre-compiled Rust) backend option, which this repo uses by default, was implemented by Dr. Clément Robert ([@neutrinoceros](https://github.com/neutrinoceros); see [rLIC](https://github.com/neutrinoceros/rLIC)). Special thanks also go to Dr. James Beattie ([@AstroJames](https://github.com/AstroJames)) for highlighting how iteration, high-pass filtering, and histogram normalisation improve the final result. Finally, Dr. Philip Mocz ([@pmocz](https://github.com/pmocz)) provided lots of helpful suggestions in restructuring and improving the codebase.

## License

This project is licensed under the MIT License; see the [LICENSE](./LICENSE) file for details.
