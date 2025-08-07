# Line Integral Convolution

Line Integral Convolutions (LICs) are an amazing way to visualise 2D vector fields, and are widely used in many different fields (e.g., weather modelling, plasma physics, etc.), however I couldn't find a simple, up-to-date implementation, so I wrote my own. I hope it can now also help you on your own vector field fueled journey!

Here is an example of the LIC code applied to two different vector fields:
- Left: modified Lotka-Volterra equations
- Right: Gaussian random vector field

<div style="display: flex; justify-content: space-between;">
  <!-- <img src="./gallery/lic_lotka_volterra.png" width="49%" /> -->
  <!-- <img src="./gallery/lic_gaussian_random.png" width="49%" /> -->
  <img src="https://raw.githubusercontent.com/AstroKriel/LineIntegralConvolution/refs/heads/main/gallery/lic_lotka_volterra.png" width="49%" />
  <img src="https://raw.githubusercontent.com/AstroKriel/LineIntegralConvolution/refs/heads/main/gallery/lic_gaussian_random.png" width="49%" />
</div>


## Getting setup

You can now install the LIC package directly from [PyPI](https://pypi.org/project/line-integral-convolutions/) or clone the [Github](https://github.com/AstroKriel/LineIntegralConvolutions/) repository if you'd like to play around with the source code.

### Option 1: Install from PyPI (for general use)

If you only need to use the package, you can install it via `pip`:

```bash
pip install line-integral-convolutions
```

After installing, import the main LIC implementation as follows:

```bash
from line_integral_convolutions import lic
```

Inside this module, you will want to use the `compute_lic_with_postprocessing` function. See its documentation for more details on how to get the most out of it.

### Option 2: Clone the GitHub repository (for development)

#### 1. Clone the repository:

```bash
git clone git@github.com:AstroKriel/LineIntegralConvolutions.git
cd LineIntegralConvolutions
```

#### 2. Set up a virtual environment (optional but recommended):

It is recommended to use a virtual environment to manage the project's dependencies. Before running any code or installing dependencies, activate the virtual environment via the following commands:

```bash
python3 -m venv venv
source venv/bin/activate # on Windows: venv\Scripts\activate
```

Once activated, you will install the dependencies and the LIC package inside this environment, keeping them isolated from the rest of your system.

When you are done working on or using the LIC code, deactivate the virtual environment by running:

```bash
deactivate
```

#### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

#### 4. Install the LIC package (optional, for using as a library):

To install the package locally for development or use in other Python scripts, run the following command:

```bash
pip install -e .
```

This will install the package in "editable" mode, allowing you to make changes to the code and have them reflected without needing to reinstall the package each time.

#### 5. Try out the demo-script

Run the demo script `examples/example_lic.py` which demonstrates how the LIC code can be applied to a vector field (the example file uses the Lotka-Volterra system). You can experiment by modifying the script or play around by adding your own vector fields!

```bash
cd examples
python3 example_lic.py
```

## Quick start

`compute_lic_with_postprocessing` handles all of the internal calls necessary to compute a LIC, and it includes optional postprocessing steps for filtering and intensity equalization. In practice, this is the only function you will need to call within this package. Here is an example of how to use it:


```python
import matplotlib.pyplot as plt
from line_integral_convolutions import lic
from line_integral_convolutions import fields, utils # for demo-ing

## generate a sample vector field
size         = 500
dict_field   = fields.vfield_swirls(size)
vfield       = dict_field["vfield"]
streamlength = dict_field["streamlength"]
bounds_rows  = dict_field["bounds_rows"]
bounds_cols  = dict_field["bounds_cols"]

## apply the LIC a few times: equivelant to painting over with a few brush strokes
sfield = lic.compute_lic_with_postprocessing(
    vfield          = vfield,
    streamlength    = streamlength,
    num_iterations  = 3,
    num_repetitions = 3,
    bool_filter     = True,
    filter_sigma    = 3.0,
    bool_equalize   = True,
)

utils.plot_lic(
    sfield      = sfield,
    vfield      = vfield,
    bounds_rows = bounds_rows,
    bounds_cols = bounds_cols,
)
plt.show()
```

## Acknowledgements

The fast (pre-compiled Rust) backend option, which this repo uses by default, was implemented by Dr. Clément Robert ([@neutrinoceros](https://github.com/neutrinoceros); see [rLIC](https://github.com/neutrinoceros/rLIC)). Special thanks also go to Dr. James Beattie ([@AstroJames](https://github.com/AstroJames)) for highlighting how iteration, high-pass filtering, and histogram normalisation improve the final result. Finally, Dr. Philip Mocz ([@pmocz](https://github.com/pmocz)) provided helpful suggestions in restructuring and improving the codebase.

## File structure

```bash
LineIntegralConvolutions/               # root (project) directory
├── src/
│   └── vegtamr/                        # package is named after Odin’s alias (translated to "Wanderer")
│       ├── __init__.py                 # package initialiser
│       ├── fields.py                   # example vector fields
│       ├── lic.py                      # core of the Line Integral Convolution (LIC) package
│       ├── utils.py                    # utility functions
│       └── visualization.py            # code for plotting lic
├── playground/
│   └── demo_script.py                  # An example script
├── gallery/
│   └── high-resolution images and gifs # example outputs
├── pyproject.toml                      # project metadata and dependencies
├── uv.lock                             # lock file (used by uv to pin dependencies)
├── LICENSE                             # terms of use and distribution for this project
├── MANIFEST.in                         # specifies which files to include when packaging the project
└── README.md                           # project overview, installation instructions, and usage examples
```

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
