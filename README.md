# Line Integral Convolution (LIC)

LICs are an amazing way to visualise 2D vector fields, and are widely used in many different fields (e.g., weather modelling, plasma physics, etc.), however I couldn't find a simple, up-to-date implementation, so I wrote my own. I hope it can now also help you on your own vector field fueled journey!

Here is an example of the LIC code applied to two different vector fields:
- Left: modified Lotka-Volterra equations
- Right: Gaussian random vector field

<div style="display: flex; justify-content: space-between;">
  <!-- <img src="./gallery/lic_lotka_volterra.png" width="49%" /> -->
  <!-- <img src="./gallery/lic_gaussian_random.png" width="49%" /> -->
  <img src="https://raw.githubusercontent.com/AstroKriel/line-integral-convolutions/refs/heads/main/gallery/lic_lotka_volterra.png" width="49%" />
  <img src="https://raw.githubusercontent.com/AstroKriel/line-integral-convolutions/refs/heads/main/gallery/lic_gaussian_random.png" width="49%" />
</div>


## Getting setup

You can now install the LIC package directly from PyPI or clone the repository if you'd like to play around with the source code.

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

### Option 2: Clone the repository (for development)

#### 1. Clone the repository:

```bash
git clone git@github.com:AstroKriel/line-integral-convolutions.git
cd line-integral-convolutions
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
python3 examples/example_lic.py
```

## Acknowledgements

Special thanks to Dr. James Beattie ([@AstroJames](https://github.com/AstroJames)) for highlighting that iteration, high-pass filtering, and histogram normalisation improves the final result. Also, thanks to Dr. Philip Mocz ([@pmocz](https://github.com/pmocz)) for his helpful suggestions in restructuring and improving the codebase.

## File structure

```bash
line-integral-convolutions/            # Root (project) directory
├── src/
│   └── line_integral_convolutions/    # Python package
│       ├── __init__.py                # Initialization file for the package
│       ├── fields.py                  # Example vector fields
│       ├── lic.py                     # Core of Line Integral Convolution (LIC) package
│       ├── utils.py                   # Utility functions
│       └── visualization.py           # Code for plotting LIC
├── examples/
│   └── example_lic.py                 # An example script showing how to run LIC computations
├── gallary/
│   └── example high-resolution LICs
├── requirements.txt                   # Lists of dependencies required to run the project
├── setup.py                           # Script to install and package-up the project
├── LICENSE                            # Terms of use for the project
└── README.md                          # This file
```

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
