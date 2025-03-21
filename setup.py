from setuptools import setup, find_packages

setup(
    name="Vegtamr",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[],
    python_requires=">=3.10",
)
