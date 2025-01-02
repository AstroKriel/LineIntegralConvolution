from setuptools import setup, find_packages

setup(
    name="line-integral-convolutions",
    author="Neco Kriel",
    author_email="neco.kriel@anu.edu.au",
    version="1.0.7",
    url="https://github.com/AstroKriel/line-integral-convolutions/tree/main",
    description="My implementation of line integral convolution (LIC).",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=open("requirements.txt").read().splitlines(),
)
