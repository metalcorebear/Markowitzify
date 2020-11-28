#!/usr/bin/env python
"""
@author: metalcorebear
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="markowitzify",
    version="0.0.3",
    author="metalcorebear",
    author_email="mark.mbailey@gmail.com",
    description="Markowitzify will implement portfolio optimization based on the theory described by Harry Markowitz (University of California, San Diego), and elaborated by Marcos M. Lopez de Prado (Cornell University).  Additionally, this repository simulates individual stock performance over time using Monte Carlo methods.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/metalcorebear/Markowitzify",
    packages=setuptools.find_packages(),
    install_requires=['requests', 'pandas', 'numpy', 'sklearn', 'pandas_datareader', 'scipy'],
    py_modules=["markowitzify", "helper_monkey"],
    package_data={},
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)