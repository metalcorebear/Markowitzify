#!/usr/bin/env python
"""
@author: metalcorebear
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="markowitzify",
    version="0.0.5",
    author="metalcorebear",
    author_email="mark.mbailey@gmail.com",
    description="Markowitzify will implement a variety of portfolio and stock/cryptocurrency analysis methods to optimize portfolios or trading strategies.  The two primary classes are portfolio and stonks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/metalcorebear/Markowitzify",
    packages=setuptools.find_packages(),
    install_requires=['requests', 'pandas', 'numpy', 'sklearn', 'pandas_datareader', 'scipy', 'statsmodels'],
    py_modules=["markowitzify", "helper_monkey", "TSP_Reader"],
    package_data={},
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)