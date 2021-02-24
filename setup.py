#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy
with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

with open("requirements.txt") as req_file:
    requirements = [req.strip() for req in req_file.readlines()]

setup_requirements = []

test_requirements = []

setup(
    author="Saket Choudhary",
    author_email="saketkc@gmail.com",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="A python package for fast fitting of Negative Binomial data",
    install_requires=requirements,
    ext_modules=cythonize(["pyfastnbfit/pyfastnbfit_cy.pyx", "pyfastnbfit/pyfastnbfit_py_cy.py"], compiler_directives={'language_level' : "3"}),
    include_dirs=[numpy.get_include()],
    license="BSD license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="pyfastnbfit",
    name="pyfastnbfit",
    packages=find_packages(include=["pyfastnbfit", "pyfastnbfit.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/saketkc/pyfastnbfit",
    version="0.1.0",
    zip_safe=False,
)
