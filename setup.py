from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("utility_cy.pyx",annotate=True),
    include_dirs=[np.get_include()]
)