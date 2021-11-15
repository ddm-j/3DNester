from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("utility_cython.pyx",annotate=True),
    include_dirs=[np.get_include()]
)