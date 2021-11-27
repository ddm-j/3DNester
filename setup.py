from setuptools import setup
from Cython.Build import cythonize
import numpy as np

utility = 'utility_cy'
example = 'example'

setup(
    ext_modules=cythonize(utility+".pyx",annotate=True),
    include_dirs=[np.get_include()]
)