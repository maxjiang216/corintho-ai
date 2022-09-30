from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Move class',
    ext_modules=cythonize("move.pyx"),
    zip_safe=False,
)
