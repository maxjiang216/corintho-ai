from distutils.extension import Extension

import numpy
from Cython.Build import cythonize
from setuptools import setup

# cython: language_level=3str

setup(
    ext_modules=cythonize(
        Extension(
            "play",
            [
                "play.pyx",
                "../cpp/node.cpp",
                "../cpp/game.cpp",
                "../cpp/move.cpp",
                "../cpp/util.cpp",
            ],
            extra_compile_args=[
                "-std=c++17",
            ],
            language="c++",
            include_dirs=[numpy.get_include()],
        ),
        nthreads=8,
    ),
    zip_safe=False,
)
