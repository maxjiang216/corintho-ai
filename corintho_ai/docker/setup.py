from distutils.extension import Extension

import numpy
from Cython.Build import cythonize
from setuptools import setup

# cython: language_level=3str

setup(
    ext_modules=cythonize(
        Extension(
            "play_corintho",
            [
                "choose_move.pyx",
                "node.cpp",
                "game.cpp",
                "move.cpp",
                "util.cpp",
            ],
            extra_compile_args=[
                "-O3",
                "-std=c++17",
                "-fopenmp",
            ],
            language="c++",
            include_dirs=[numpy.get_include()],
            extra_link_args=["-fopenmp"],
        ),
        nthreads=4,
    ),
    zip_safe=False,
)
