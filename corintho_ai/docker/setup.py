import os

import numpy
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

# cython: language_level=3str

current_dir = os.path.dirname(os.path.realpath(__file__))

setup(
    ext_modules=cythonize(
        [
            Extension(
                "play_corintho",
                [
                    os.path.join(current_dir, "choose_move.pyx"),
                    os.path.join(current_dir, "../cpp/src/trainmc.cpp"),
                    os.path.join(current_dir, "../cpp/src/node.cpp"),
                    os.path.join(current_dir, "../cpp/src/game.cpp"),
                    os.path.join(current_dir, "../cpp/src/move.cpp"),
                    os.path.join(current_dir, "../cpp/src/util.cpp"),
                ],
                extra_compile_args=[
                    "-O3",
                    "-std=c++17",
                    "-fopenmp",
                    "-DNDEBUG",
                ],
                language="c++",
                include_dirs=[
                    numpy.get_include(),
                    os.path.join(current_dir, "../cpp/include"),
                    os.path.join(current_dir, "../../gsl/include"),
                ],
                extra_link_args=["-fopenmp"],
            )
        ],
        nthreads=4,
    ),
    zip_safe=False,
)
