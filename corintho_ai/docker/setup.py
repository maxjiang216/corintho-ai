import numpy
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

# cython: language_level=3str

setup(
    ext_modules=cythonize(
        [
            Extension(
                "play_corintho",
                [
                    "choose_move.pyx",
                    "../cpp/src/node.cpp",
                    "../cpp/src/game.cpp",
                    "../cpp/src/move.cpp",
                    "../cpp/src/util.cpp",
                ],
                extra_compile_args=[
                    "-O3",
                    "-std=c++17",
                    "-fopenmp",
                ],
                language="c++",
                include_dirs=[
                    numpy.get_include(),
                    "../cpp/include",
                    "../../gsl/include",
                ],
                extra_link_args=["-fopenmp"],
            )
        ],
        nthreads=4,
    ),
    zip_safe=False,
)
