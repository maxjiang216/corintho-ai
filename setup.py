from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

# cython: language_level=3str

setup(
    ext_modules=cythonize(
        Extension(
            "corintho",
            [
                "main.pyx",
                "cpp/selfplayer.cpp",
                "cpp/trainmc.cpp",
                "cpp/node.cpp",
                "cpp/game.cpp",
                "cpp/move.cpp",
            ],
            extra_compile_args=[
                "-O3",
                # "-Ofast",
                "-std=c++20",
                "-fopenmp",
                # "-g0",
                # "-fstrict-overflow",
            ],
            language="c++",
            include_dirs=[numpy.get_include()],
            extra_link_args=["-fopenmp"],
        ),
        nthreads=4,
    ),
    zip_safe=False,
)
