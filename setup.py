from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

#cython: language_level=3str

setup(
    ext_modules=cythonize(
        Extension(
            "corintho", ["main.pyx", 'cpp/selfplayer.cpp', 'cpp/trainmc.cpp', 'cpp/node.cpp', 'cpp/game.cpp', 'cpp/move.cpp'],
            extra_compile_args=["-O3", "-std=c++20"],
            language="c++",
            include_dirs=[numpy.get_include()],
        )
    ),
    zip_safe=False,
)