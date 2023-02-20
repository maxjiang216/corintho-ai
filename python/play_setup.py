from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

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
