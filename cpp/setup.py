from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        Extension(
            "main", ["main.pyx", "selfplayer.cpp", "trainmc.cpp", "node.cpp", "game.cpp", "move.cpp"],
            extra_compile_args=["-O3", "-std=c++20"],
            language="c++",
        )
    ),
    zip_safe=False,
)