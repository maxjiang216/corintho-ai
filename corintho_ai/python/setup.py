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
                "corintho",
                [
                    os.path.join(current_dir, "main.pyx"),
                    os.path.join(current_dir, "../cpp/src/selfplayer.cpp"),
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
                    # Link-time optimization. The one-line Node accessors
                    # (move_id, probability, num_legal_moves) are defined in
                    # node.cpp but called from trainmc.cpp, so without LTO they
                    # are cross-translation-unit calls that cannot inline.
                    # chooseNext calls all three per edge. Measured at 10.7% of
                    # all instructions; enabling LTO is worth 23% engine time.
                    # See bench/README.md and worklog entry 2026-09-20-03.
                    "-flto",
                ],
                language="c++",
                include_dirs=[
                    numpy.get_include(),
                    os.path.join(current_dir, "../cpp/include"),
                    os.path.join(current_dir, "../../gsl/include"),
                ],
                # -flto and -O3 must both be repeated at link time: with LTO
                # the optimization happens during linking, so passing the flag
                # only to the compiler yields none of the benefit.
                extra_link_args=["-fopenmp", "-flto", "-O3"],
            )
        ],
        nthreads=4,
    ),
    zip_safe=False,
)
