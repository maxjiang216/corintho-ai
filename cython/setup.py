from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("move/move.pyx"),
    zip_safe=False,
)

setup(
    ext_modules=cythonize("board/board.pyx"),
    zip_safe=False,
)

setup(
    ext_modules=cythonize("game/game.pyx"),
    zip_safe=False,
)

setup(
    ext_modules=cythonize("trainmc/trainmc.pyx"),
    zip_safe=False,
)
