"""Rules-derived features and board symmetries for the architecture experiments.

  symmetries()       gather indices for the 8 board symmetries, from the
                     engine's own tables (corintho_ai/cpp/include/util.h)
  features(states)   legal-move masks and line spaces, computed by the
                     engine's rules (build/libcorintho_features.so, `make`)

A symmetric copy k of a position is x[..., idx[k]] for every array here:
states [70], policies and masks [96], per-space features [16].
"""
import ctypes
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LIB = os.path.join(os.path.dirname(HERE), "build", "libcorintho_features.so")
S, M = 70, 96


def _table(name):
    text = open(os.path.join(ROOT, "corintho_ai/cpp/include/util.h")).read()
    body = re.search(name + r"\[[^=]*=\s*\{(.*?)\};", text, re.S).group(1)
    rows = re.findall(r"\{([^{}]*)\}", body)
    return np.array(
        [[int(v) for v in r.split(",") if v.strip()] for r in rows]
    )


def symmetries():
    """(state [8, 70], move [8, 96], space [8, 16]) gather indices."""
    space = _table("space_symmetries")
    move = _table("move_symmetries")
    assert space.shape == (8, 16) and move.shape == (8, M)
    state = np.array(
        [
            [space[k][j // 4] * 4 + j % 4 for j in range(64)]
            + list(range(64, S))
            for k in range(8)
        ]
    )
    return state, move, space


def features(states, stride=1):
    """Masks and line spaces for states[::stride] (float32 [rows, 70]).

    Returns legal (bool [n, 96]) and lines (uint16 [n, 3], a bit per space,
    by top type: base, column, capital).
    """
    states = np.asarray(states)
    assert states.dtype == np.float32 and states.flags.c_contiguous
    n = (states.shape[0] + stride - 1) // stride
    lib = ctypes.CDLL(LIB)
    legal = np.zeros((n, 12), np.uint8)
    lines = np.zeros((n, 3), np.uint16)
    lib.corintho_features(
        states.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(n),
        ctypes.c_int64(stride),
        legal.ctypes.data_as(ctypes.c_void_p),
        lines.ctypes.data_as(ctypes.c_void_p),
    )
    return np.unpackbits(legal, axis=1, bitorder="little")[:, :M] > 0, lines


def space_bits(lines):
    """uint16 [n] bit masks -> bool [n, 16], space i at column i."""
    return (lines[:, None].astype(np.int64) >> np.arange(16)) & 1 > 0
