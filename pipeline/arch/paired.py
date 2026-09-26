#!/usr/bin/env python3
# Worklog 2026-09-25-nn-architectures, entry 15.
"""arch/paired.py BASE_DIR TREAT_DIR [...]: game-by-game comparison of matches
played with the same seeds (pairs of dirs: base1 treat1 base2 treat2 ...).
A game counts as better/worse for the new side when its score changed."""
import math
import sys

d = sys.argv[1:]
better = worse = same = 0
for b, t in zip(d[0::2], d[1::2]):
    sb = [float(x) for x in open(f"{b}/game_scores.txt")]
    st = [float(x) for x in open(f"{t}/game_scores.txt")]
    assert len(sb) == len(st)
    for x, y in zip(sb, st):
        if y > x:
            better += 1
        elif y < x:
            worse += 1
        else:
            same += 1
n = better + worse
z = (better - worse) / math.sqrt(n) if n else 0
print(
    f"games {better + worse + same}: better {better} worse {worse} same {same}; sign test z {z:+.2f}"
)
