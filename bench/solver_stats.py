"""Move-ordering statistics from a SOLVER_STATS run (worklog
2026-09-25-nn-architectures, entry 10).

    SOLVER_STATS_OUT=stats.bin solver_bench POSITIONS.bin MAX_P CAP 1
    python bench/solver_stats.py stats.bin

One record per move tried (solver.cpp StatRecord). phase: 0 table move, 1
line-making, 2 quiet, 3 immediate win. For each feature: moves tried, share
that cut off, and where cutoffs were found in the order.
"""
import sys

import numpy as np

dt = np.dtype(
    [
        (n, "u1")
        for n in (
            "p",
            "n_moves",
            "phase",
            "k",
            "type",
            "frm",
            "to",
            "dest_occ",
            "height",
            "top",
            "replies",
            "dist",
            "cut",
        )
    ]
    + [("pad", "u1", 3), ("cost", "<u4")]
)
d = np.fromfile(sys.argv[1], dtype=dt)
cut = d["cut"] == 1
print(f"moves tried {len(d):,}; cutoffs {cut.sum():,}")
k = d["k"][cut]
print(
    "cutoffs found at order position 0 / 1 / 2 / 3+: "
    + " / ".join(f"{(k == i).mean():.3f}" for i in range(3))
    + f" / {(k >= 3).mean():.3f}"
)


def table(name, values, mask=None, labels=None):
    m = np.ones(len(d), bool) if mask is None else mask
    print(
        f"\n{name}: value | tried | cut rate | share of all cutoffs | "
        "mean cost (positions) | cutoffs per 1000 positions spent"
    )
    for v in np.unique(values[m]):
        sel = m & (values == v)
        lab = labels.get(int(v), int(v)) if labels else int(v)
        cost = d["cost"][sel].astype(np.float64)
        per = 1000 * cut[sel].sum() / max(1.0, cost.sum())
        print(
            f"  {str(lab):>12} | {sel.sum():>9,} | {cut[sel].mean():.3f} "
            f"| {(cut & sel).sum() / cut.sum():.3f} | {cost.mean():8.1f} "
            f"| {per:8.1f}"
        )


table(
    "phase", d["phase"], labels={0: "table", 1: "line", 2: "quiet", 3: "win"}
)
quiet = d["phase"] == 2
table(
    "quiet: move type",
    d["type"],
    quiet,
    {0: "stack move", 1: "base", 2: "column", 3: "capital"},
)
sq = lambda s: np.where(
    s == 255,
    3,
    np.where(
        ((s // 4) % 3 == 0) & ((s % 4) % 3 == 0),
        0,
        np.where(((s // 4) % 3 == 0) | ((s % 4) % 3 == 0), 1, 2),
    ),
)
lab_sq = {0: "corner", 1: "edge", 2: "centre", 3: "(place)"}
table("quiet: destination square", sq(d["to"]), quiet, lab_sq)
table(
    "quiet stack moves: source square",
    sq(d["frm"]),
    quiet & (d["type"] == 0),
    lab_sq,
)
table(
    "quiet: onto an existing stack",
    d["dest_occ"],
    quiet,
    {0: "empty", 1: "stack"},
)
table("quiet stack moves: height moved", d["height"], quiet & (d["type"] == 0))
table(
    "quiet stack moves: top piece",
    d["top"],
    quiet & (d["type"] == 0),
    {0: "base", 1: "column", 2: "capital"},
)
table("quiet: distance to the frozen square", d["dist"], quiet, {255: "none"})
table(
    "line-making: opponent replies",
    np.minimum(d["replies"], 6),
    d["phase"] == 1,
    {6: "6+"},
)
table("quiet: order position", np.minimum(d["k"], 10), quiet, {10: "10+"})
