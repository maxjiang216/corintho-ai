#!/usr/bin/env python3
"""Summarize a pgo_experiment.sh run: pgo_analyze.py <out_dir>.

For each timing mode and each PGO arm, the paired ratio arm/base of engine
seconds (and wall seconds) over seeds: mean of the per-seed log ratios, with a
95% t confidence interval. Also checks that every seed played the same number
of turns in every arm, which bit-identical builds must.
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

# Two-sided 95% t quantiles by degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
       14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
       20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000}


def t95(df):
    keys = [k for k in sorted(T95) if k <= df]
    return T95[keys[-1]] if keys else float("nan")


def paired(base, arm):
    """Percent change arm vs base with a 95% CI, from per-seed log ratios."""
    seeds = sorted(set(base) & set(arm))
    logs = [math.log(arm[s] / base[s]) for s in seeds]
    n = len(logs)
    if n < 2:
        return n, float("nan"), float("nan"), float("nan")
    mean = sum(logs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in logs) / (n - 1))
    half = t95(n - 1) * sd / math.sqrt(n)
    pct = lambda x: (math.exp(x) - 1) * 100
    return n, pct(mean), pct(mean - half), pct(mean + half)


def main():
    out = Path(sys.argv[1])
    print(f"PGO experiment: {out}\n")
    for name in ("provenance.txt", "binary-sizes.txt", "inspect.txt", "gates.txt"):
        p = out / name
        if p.exists():
            print(f"== {name}")
            print(p.read_text().rstrip() + "\n")

    rows = defaultdict(dict)   # (mode, arm) -> seed -> row
    lines = (out / "timing.tsv").read_text().splitlines()
    header = lines[0].split("\t")
    for line in lines[1:]:
        r = dict(zip(header, line.split("\t")))
        rows[(r["mode"], r["arm"])][r["seed"]] = r

    print("== timing (arm vs base, paired by seed; negative = faster)")
    for mode in ("mt", "st"):
        base = rows.get((mode, "base"), {})
        if not base:
            continue
        label = {"mt": "real network, 20 threads",
                 "st": "stub, 1 thread pinned"}[mode]
        mean_base = sum(float(r["engine_s"]) for r in base.values()) / len(base)
        print(f"\n{mode} ({label}); base engine mean {mean_base:.3f} s")
        arms = sorted({a for (m, a) in rows if m == mode and a != "base"})
        for arm in arms:
            a = rows.get((mode, arm), {})
            for metric in ("engine_s", "wall_s"):
                b = {s: float(r[metric]) for s, r in base.items()}
                x = {s: float(r[metric]) for s, r in a.items()}
                n, m, lo, hi = paired(b, x)
                print(f"  {arm:8s} {metric:9s} n={n:2d}  {m:+6.2f}%  "
                      f"(95% CI {lo:+6.2f} to {hi:+6.2f})")
            mism = [s for s in a if s in base and a[s]["turns"] != base[s]["turns"]]
            print(f"  {arm:8s} turns identical to base on "
                  f"{len(set(a) & set(base)) - len(mism)} of "
                  f"{len(set(a) & set(base))} seeds"
                  + (f"; DIFFER on {mism}" if mism else ""))

    temps = out / "temps.tsv"
    if temps.exists():
        t = [l.split("\t") for l in temps.read_text().splitlines()[1:]]
        c = [int(x[1]) for x in t if x[1].isdigit()]
        if c:
            print(f"\n== package temperature: max {max(c)} C, "
                  f"mean {sum(c) / len(c):.0f} C over {len(c)} minutes")


if __name__ == "__main__":
    main()
