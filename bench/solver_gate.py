"""Correctness gate for solver changes (worklog 2026-09-25-nn-architectures,
entry 08).

    python bench/solver_gate.py BASELINE.tsv NEW.tsv

Both from solver_bench. Every position solved by both must have the same
result; the new solver may solve positions the baseline gave up on.
Exits 1 on any disagreement.
"""
import csv
import sys


def load(path):
    with open(path) as f:
        return {int(r["index"]): r for r in csv.DictReader(f, delimiter="\t")}


base, new = load(sys.argv[1]), load(sys.argv[2])
unknown = "2"
both = [
    i
    for i in base
    if i in new
    and base[i]["result"] != unknown
    and new[i]["result"] != unknown
]
bad = [i for i in both if base[i]["result"] != new[i]["result"]]
gained = sum(
    1
    for i in base
    if base[i]["result"] == unknown
    and i in new
    and new[i]["result"] != unknown
)
lost = sum(
    1
    for i in base
    if base[i]["result"] != unknown
    and i in new
    and new[i]["result"] == unknown
)
nodes = lambda d: sum(int(d[i]["nodes"]) for i in both)
print(
    f"compared {len(both)}, disagreements {len(bad)}, newly solved {gained}, "
    f"no longer solved {lost}; nodes on common solves {nodes(base)} -> "
    f"{nodes(new)} ({nodes(new) / max(1, nodes(base)):.3f}x)"
)
sys.exit(1 if bad else 0)
