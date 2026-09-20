#!/usr/bin/env python3
"""Compare two benchmark records produced by run_suite.sh.

    ./compare.py results/before.tsv results/after.tsv

Prints a table suitable for pasting into a commit message.

Two rules are enforced, because they are the ones easy to get wrong by hand:

  * Digests are compared exactly. A changed digest on a run labelled as a pure
    optimization is a failure, not a curiosity.
  * Timing deltas are reported against the observed spread. A change smaller
    than the noise band is reported as "noise", never as a win.
"""

import sys


def load(path):
    meta, digests, metrics = {}, {}, {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                key, _, value = line[1:].partition("\t")
                meta[key] = value
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                digests[parts[0]] = parts[1]
            elif len(parts) == 4:
                metrics[parts[0]] = tuple(float(x) for x in parts[1:])
    return meta, digests, metrics


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    before_meta, before_dig, before_met = load(sys.argv[1])
    after_meta, after_dig, after_met = load(sys.argv[2])

    print(f"before: {before_meta.get('label','?')}  "
          f"(commit {before_meta.get('commit','?')}, "
          f"dirty={before_meta.get('dirty','?')})")
    print(f"after:  {after_meta.get('label','?')}  "
          f"(commit {after_meta.get('commit','?')}, "
          f"dirty={after_meta.get('dirty','?')})")
    print()

    # --- Digests ---
    failed = False
    print("Behaviour")
    for key in sorted(set(before_dig) | set(after_dig)):
        b = before_dig.get(key, "-")
        a = after_dig.get(key, "-")
        if b == a:
            print(f"  {key:<16} unchanged  {b}")
        else:
            failed = True
            print(f"  {key:<16} CHANGED    {b} -> {a}")
    if failed:
        print("\n  !! A digest changed. This run is not a pure optimization.")
        print("     Either it is a bug, or the behaviour change is intended")
        print("     and must be stated explicitly in the commit message.")
    print()

    # --- Exact counters first: deterministic, so any delta is real ---
    exact = [k for k in before_met
             if k in after_met
             and before_met[k][1] == before_met[k][2]
             and after_met[k][1] == after_met[k][2]]
    noisy = [k for k in before_met if k in after_met and k not in exact]

    if exact:
        print("Exact counters (deterministic to ~0.01% -- deltas above that are real)")
        header = f"  {'metric':<34} {'before':>14} {'after':>14} {'delta':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for key in exact:
            b = before_met[key][0]
            a = after_met[key][0]
            if b == 0:
                continue
            pct = (a - b) / b * 100.0
            # Not quite perfectly deterministic: instruction counts drift by a
            # few dozen out of ~1e9 between runs, presumably dynamic loader
            # noise. Anything under 0.01% is that, not a result.
            verdict = "same" if abs(pct) < 0.01 else f"{pct:+.2f}%"
            fmt = "{:>14.0f}" if b == int(b) and a == int(a) else "{:>14.4f}"
            print(f"  {key:<34} {fmt.format(b)} {fmt.format(a)} {verdict:>10}")
        print()

    # --- Timings ---
    print("Timings (median, with min/max across reps)")
    header = f"  {'metric':<34} {'before':>11} {'after':>11} {'delta':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in noisy:
        b_med, b_min, b_max = before_met[key]
        a_med, a_min, a_max = after_met[key]
        if b_med == 0:
            continue
        pct = (a_med - b_med) / b_med * 100.0
        # Noise band: half the summed relative spread of the two samples.
        spread = ((b_max - b_min) / b_med + (a_max - a_min) / a_med) * 50.0
        if abs(pct) <= max(spread, 1.0):
            verdict = "noise"
        else:
            verdict = f"{pct:+.1f}%"
        print(f"  {key:<34} {b_med:>11.1f} {a_med:>11.1f} {verdict:>10}")
    print()
    print("  Lower is better for ns_* and *_seconds; higher is better for")
    print("  requests_per_engine_second. 'noise' means the change is within the")
    print("  observed run-to-run spread and should not be claimed as a result.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
