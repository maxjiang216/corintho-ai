"""Sum self costs from a sequence of callgrind dump files, by function and line.

  python callgrind_merge.py callgrind.out.1 callgrind.out.2 ... callgrind.out

Why: `callgrind_control -d` checkpoint dumps zero the counters, so a long run's
totals are spread over several files. Callgrind also compresses names across
them: a later file refers to "fn=(2076)" and relies on an earlier file for the
name. callgrind_annotate reads one file at a time, so it cannot resolve those.
Pass the files in dump order (the final file, without a suffix, last).

Only self (exclusive) costs are summed: a cost line directly after "calls=" is
the inclusive cost of that call and is skipped. Positions are line numbers
(callgrind's default); relative positions (+n, -n, *) are resolved.
"""

import re
import sys
from collections import defaultdict

NAME = re.compile(r"^\((\d+)\)(?: (.*))?$")


def resolve(table, value):
    m = NAME.match(value)
    if not m:
        return value
    ident, name = m.group(1), m.group(2)
    if name is not None:
        table[ident] = name
        return name
    return table.get(ident, f"<id {ident}>")


def main():
    files = sys.argv[1:]
    fn_names, file_names = {}, {}
    events = None
    by_fn = defaultdict(lambda: None)
    by_line = defaultdict(lambda: None)

    def add(table, key, costs):
        cur = table[key]
        if cur is None:
            table[key] = list(costs)
        else:
            for i, c in enumerate(costs):
                cur[i] += c

    for path in files:
        fl = fn = None
        last_pos = 0
        skip_next = False
        with open(path) as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line:
                    continue
                if line.startswith("events:"):
                    ev = line.split()[1:]
                    if events is None:
                        events = ev
                    elif ev != events:
                        sys.exit(f"{path}: events differ")
                    continue
                key, _, val = line.partition("=")
                if key in ("fl", "fi", "fe"):
                    fl = resolve(file_names, val)
                    continue
                if key == "fn":
                    fn = resolve(fn_names, val)
                    continue
                if key in ("cfl", "cfi"):
                    resolve(file_names, val)
                    continue
                if key == "cfn":
                    resolve(fn_names, val)
                    continue
                if key == "calls":
                    skip_next = True
                    continue
                if not (line[0].isdigit() or line[0] in "+-*"):
                    continue  # headers and other keys
                parts = line.split()
                pos = parts[0]
                if pos == "*":
                    p = last_pos
                elif pos[0] in "+-":
                    p = last_pos + int(pos)
                else:
                    p = int(pos)
                last_pos = p
                if skip_next:
                    skip_next = False
                    continue
                costs = [int(x) for x in parts[1:]]
                costs += [0] * (len(events) - len(costs))
                add(by_fn, fn, costs)
                add(by_line, (fl, p, fn), costs)

    total = [0] * len(events)
    for c in by_fn.values():
        for i, x in enumerate(c):
            total[i] += x
    show = [e for e in ("Ir", "D1mr", "DLmr", "Bcm") if e in events]
    idx = [events.index(e) for e in show]

    def row(c, label):
        cells = []
        for i in idx:
            pct = 100.0 * c[i] / total[i] if total[i] else 0.0
            cells.append(f"{c[i]:>15,} {pct:5.1f}%")
        return "  ".join(cells) + "  " + label

    print(f"# {len(files)} files: {' '.join(files)}")
    print("  ".join(f"{e:>22}" for e in show))
    print(row(total, "TOTAL"))
    print("\n## By function (self cost), top 40 by Ir")
    for fname, c in sorted(by_fn.items(), key=lambda kv: -kv[1][idx[0]])[:40]:
        print(row(c, fname))
    for e in show:
        j = events.index(e)
        print(f"\n## By source line (self cost), top 60 by {e}")
        for (fl, p, fname), c in sorted(by_line.items(), key=lambda kv: -kv[1][j])[:60]:
            print(row(c, f"{fl}:{p}  [{fname}]"))


if __name__ == "__main__":
    main()
