import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text

names = [
    "p",
    "type",
    "to",
    "from_",
    "sq_class",
    "dest_top",
    "height",
    "new_top",
    "near",
    "reserve_after",
    "hist_rank",
    "opp_replies",
    "opp_lines",
    "tried",
    "cut",
    "order_pos",
    "frozen_dist",
    "n_quiet",
]
dt = np.dtype(
    [("node", "<u4"), ("cost", "<u4"), ("hist", "<u4")]
    + [(n, "u1") for n in names]
    + [("pad", "u1", 2)]
)
a = np.fromfile("oracle.bin", dtype=dt)
node = a["node"].astype(np.int64)
cut = a["cut"] == 1
cost = a["cost"].astype(float)
# decisions where some candidate cuts
has = np.zeros(node.max() + 1, bool)
has[node[cut]] = True
keep = has[node]
a, node, cut, cost = a[keep], node[keep], cut[keep], cost[keep]
D = len(np.unique(node))
print(
    f"{len(a)} candidates in {D} decisions with a quiet cutter; share of candidates that cut {cut.mean()*100:.1f}%"
)
test = (node % 2) == 0


def order_cost(score, mask):
    nd = node[mask]
    sc = score[mask]
    c = cut[mask]
    co = cost[mask]
    pos = a["order_pos"][mask]
    k = np.lexsort((pos, -sc, nd))
    nd, c, co = nd[k], c[k], co[k]
    starts = np.r_[0, np.flatnonzero(np.diff(nd)) + 1]
    ends = np.r_[starts[1:], len(nd)]
    tot = 0.0
    first = 0
    for s_, e_ in zip(starts, ends):
        f = np.flatnonzero(c[s_:e_])[0]
        tot += co[s_ : s_ + f + 1].sum()
        first += f == 0
    return tot, first / len(starts)


base, bf = order_cost(-a["order_pos"].astype(float), test)
# oracle: cheapest cutter alone
ob = 0.0
for n_ in np.unique(node[test]):
    m = (node == n_) & cut
    ob += cost[m].min()
print(
    f"test decisions: current order {base/1e6:.2f}M nodes (cutter first {bf*100:.1f}%); oracle {ob/1e6:.2f}M ({ob/base:.2f}x)"
)


def feats(cols):
    return np.stack(
        [
            np.log1p(a[c].astype(float)) if c == "hist" else a[c].astype(float)
            for c in cols
        ],
        1,
    )


cols = [
    "type",
    "sq_class",
    "dest_top",
    "height",
    "new_top",
    "near",
    "reserve_after",
    "hist",
    "hist_rank",
    "frozen_dist",
    "p",
    "n_quiet",
    "opp_replies",
    "opp_lines",
]
cheapc = [c for c in cols if c not in ("opp_replies", "opp_lines")]
for cs, lab in ((cheapc, "cheap"), (cols, "cheap+opp")):
    X = feats(cs)
    tr = ~test
    for depth in (1, 2, 3, 4, 6):
        t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=100).fit(
            X[tr], cut[tr]
        )
        p = t.predict_proba(X)[:, 1]
        oc, f = order_cost(p, test)
        # keep the type groups
        og, fg = order_cost(-a["type"].astype(float) * 10 + p, test)
        print(
            f"  {lab:9s} tree d{depth}: {oc/base:.3f}x (first {f*100:4.1f}%)   within groups {og/base:.3f}x (first {fg*100:4.1f}%)"
        )
        if depth == 2 and lab == "cheap":
            print(export_text(t, feature_names=cs))
    lr = LogisticRegression(max_iter=500).fit(X[tr], cut[tr])
    p = lr.decision_function(X)
    oc, f = order_cost(p, test)
    og, fg = order_cost(-a["type"].astype(float) * 100 + p, test)
    print(
        f"  {lab:9s} logistic: {oc/base:.3f}x (first {f*100:4.1f}%)   within groups {og/base:.3f}x (first {fg*100:4.1f}%)"
    )
