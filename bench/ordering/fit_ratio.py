import numpy as np
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    export_text,
)

exec(
    open(__file__.replace("fit_ratio", "fit_oracle"))
    .read()
    .split("def feats")[0]
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
]
X = feats(cols)
tr = ~test
lc = np.log1p(cost)
for dc in (1, 2, 3, 4):
    for dr in (1, 2, 3, 4):
        pc = (
            DecisionTreeClassifier(max_depth=dc, min_samples_leaf=100)
            .fit(X[tr], cut[tr])
            .predict_proba(X)[:, 1]
        )
        reg = DecisionTreeRegressor(max_depth=dr, min_samples_leaf=100).fit(
            X[tr], lc[tr]
        )
        score = np.log(pc + 1e-3) - reg.predict(X)
        oc, f = order_cost(score, test)
        og, fg = order_cost(-a["type"].astype(float) * 100 + score, test)
        print(
            f"  P(cut) tree d{dc}, cost tree d{dr}: {oc/base:.3f}x (first {f*100:4.1f}%)   within groups {og/base:.3f}x"
        )
# the reserve-only rule, as an explicit heuristic on top of the current order
rule = -a["order_pos"].astype(float) - 100 * (a["reserve_after"] <= 1)
oc, f = order_cost(rule, test)
print(
    f"  current order, but placing one of the last two pieces goes last: {oc/base:.3f}x"
)
rule = -a["order_pos"].astype(float) - 100 * (a["reserve_after"] == 0)
oc, f = order_cost(rule, test)
print(f"  current order, but placing the last piece goes last: {oc/base:.3f}x")
