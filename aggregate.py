"""Aggregate runs/*/*/seed*/metrics.json -> mean +- std (n-1) tables, seed-matched differences, hypothesis rule."""
import glob, json, sys
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "runs"
R = {}
for f in glob.glob(f"{root}/*/*/seed*/metrics.json"):
    m = json.load(open(f)); R.setdefault((m["protocol"], m["variant"]), {})[m["seed"]] = m

KEYS = ["precision", "recall", "f1", "iou", "miou", "seg_f1"]


def vals(proto, var, key, region="all", which="tuned"):
    return {s: m["test"][which][region][key] for s, m in R.get((proto, var), {}).items() if region in m["test"][which]}


def ms(v):
    v = list(v.values())
    return (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else float("nan"), len(v)) if v else (float("nan"),) * 2 + (0,)


def fmt(t):
    return f"{100*t[0]:.1f} $\\pm$ {100*t[1]:.1f}" if t[2] > 1 else "n/a"


print("=== Table: flooded-road metrics, test blocks (Protocol A), % mean +- std over seeds ===")
for var in ["post_only", "early_fusion", "siam_full", "siam_no_diff", "siam_no_prior", "siam_focal_only", "siam_ce_only"]:
    row = [fmt(ms(vals("A", var, k))) for k in KEYS]
    n = ms(vals("A", var, "f1"))[2]
    print(f"{var:16s} (n={n}) & " + " & ".join(row) + " \\\\")

print("\n=== Cross-region (Protocol A Louisiana-East subset vs Protocol B_G2L), flooded F1 / recall ===")
for var in ["post_only", "early_fusion", "siam_full"]:
    a = vals("A", var, "f1", "Louisiana-East"); b = vals("B_G2L", var, "f1")
    ra = vals("B_G2L", var, "recall")
    drop = {s: a[s] - b[s] for s in a if s in b}
    print(f"{var:16s} F1_A(LA)={fmt(ms(a))}  F1_B={fmt(ms(b))}  drop(A-B)={fmt(ms(drop))}  recall_B={fmt(ms(ra))}")

print("\n=== Seed-matched differences (same seed index; not strictly paired) ===")
PAIRS = [("H1 pre-event helps (early - post)", "early_fusion", "post_only"),
         ("H1b pre-event helps (siamese - post)", "siam_full", "post_only"),
         ("H2 siamese diff vs early fusion", "siam_full", "early_fusion"),
         ("H2b difference term", "siam_full", "siam_no_diff"),
         ("H3 focal+Dice vs CE", "siam_full", "siam_ce_only"),
         ("H3b Dice term (vs focal only)", "siam_full", "siam_focal_only"),
         ("H4 road prior", "siam_full", "siam_no_prior")]


def verdict(d):
    d = np.array(list(d.values()))
    if len(d) < 3: return "insufficient seeds"
    if d.mean() <= 0: return "Not Supported"
    return "Supported" if (d > 0).all() and d.mean() > d.std(ddof=1) else "Partially Supported"


for name, x, y in PAIRS:
    for key in ("f1", "recall", "iou"):
        a, b = vals("A", x, key), vals("A", y, key); d = {s: a[s] - b[s] for s in a if s in b}
        if d:
            print(f"{name:42s} {key:7s} diff={100*np.mean(list(d.values())):+.2f} pp  "
                  f"std={100*np.std(list(d.values()), ddof=1) if len(d)>1 else float('nan'):.2f}  "
                  f"positive in {sum(v>0 for v in d.values())}/{len(d)} seeds -> {verdict(d)}")
print("\nRule (fixed before running): Supported = gap>0 in all 3 seeds AND mean gap > std of the gap; "
      "Partially = mean gap>0 otherwise; Not Supported = mean gap<=0. n=3 allows no significance test.")

print("\n=== Cost ===")
for var in ["post_only", "early_fusion", "siam_full"]:
    ms_ = R.get(("A", var), {})
    if ms_:
        m0 = next(iter(ms_.values()))
        print(f"{var:14s} params={m0['params_M']:.1f}M GFLOPs@512={m0['gflops_512']} train_min={np.mean([m['train_seconds'] for m in ms_.values()])/60:.1f}")
