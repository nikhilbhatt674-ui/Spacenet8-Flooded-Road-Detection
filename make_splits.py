"""Geographic block split (Protocol A) and cross-region splits (Protocol B).

Protocol A: per region, tile centres are binned into square-ish blocks (3x3 tiles). Whole blocks are
  assigned to train/val/test (~70/15/15 by tile count). Tiles of a lower-priority split that lie within
  `buffer` tile-widths of a higher-priority split (test > val > train) are dropped.
Protocol B: train/val = train/val blocks of the source region (same block split as A),
  test = ALL tiles of the other region. B_G2L: Germany -> Louisiana-East, B_L2G the reverse.
The split is generated once with a fixed seed and is NOT varied across training seeds.
"""
import argparse, json
import numpy as np
import pandas as pd

RANK = {"test": 0, "val": 1, "train": 2}


def block_split(df, rng, frac=(0.70, 0.15, 0.15), cell_tiles=3.0, buffer=0.5):
    cw, ch = cell_tiles * df.w.median(), cell_tiles * df.h.median()
    df = df.copy()
    df["block"] = list(zip((df.cx // cw).astype(int), (df.cy // ch).astype(int)))
    blocks = df.groupby("block").size()
    order = list(blocks.index); rng.shuffle(order)
    n = len(df); target = dict(zip(("train", "val", "test"), (f * n for f in frac)))
    cur = dict(train=0, val=0, test=0); assign = {}
    for b in order:
        s = max(target, key=lambda k: (target[k] - cur[k]) / target[k])
        assign[b] = s; cur[s] += blocks[b]
    df["split"] = df.block.map(assign)
    # buffer: drop lower-priority tiles that are close to higher-priority ones
    x0, x1 = (df.cx - df.w / 2 - buffer * df.w).values, (df.cx + df.w / 2 + buffer * df.w).values
    y0, y1 = (df.cy - df.h / 2 - buffer * df.h).values, (df.cy + df.h / 2 + buffer * df.h).values
    sp = df.split.values; drop = np.zeros(n, bool)
    for i in range(n):
        for j in range(n):
            if RANK[sp[j]] < RANK[sp[i]] and x0[i] < x1[j] and x1[i] > x0[j] and y0[i] < y1[j] and y1[i] > y0[j]:
                drop[i] = True; break
    df.loc[drop, "split"] = "drop"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--split_seed", type=int, default=42)
    a = ap.parse_args()
    m = pd.read_csv(a.manifest); rng = np.random.RandomState(a.split_seed)
    parts = [block_split(m[m.region == r], rng) for r in sorted(m.region.unique())]
    s = pd.concat(parts)[["tid", "region", "split"]].rename(columns={"split": "A"})
    for name, src, dst in (("B_G2L", "Germany", "Louisiana-East"), ("B_L2G", "Louisiana-East", "Germany")):
        s[name] = np.where((s.region == src) & s.A.isin(["train", "val"]), s.A,
                           np.where(s.region == dst, "test", "drop"))
    s.to_csv(a.out, index=False)
    summ = {c: s.groupby(["region", c]).size().unstack(fill_value=0).to_dict("index") for c in ("A", "B_G2L", "B_L2G")}
    print(json.dumps(summ, indent=1)); json.dump(summ, open(a.out.replace(".csv", "_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
