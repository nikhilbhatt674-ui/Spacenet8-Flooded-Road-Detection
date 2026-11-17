"""Qualitative figure: 2 typical + 2 failure test tiles (pre | post | GT | prediction) for one trained run.
Selection is automatic (median-F1 tiles and lowest-F1 tiles with >=500 GT flooded px) so cases are not cherry-picked.
usage: python make_figures.py --data data/processed --splits data/splits.csv --run runs/A/siam_full/seed0 --out figs/qualitative.pdf
"""
import argparse, json
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from models import FloodNet
from train import Tiles, VARIANTS, decide, predict_probs

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--splits", required=True); ap.add_argument("--run", required=True)
ap.add_argument("--out", default="figs/qualitative.pdf"); ap.add_argument("--protocol", default="A")
a = ap.parse_args()
m = json.load(open(f"{a.run}/metrics.json")); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mode, road, _ = VARIANTS[m["variant"]]
net = FloodNet(mode, road, pretrained=False).to(dev); net.load_state_dict(torch.load(f"{a.run}/best.pt", map_location=dev)); net.eval()
sp = pd.read_csv(a.splits); tids = sp[sp[a.protocol] == "test"].tid.tolist()
ds = Tiles(a.data, tids, m["norm_stats"], False); t = m["threshold"]; rows = []
with torch.no_grad():
    for i in range(len(ds)):
        x, y, seg, fl, hw, _ = ds[i]; H, W = hw.tolist()
        p = predict_probs(net, x[None].to(dev), y[None].to(dev), dev, False)[:, :, :H, :W]
        pred = decide(p, t)[0].cpu().numpy(); lab = np.concatenate([[0], 1 + fl.numpy()])[seg.numpy()[:H, :W]]
        tp = ((pred == 2) & (lab == 2)).sum(); f1 = 2 * tp / max((pred == 2).sum() + (lab == 2).sum(), 1)
        rows.append((tids[i], f1, int((lab == 2).sum()), pred, lab))
ok = sorted([r for r in rows if r[2] >= 500], key=lambda r: r[1])
pick = [("typical", ok[len(ok) // 2]), ("typical", ok[3 * len(ok) // 4]), ("failure", ok[0]), ("failure", ok[1])]
cm = ListedColormap(["black", "dodgerblue", "red"]); S = 4
fig, ax = plt.subplots(len(pick), 4, figsize=(7.0, 1.75 * len(pick)))
for r, (kind, (tid, f1, _, pred, lab)) in enumerate(pick):
    j = tids.index(tid); pre = np.load(f"{a.data}/{tid}_pre.npy", mmap_mode="r"); post = np.load(f"{a.data}/{tid}_post.npy", mmap_mode="r")
    for c, (img, ttl) in enumerate([(np.moveaxis(np.array(pre[:, ::S, ::S]), 0, -1), "pre"), (np.moveaxis(np.array(post[:, ::S, ::S]), 0, -1), "post"),
                                    (lab[::S, ::S], "ground truth"), (pred[::S, ::S], "prediction")]):
        ax[r, c].imshow(img, cmap=cm if c > 1 else None, vmin=0, vmax=2, interpolation="nearest"); ax[r, c].axis("off")
        if r == 0: ax[r, c].set_title(ttl, fontsize=7)
    ax[r, 0].text(0, -0.02, f"{kind} F1={f1:.2f}", transform=ax[r, 0].transAxes, fontsize=6, va="top")
plt.tight_layout(); import os; os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); plt.savefig(a.out, bbox_inches="tight")
print("saved", a.out, [(k, t_[0], round(float(t_[1]), 3)) for k, t_ in pick])
