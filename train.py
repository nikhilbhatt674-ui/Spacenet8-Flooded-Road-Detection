"""Train + evaluate one (protocol, variant, seed) run and write metrics.json."""
import argparse, json, os, random, time
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from models import FloodNet, total_loss

VARIANTS = {  # name: (mode, road prior, loss)
    "post_only":       ("post", False, "focal_dice"),
    "early_fusion":    ("early", False, "focal_dice"),
    "siam_full":       ("siam_diff", True, "focal_dice"),
    "siam_no_diff":    ("siam_nodiff", True, "focal_dice"),
    "siam_no_prior":   ("siam_diff", False, "focal_dice"),
    "siam_focal_only": ("siam_diff", True, "focal"),
    "siam_ce_only":    ("siam_diff", True, "ce"),
}
CFG = dict(crop=512, bs=8, lr=1e-4, wd=1e-2, epochs=50, patience=10, gamma=2.0, lam=1.0, mu=0.5, seg_frac=0.5)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False


class Tiles(Dataset):
    def __init__(self, root, tids, stats, train):
        self.root, self.tids, self.train = root, list(tids), train
        self.mean = np.array(stats["mean"], np.float32)[:, None, None]; self.std = np.array(stats["std"], np.float32)[:, None, None]
        self.pre = [np.load(f"{root}/{t}_pre.npy", mmap_mode="r") for t in self.tids]
        self.post = [np.load(f"{root}/{t}_post.npy", mmap_mode="r") for t in self.tids]
        self.seg = [np.load(f"{root}/{t}_seg.npy", mmap_mode="r") for t in self.tids]
        self.flags = [np.array(json.load(open(f"{root}/{t}_seg.json"))["flooded"], np.int64) for t in self.tids]

    def __len__(self):
        return 2 * len(self.tids) if self.train else len(self.tids)  # 2 random crops per tile per epoch

    def norm(self, x):
        return (x - self.mean) / self.std

    def __getitem__(self, i):
        i = i % len(self.tids); pre, post, seg = self.pre[i], self.post[i], self.seg[i]
        H, W = seg.shape; lut = np.concatenate([[0], 1 + self.flags[i]]).astype(np.int64)
        if self.train:
            c = CFG["crop"]; y, x = np.random.randint(0, H - c + 1), np.random.randint(0, W - c + 1)
            a, b, s = (np.array(v[..., y:y + c, x:x + c]) for v in (pre, post, seg))
            a, b = a.astype(np.float32) / 255, b.astype(np.float32) / 255
            k = np.random.randint(4); a, b, s = (np.rot90(v, k, axes=(-2, -1)) for v in (a, b, s))
            if np.random.rand() < 0.5: a, b, s = (np.flip(v, -1) for v in (a, b, s))
            if np.random.rand() < 0.5: a, b, s = (np.flip(v, -2) for v in (a, b, s))
            # per-date radiometric jitter (imitates sensor/illumination differences)
            a = np.clip(a * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05), 0, 1)
            b = np.clip(b * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05), 0, 1)
            lab = lut[np.ascontiguousarray(s)]
            return (torch.from_numpy(self.norm(np.ascontiguousarray(a)).astype(np.float32)),
                    torch.from_numpy(self.norm(np.ascontiguousarray(b)).astype(np.float32)),
                    torch.from_numpy(lab), torch.from_numpy((lab > 0).astype(np.float32)))
        a, b = np.array(pre, np.float32) / 255, np.array(post, np.float32) / 255
        ph, pw = (-H) % 32, (-W) % 32
        a, b = (np.pad(self.norm(v), ((0, 0), (0, ph), (0, pw)), mode="reflect") for v in (a, b))
        return (torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(np.array(seg, np.int64)),
                torch.from_numpy(self.flags[i]), torch.tensor([H, W]), i)


def train_stats(root, tids):
    px = []
    for t in tids:
        for k in ("pre", "post"):
            v = np.load(f"{root}/{t}_{k}.npy", mmap_mode="r")[:, ::8, ::8].astype(np.float32) / 255
            px.append(v.reshape(3, -1)[:, v.reshape(3, -1).max(0) > 0])
    px = np.concatenate(px, 1)
    return dict(mean=px.mean(1).tolist(), std=px.std(1).tolist())


def class_alpha(root, tids):
    cnt = np.zeros(3)
    for t in tids:
        seg = np.load(f"{root}/{t}_seg.npy", mmap_mode="r"); fl = np.array(json.load(open(f"{root}/{t}_seg.json"))["flooded"])
        lab = np.concatenate([[0], 1 + fl])[np.asarray(seg)]
        cnt += np.bincount(lab.ravel(), minlength=3)
    f = cnt / cnt.sum(); a = 1 / np.sqrt(f + 1e-8)
    return (a / a.mean()).tolist(), f.tolist()


def prf(conf):
    conf = np.asarray(conf, np.float64); tp = conf[2, 2]; fn = conf[2].sum() - tp; fp = conf[:, 2].sum() - tp
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    iou = [conf[c, c] / max(conf[c].sum() + conf[:, c].sum() - conf[c, c], 1) for c in range(3)]
    return dict(precision=P, recall=R, f1=2 * P * R / (P + R) if P + R else 0.0, iou=iou[2], miou=float(np.mean(iou)), iou_per_class=iou)


def seg_prf(c):  # c = [tp, fp, fn, tn] at segment level, flooded = positive
    tp, fp, fn, _ = c; P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    return dict(seg_precision=P, seg_recall=R, seg_f1=2 * P * R / (P + R) if P + R else 0.0)


@torch.no_grad()
def predict_probs(model, a, b, dev, amp):
    with torch.autocast(dev.type, enabled=amp):
        return torch.softmax(model(a, b)["flood"].float(), 1)


def decide(p, t):  # flooded if p_flood > t, otherwise best of {background, road}
    return torch.where(p[:, 2] > t, torch.full_like(p[:, 0], 2, dtype=torch.long), p[:, :2].argmax(1))


@torch.no_grad()
def evaluate(model, ds, thresholds, dev, amp, seg_level=False, frac=0.5):
    model.eval(); dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    regions = [ds.tids[i].split("__")[0] for i in range(len(ds))]
    acc = {r: {t: dict(conf=np.zeros((3, 3), np.int64), seg=np.zeros(4, np.int64)) for t in thresholds} for r in set(regions) | {"all"}}
    for a, b, seg, fl, hw, i in dl:
        H, W = hw[0].tolist(); reg = regions[int(i)]
        p = predict_probs(model, a.to(dev), b.to(dev), dev, amp)[:, :, :H, :W]
        lab = torch.cat([torch.zeros(1, dtype=torch.long), 1 + fl[0]])[seg[0][:H, :W]].to(dev)
        for t in thresholds:
            pred = decide(p, t)[0]
            cm = torch.bincount((lab * 3 + pred).flatten(), minlength=9).view(3, 3).cpu().numpy()
            for k in (reg, "all"): acc[k][t]["conf"] += cm
            if seg_level and fl.shape[1] > 0:
                sid = seg[0][:H, :W].to(dev).flatten(); n = fl.shape[1] + 1
                tot = torch.bincount(sid, minlength=n).float()
                hit = torch.bincount(sid, weights=(pred == 2).flatten().float(), minlength=n)
                ok = tot[1:] > 0; pf = ((hit[1:] / tot[1:].clamp(min=1)) >= frac)[ok].cpu().numpy(); gf = (fl[0].numpy() == 1)[ok.cpu().numpy()]
                s = np.array([(pf & gf).sum(), (pf & ~gf).sum(), (~pf & gf).sum(), (~pf & ~gf).sum()])
                for k in (reg, "all"): acc[k][t]["seg"] += s
    return acc


def profile(model, dev):
    n = sum(p.numel() for p in model.parameters()) / 1e6
    try:
        from torch.utils.flop_counter import FlopCounterMode
        x = torch.zeros(1, 3, 512, 512, device=dev); model.eval()
        with torch.no_grad(), FlopCounterMode(display=False) as fc: model(x, x)
        return n, fc.get_total_flops() / 1e9
    except Exception:
        return n, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--splits", required=True)
    ap.add_argument("--protocol", default="A", choices=["A", "B_G2L", "B_L2G"])
    ap.add_argument("--variant", required=True, choices=list(VARIANTS)); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--max_tiles", type=int, default=None, help="debug only")
    args = ap.parse_args()
    if args.epochs: CFG["epochs"] = args.epochs
    out = f"{args.out}/{args.protocol}/{args.variant}/seed{args.seed}"; os.makedirs(out, exist_ok=True)
    set_seed(args.seed); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = dev.type == "cuda"
    sp = pd.read_csv(args.splits); col = args.protocol
    tr, va, te = (sp[sp[col] == s].tid.tolist() for s in ("train", "val", "test"))
    if args.max_tiles: tr, va, te = tr[:args.max_tiles], va[:args.max_tiles], te[:args.max_tiles]
    stats = train_stats(args.data, tr); alpha_l, freq = class_alpha(args.data, tr)
    alpha = torch.tensor(alpha_l, dtype=torch.float32, device=dev)
    mode, road, loss_kind = VARIANTS[args.variant]
    model = FloodNet(mode, road, pretrained=not args.no_pretrained).to(dev)
    params, gflops = profile(model, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    g = torch.Generator(); g.manual_seed(args.seed)
    wif = lambda w: np.random.seed((torch.initial_seed() + w) % 2**32)
    dl = DataLoader(Tiles(args.data, tr, stats, True), batch_size=CFG["bs"], shuffle=True, num_workers=args.workers,
                    drop_last=True, generator=g, worker_init_fn=wif, persistent_workers=args.workers > 0)
    val_ds = Tiles(args.data, va, stats, False)
    best, best_ep, bad, t0, log = -1.0, -1, 0, time.time(), []
    for ep in range(CFG["epochs"]):
        model.train(); run = 0.0
        for a, b, y, ry in dl:
            a, b, y, ry = (v.to(dev, non_blocking=True) for v in (a, b, y, ry))
            with torch.autocast(dev.type, enabled=amp):
                o = model(a, b)
            loss = total_loss(o, y, ry, loss_kind, alpha, CFG["gamma"], CFG["lam"], CFG["mu"])
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); run += loss.item()
        sched.step()
        vf1 = prf(evaluate(model, val_ds, [0.5], dev, amp)["all"][0.5]["conf"])["f1"]
        log.append(dict(epoch=ep, train_loss=run / len(dl), val_f1=vf1)); print(log[-1], flush=True)
        if vf1 > best: best, best_ep, bad = vf1, ep, 0; torch.save(model.state_dict(), f"{out}/best.pt")
        else:
            bad += 1
            if bad >= CFG["patience"]: break
    train_seconds = time.time() - t0
    model.load_state_dict(torch.load(f"{out}/best.pt", map_location=dev))
    grid = [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)]
    vacc = evaluate(model, val_ds, grid, dev, amp)["all"]; vf = {t: prf(vacc[t]["conf"])["f1"] for t in grid}
    t_star = max(grid, key=lambda t: (vf[t], -abs(t - 0.5)))
    test_ds = Tiles(args.data, te, stats, False)
    res = {}
    for name, t in (("tuned", t_star), ("t0.5", 0.5)):
        acc = evaluate(model, test_ds, [t], dev, amp, seg_level=True, frac=CFG["seg_frac"])
        res[name] = {r: {**prf(v[t]["conf"]), **seg_prf(v[t]["seg"]), "n_gt_flooded_px": int(np.asarray(v[t]["conf"])[2].sum()),
                         "conf": np.asarray(v[t]["conf"]).tolist()} for r, v in acc.items()}
    json.dump(dict(protocol=args.protocol, variant=args.variant, seed=args.seed, threshold=t_star, val_f1_at_threshold=vf[t_star],
                   best_epoch=best_ep, epochs_run=len(log), train_seconds=train_seconds, params_M=params, gflops_512=gflops,
                   peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9 if amp else None, alpha=alpha_l, class_freq=freq,
                   n_train=len(tr), n_val=len(va), n_test=len(te), cfg=CFG, test=res, log=log, norm_stats=stats),
              open(f"{out}/metrics.json", "w"), indent=1)
    print("done", out, "val-tuned threshold", t_star, "test flooded F1", res["tuned"]["all"]["f1"])


if __name__ == "__main__":
    main()
