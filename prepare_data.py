"""Convert raw SpaceNet 8 training areas into aligned pre/post/segment arrays.

For every tile pair we store (all on the PRE-event pixel grid):
  <tid>_pre.npy   uint8 (3,H,W)
  <tid>_post.npy  uint8 (3,H,W)   post image re-projected onto the pre grid
  <tid>_seg.npy   int16 (H,W)     road-segment id raster (0 = no road)
  <tid>_seg.json  {"flooded":[0/1,...]}  flooded flag per segment id (1..N)
and one manifest.csv row with the tile centre / extent in lon-lat (used for splits).

The three-class label is derived on the fly: 0 background, 1 non-flooded road, 2 flooded road.
NOTE: the CSV column names and the 'flooded' property name are matched fuzzily because they
could not be checked against the raw files when this script was written. The script prints
what it found; inspect that output before training.
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from pyproj import Transformer

TRUE_VALUES = {"true", "yes", "1", "flooded", "t", "y"}


def index_files(region_dir):
    idx = {}
    for dp, _, fns in os.walk(region_dir):
        for fn in fns:
            idx.setdefault(fn, os.path.join(dp, fn))
    return idx


def find_csv(region_dir):
    csvs = [str(p) for p in Path(region_dir).rglob("*.csv")]
    pref = [c for c in csvs if "mapping" in c.lower() or "reference" in c.lower()]
    if not (pref or csvs):
        raise FileNotFoundError(f"no csv found under {region_dir}")
    return (pref or csvs)[0]


def pick(df, *needles, exclude=()):
    for c in df.columns:
        lc = c.lower()
        if all(n in lc for n in needles) and not any(e in lc for e in exclude):
            return c
    return None


def segments_to_raster(road_path, src_crs, transform, H, W, width_px):
    """Rasterise road lines (lon/lat GeoJSON) to an id raster on the pre grid."""
    if road_path is None or not os.path.exists(road_path):
        return None, None, 0
    with open(road_path) as f:
        feats = json.load(f).get("features", [])
    tr = None
    if src_crs is not None and src_crs.to_string() not in ("EPSG:4326",):
        tr = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
    inv = ~transform
    a, b, c, d, e, f_ = inv[:6]

    def to_pix(x, y, z=None):
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        X, Y = tr.transform(x, y) if tr is not None else (x, y)
        return tuple((a * X + b * Y + c).tolist()), tuple((d * X + e * Y + f_).tolist())

    geoms, flags, missing = [], [], 0
    for ft in feats:
        props = {k.lower(): v for k, v in (ft.get("properties") or {}).items()}
        if "flooded" in props:
            fl = str(props["flooded"]).strip().lower() in TRUE_VALUES
        else:
            fl, missing = False, missing + 1
        g = shp_transform(to_pix, shape(ft["geometry"])).buffer(width_px / 2.0)
        if g.is_empty:
            continue
        geoms.append(g); flags.append(int(fl))
    if not geoms:
        return np.zeros((H, W), np.int16), [], missing
    order = sorted(range(len(geoms)), key=lambda i: flags[i])  # flooded drawn last
    shapes = [(geoms[i], i + 1) for i in order]
    seg = rasterize(shapes, out_shape=(H, W), transform=Affine.identity(), fill=0,
                    dtype="int32", all_touched=True)
    return seg.astype(np.int16 if len(geoms) < 32000 else np.int32), flags, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="dir containing the extracted region folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--regions", nargs="+", default=["Germany_Training_Public", "Louisiana-East_Training_Public"])
    ap.add_argument("--road_width_px", type=float, default=8.0)
    ap.add_argument("--min_valid_post", type=float, default=0.5, help="min fraction of non-zero post pixels")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    manifest, stats, skipped = [], {}, []

    for reg_dir in args.regions:
        region = reg_dir.replace("_Training_Public", "")
        rdir = os.path.join(args.raw, reg_dir)
        files = index_files(rdir)
        csv_path = find_csv(rdir)
        df = pd.read_csv(csv_path)
        c_pre = pick(df, "pre"); c_post = pick(df, "post", "1") or pick(df, "post")
        c_road = pick(df, "road")
        print(f"[{region}] csv={csv_path}\n  columns={list(df.columns)}\n  using pre={c_pre} post={c_post} road={c_road}")
        if None in (c_pre, c_post, c_road):
            raise RuntimeError("could not match csv columns; edit pick() calls")
        st = dict(tiles=0, road_px=0, flooded_px=0, segs=0, flooded_segs=0, missing_flood_prop=0)
        for _, row in df.iterrows():
            pre_p = files.get(os.path.basename(str(row[c_pre])))
            post_p = files.get(os.path.basename(str(row[c_post])))
            road_p = files.get(os.path.basename(str(row[c_road]))) if isinstance(row[c_road], str) else None
            if pre_p is None or post_p is None:
                skipped.append((region, str(row[c_pre]), "missing image")); continue
            with rasterio.open(pre_p) as sp:
                pre = sp.read()[:3]; tfm, crs = sp.transform, sp.crs; H, W = sp.height, sp.width
                bounds = sp.bounds
            if min(H, W) < 512:
                skipped.append((region, pre_p, "tile smaller than 512")); continue
            post = np.zeros_like(pre)
            with rasterio.open(post_p) as ss:
                for i in range(3):
                    reproject(rasterio.band(ss, i + 1), post[i], dst_transform=tfm, dst_crs=crs,
                              resampling=Resampling.bilinear)
            valid = float((post.max(0) > 0).mean())
            if valid < args.min_valid_post:
                skipped.append((region, post_p, f"post valid fraction {valid:.2f}")); continue
            seg, flags, miss = segments_to_raster(road_p, crs, tfm, H, W, args.road_width_px)
            if seg is None:
                skipped.append((region, str(row[c_road]), "missing road label")); continue
            tid = f"{region}__{Path(pre_p).stem}"
            np.save(f"{args.out}/{tid}_pre.npy", pre.astype(np.uint8))
            np.save(f"{args.out}/{tid}_post.npy", post.astype(np.uint8))
            np.save(f"{args.out}/{tid}_seg.npy", seg)
            json.dump({"flooded": flags}, open(f"{args.out}/{tid}_seg.json", "w"))
            tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True) if crs and crs.to_string() != "EPSG:4326" else None
            xs, ys = [bounds.left, bounds.right], [bounds.bottom, bounds.top]
            if tr is not None:
                xs, ys = tr.transform(xs, ys)
            manifest.append(dict(tid=tid, region=region, H=H, W=W, cx=(xs[0] + xs[1]) / 2, cy=(ys[0] + ys[1]) / 2,
                                 w=abs(xs[1] - xs[0]), h=abs(ys[1] - ys[0]), post_valid=valid))
            fl = np.array([0] + flags, dtype=np.int64)
            st["tiles"] += 1; st["segs"] += len(flags); st["flooded_segs"] += int(sum(flags))
            st["road_px"] += int((seg > 0).sum()); st["flooded_px"] += int((fl[seg] == 1).sum())
            st["missing_flood_prop"] += miss
        stats[region] = st
    pd.DataFrame(manifest).to_csv(f"{args.out}/manifest.csv", index=False)
    for r, s in stats.items():
        s["flooded_px_share_of_road"] = s["flooded_px"] / max(s["road_px"], 1)
        s["flooded_seg_share"] = s["flooded_segs"] / max(s["segs"], 1)
    json.dump(dict(stats=stats, skipped=skipped), open(f"{args.out}/dataset_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1)); print(f"skipped {len(skipped)} tiles (see dataset_stats.json)")
    if any(s["missing_flood_prop"] for s in stats.values()):
        print("WARNING: some road features had no 'flooded' property -> treated as not flooded. Check the geojson keys.")


if __name__ == "__main__":
    main()
