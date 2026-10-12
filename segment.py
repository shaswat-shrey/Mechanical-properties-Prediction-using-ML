"""
Segment raw AE hit files into layers and build lay_table.csv + block_table.csv.

Layer boundaries are found from the ~70 s inter-layer pause in the hit
timestamps (instead of splitting each file 50/50 by row count). Files whose
pauses don't match the expected layer count fall back to an equal-time split.

Each sample is then cut into blocks of BLOCK_SIZE consecutive layers, so
block_table has many rows per sample instead of one.

Usage: python segment.py [BLOCK_SIZE]
"""
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = "SAMPLE 1 TO 8"
BLOCK_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 4
GAP_S = 30.0  # pause longer than this = new layer

COLS = ["hit_id", "date", "time", "PARA1", "CH", "RISE", "COUNTS", "ENERGY", "DURATION",
        "AMP", "A_FRQ", "RMS", "ASL", "PCNTS", "THR", "R_FRQ", "I_FRQ", "SIG_STRNGTH",
        "ABS_ENERGY", "FRQ_C", "P_FRQ"]
FEATURES = ["ABS_ENERGY", "RISE", "COUNTS", "DURATION", "AMP", "A_FRQ", "FRQ_C", "P_FRQ",
            "SIG_STRNGTH"]

HARDNESS = {1: 310.2, 2: 304.7, 3: 291.6, 4: 305.1, 5: 313.3, 6: 307.2, 7: 303.2, 8: 305.6,
            9: 293.4}


def time_to_seconds(t):
    h, ap, m, s = re.match(r"(\d+)([ap]?):(\d+):([\d.]+)", str(t).strip()).groups()
    h = int(h)
    if ap == "p" and h != 12:
        h += 12
    if ap == "a" and h == 12:
        h = 0
    return h * 3600 + int(m) * 60 + float(s)


def parse_layer_file(path):
    df = pd.read_csv(path, encoding="latin1")
    df.columns = COLS
    df[COLS[5:]] = df[COLS[5:]].apply(pd.to_numeric, errors="coerce")
    df["t"] = df["time"].map(time_to_seconds)
    return df


def split_layers(df, layer_ids):
    """Return a layer_id per row, cutting at the largest inter-hit pauses."""
    k = len(layer_ids)
    if k == 1:
        return np.full(len(df), layer_ids[0]), "single"
    gaps = df["t"].diff()
    cuts = gaps.nlargest(k - 1)
    if (cuts > GAP_S).all():
        cut_pos = sorted(df.index.get_indexer(cuts.index))
        method = "gap"
    else:
        t0, t1 = df["t"].iloc[0], df["t"].iloc[-1]
        edges = t0 + (t1 - t0) * np.arange(1, k) / k
        cut_pos = list(np.searchsorted(df["t"].values, edges))
        method = "equal_time"
    labels = np.empty(len(df), dtype=int)
    bounds = [0] + cut_pos + [len(df)]
    for lid, a, b in zip(layer_ids, bounds[:-1], bounds[1:]):
        labels[a:b] = lid
    return labels, method


def layer_features(g):
    out = {"n_hits": len(g), "duration_s": g["t"].iloc[-1] - g["t"].iloc[0]}
    out["hit_rate"] = out["n_hits"] / max(out["duration_s"], 1.0)
    for c in FEATURES:
        v = g[c].dropna()
        out[f"{c}_mean"] = v.mean()
        out[f"{c}_std"] = v.std()
        out[f"{c}_median"] = v.median()
    e = np.log10(g["ABS_ENERGY"].clip(lower=1e-3))
    out["logE_mean"] = e.mean()
    out["logE_std"] = e.std()
    out["logE_p95"] = e.quantile(0.95)
    return out


def make_blocks(lay, block_size):
    """Group each sample's layers into blocks of block_size consecutive layers."""
    lay = lay.copy()
    lay["block_id"] = lay.groupby("sample_id").cumcount() // block_size
    feat_cols = [c for c in lay.columns
                 if c not in ("sample_id", "layer_id", "block_id", "level", "travel_speed", "heat_input")]
    block = lay.groupby(["sample_id", "block_id", "level", "travel_speed", "heat_input"]).agg(
        layer_start=("layer_id", "min"),
        layer_end=("layer_id", "max"),
        n_layers=("layer_id", "count"),
        ABS_ENERGY_blockstd=("ABS_ENERGY_mean", "std"),
        RISE_blockmean=("RISE_mean", "mean"),
        **{c: (c, "mean") for c in feat_cols if c != "RISE_mean"},
    ).reset_index()
    block = block[block["n_layers"] >= 2].copy()  # blockstd needs at least 2 layers
    block["hardness"] = block["sample_id"].map(HARDNESS)
    return block


def main():
    process = pd.read_csv("process_table.csv", skipinitialspace=True)
    process.columns = process.columns.str.strip()

    layer_rows, methods = [], []
    for sample_dir in sorted(os.listdir(ROOT)):
        sid = int(re.search(r"SAMPLE\s+(\d+)", sample_dir).group(1))
        files_dir = os.path.join(ROOT, sample_dir, "files")
        for fname in sorted(os.listdir(files_dir)):
            m = re.search(r"Layer_(\d+)(?:_(?:and|to)_(\d+))?", fname, re.IGNORECASE)
            if not m:
                print("SKIPPED:", fname)
                continue
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) else a
            layer_ids = list(range(a, b + 1))  # "25_and_27" / "29_to_32" hold every layer in between
            df = parse_layer_file(os.path.join(files_dir, fname))
            df["layer_id"], method = split_layers(df, layer_ids)
            methods.append((sid, fname, method))
            for lid, g in df.groupby("layer_id"):
                layer_rows.append({"sample_id": sid, "layer_id": lid, **layer_features(g)})

    lay = pd.DataFrame(layer_rows).sort_values(["sample_id", "layer_id"]).reset_index(drop=True)
    lay = lay.merge(process, on="sample_id", how="left")
    lay.to_csv("lay_table.csv", index=False)

    m = pd.DataFrame(methods, columns=["sample_id", "file", "method"])
    print("Split method per file:", m["method"].value_counts().to_dict())
    print(m[m["method"] == "equal_time"].to_string(index=False))
    print("\nLayers per sample:", lay.groupby("sample_id").size().to_dict())

    block = make_blocks(lay, BLOCK_SIZE)
    block.to_csv("block_table.csv", index=False)
    print(f"\nblock_table.csv: {block.shape} (BLOCK_SIZE={BLOCK_SIZE})")
    print("Blocks per sample:", block.groupby("sample_id").size().to_dict())
    print("NaNs:", int(block.isna().sum().sum()))


if __name__ == "__main__":
    main()
