"""
Train hardness models with leave-one-sample-out CV and save every raw result.

All blocks of the held-out sample are left out together, so no block from the
test sample leaks into training. Sample-level MAE averages the block
predictions of each held-out sample.

Reads lay_table.csv (built by segment.py), evaluates several block sizes,
and writes raw outputs to results/:
  cv_results.csv          every (block_size, feature set, model) -> metrics
  block_predictions.csv   every out-of-fold prediction, per block
  sample_predictions.csv  per-sample mean of block predictions
  feature_importance.csv  coefficients / importances of each model fit on all data
  ridge_alphas.csv        Ridge alpha chosen in each CV fold
  train_log.txt           full console output
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from segment import make_blocks

warnings.filterwarnings("ignore")

OUT = "results"
BLOCK_SIZES = [2, 3, 4, 6, 8, 100]  # 100 = whole sample in one block (the original setup)
META = ["sample_id", "block_id", "layer_start", "layer_end", "n_layers", "hardness", "level"]
MODELS = {
    "Mean": lambda: DummyRegressor(),
    "Ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 50))),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0),
}


class Tee:
    def __init__(self, path):
        self.f, self.out = open(path, "w"), sys.stdout

    def write(self, s):
        self.f.write(s)
        self.out.write(s)

    def flush(self):
        self.f.flush()
        self.out.flush()


def feature_sets(block):
    all_feats = [c for c in block.columns if c not in META]
    ae = [c for c in all_feats if c not in ("heat_input", "travel_speed")]
    return {
        "heat_input only": ["heat_input"],
        "process params": ["heat_input", "travel_speed"],
        "notebook feats": ["heat_input", "ABS_ENERGY_blockstd", "RISE_blockmean"],
        "AE feats only": ae,
        "all feats": all_feats,
    }


def run_cv(make_model, X, y, groups):
    pred = np.empty(len(y))
    alphas = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = make_model().fit(X.iloc[tr], y.iloc[tr])
        pred[te] = m.predict(X.iloc[te])
        if hasattr(m, "named_steps"):
            alphas.append((groups.iloc[te[0]], m[-1].alpha_))
    return pred, alphas


def metrics(y, pred, groups):
    err = pred - y
    ps = pd.DataFrame({"sample_id": groups, "true": y, "pred": pred}).groupby("sample_id").mean()
    serr = ps["pred"] - ps["true"]
    ss_tot = ((ps["true"] - ps["true"].mean()) ** 2).sum()
    return {
        "block_MAE": np.abs(err).mean(),
        "block_RMSE": np.sqrt((err ** 2).mean()),
        "sample_MAE": np.abs(serr).mean(),
        "sample_RMSE": np.sqrt((serr ** 2).mean()),
        "sample_R2": 1 - (serr ** 2).sum() / ss_tot,
        "sample_max_err": np.abs(serr).max(),
        "sample_pearson_r": np.corrcoef(ps["true"], ps["pred"])[0, 1] if ps["pred"].std() > 0 else np.nan,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    sys.stdout = Tee(os.path.join(OUT, "train_log.txt"))

    lay = pd.read_csv("lay_table.csv")
    print(f"lay_table: {lay.shape[0]} layers, {lay['sample_id'].nunique()} samples\n")

    cv_rows, pred_rows, sample_rows, imp_rows, alpha_rows = [], [], [], [], []
    for bs in BLOCK_SIZES:
        block = make_blocks(lay, bs).reset_index(drop=True)
        y, groups = block["hardness"], block["sample_id"]
        print(f"=== block_size={bs}: {len(block)} blocks "
              f"{block.groupby('sample_id').size().to_dict()} ===")
        for fs_name, cols in feature_sets(block).items():
            X = block[cols]
            for mname, make_model in MODELS.items():
                if mname == "Mean" and fs_name != "heat_input only":
                    continue  # mean predictor ignores features; run it once
                fs = "-" if mname == "Mean" else fs_name
                pred, alphas = run_cv(make_model, X, y, groups)
                m = metrics(y, pred, groups)
                cv_rows.append({"block_size": bs, "n_blocks": len(block), "features": fs,
                                "n_features": 0 if mname == "Mean" else len(cols),
                                "model": mname, **m})
                tag = {"block_size": bs, "features": fs, "model": mname}
                pred_rows.append(pd.DataFrame({
                    **tag, "sample_id": groups, "block_id": block["block_id"],
                    "layer_start": block["layer_start"], "layer_end": block["layer_end"],
                    "true": y, "pred": pred, "error": pred - y}))
                ps = pd.DataFrame({"sample_id": groups, "true": y, "pred": pred}) \
                    .groupby("sample_id").mean().reset_index()
                sample_rows.append(ps.assign(**tag, error=ps["pred"] - ps["true"]))
                alpha_rows += [{**tag, "held_out_sample": s, "alpha": a} for s, a in alphas]
                if mname != "Mean":
                    full = make_model().fit(X, y)
                    vals = full[-1].coef_ if mname == "Ridge" else full.feature_importances_
                    imp_rows += [{**tag, "feature": c, "value": v} for c, v in zip(cols, vals)]

    cv = pd.DataFrame(cv_rows)
    cv.to_csv(os.path.join(OUT, "cv_results.csv"), index=False)
    pd.concat(pred_rows).to_csv(os.path.join(OUT, "block_predictions.csv"), index=False)
    pd.concat(sample_rows)[["block_size", "features", "model", "sample_id", "true", "pred", "error"]] \
        .to_csv(os.path.join(OUT, "sample_predictions.csv"), index=False)
    pd.DataFrame(imp_rows).to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)
    pd.DataFrame(alpha_rows).to_csv(os.path.join(OUT, "ridge_alphas.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\nAll CV results (leave-one-sample-out, hardness in HV):")
    print(cv.round(3).to_string(index=False))

    best = cv[cv.model != "Mean"].sort_values("sample_MAE").iloc[0]
    base = cv[(cv.model == "Mean") & (cv.block_size == best.block_size)].iloc[0]
    print(f"\nBest: {best.model} | {best.features} | block_size={best.block_size} -> "
          f"sample MAE {best.sample_MAE:.3f}, R2 {best.sample_R2:.3f} "
          f"(mean predictor MAE {base.sample_MAE:.3f})")
    sp = pd.concat(sample_rows)
    print(sp[(sp.block_size == best.block_size) & (sp.features == best.features)
             & (sp.model == best.model)][["sample_id", "true", "pred", "error"]].round(2).to_string(index=False))
    print(f"\nRaw results written to {OUT}/")


if __name__ == "__main__":
    main()
