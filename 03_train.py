"""Step 3: elastic-net age clocks -- genes only, and genes + LeafCutter splicing.

    python 03_train.py --data data/processed --out results [--jobs 5] [--no-loso]

Evaluation: leave-one-study-out. Each of GSE94438, GSE79362, GSE134080 and GTEx is held out in turn (GSE122485,
n=3, is always in training); every fitted step is redone on the remaining studies:
  genes   gene filter, TMM log-CPM, GTEx technical covariates removed (age protected), study gene filters,
          top 5,000 genes by training variance, z-scored
  splice  LeafCutter clusters and PSI fitted on training rows, top 5,000 introns, z-scored
  model   elastic net (l1_ratio 0.5, one-SE alpha from 5-fold CV stratified by study x age bin) on Horvath age
GTEx ages are 10-year bin midpoints, so its held-out MAE is against midpoints.
Final models: both models refitted on all samples and saved with everything needed to score new samples.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import clock

HELD_OUT = ["GSE94438", "GSE79362", "GSE134080", "GTEx"]
MODELS = ["gene", "gene+leafcutter"]


def load(d):
    d = Path(d)
    s = pd.read_csv(d / "samples.csv", dtype={"rail_id": str})
    return (s, pd.read_csv(d / "genes.csv.gz"), sp.load_npz(d / "gene_counts.npz").tocsr(),
            pd.read_csv(d / "junctions.csv.gz"), sp.load_npz(d / "junction_counts.npz").tocsr())


def features(s, genes, G, jx, J, train):
    Zg, gp = clock.gene_features(G, genes, s, train)
    Zs, spar = clock.splice_features(J, jx, train)
    return {"gene": Zg, "gene+leafcutter": np.hstack([Zg, Zs])}, gp, spar


def metrics(age, pred):
    e = pred - age
    out = {"n": len(age), "MAE": np.abs(e).mean(), "r": np.corrcoef(age, pred)[0, 1], "slope": np.polyfit(age, pred, 1)[0],
           "bias": e.mean()}
    for lo, hi, lab in ((0, 30, "<30"), (30, 50, "30-49"), (50, 200, "50+")):
        m = (age >= lo) & (age < hi)
        out[f"MAE_{lab}"], out[f"n_{lab}"] = (np.abs(e[m]).mean() if m.any() else np.nan), int(m.sum())
    return out


def loso(s, genes, G, jx, J, jobs, out):
    study, age = s["study"].to_numpy(), s["age"].to_numpy()
    y = clock.horvath(age)
    rows, preds, t0 = [], [], time.time()
    for held in HELD_OUT:
        test, train = np.flatnonzero(study == held), np.flatnonzero(study != held)
        Z, _, _ = features(s, genes, G, jx, J, train)
        splits = clock.cv_splits(study[train], age[train])
        for m in MODELS:
            p = clock.horvath_inv(clock.fit_enet(Z[m][train], y[train], splits, jobs).predict(Z[m][test]))
            rows.append({"held_out": held, "model": m, **metrics(age[test], p)})
            preds.append(pd.DataFrame({"held_out": held, "model": m, "rail_id": s["rail_id"].to_numpy()[test],
                                       "age": age[test], "pred": p}))
        print(f"held out {held}: {time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(rows)
    pooled = []
    for m in MODELS:
        p = pd.concat(preds)
        p = p[(p["model"] == m) & (p["held_out"] != "GTEx")]
        pooled.append({"held_out": "GEO studies pooled", "model": m, **metrics(p["age"].to_numpy(), p["pred"].to_numpy())})
    res = pd.concat([res, pd.DataFrame(pooled)], ignore_index=True)
    res.to_csv(out / "loso_metrics.csv", index=False)
    pd.concat(preds).to_csv(out / "loso_predictions.csv", index=False)
    return res


def final_models(s, genes, G, jx, J, jobs, out):
    study, age = s["study"].to_numpy(), s["age"].to_numpy()
    allr = np.arange(len(s))
    Z, gp, spar = features(s, genes, G, jx, J, allr)
    splits = clock.cv_splits(study, age)
    n_gene = len(gp["top"])
    gene_ids = genes["gene_id"].to_numpy()[gp["gene_cols"]]
    fi = spar["feature_introns"]
    for m in MODELS:
        e = clock.fit_enet(Z[m], clock.horvath(age), splits, jobs)
        extra = {} if m == "gene" else {
            "cluster_intron_keys": clock.junction_key(jx.iloc[spar["introns"]]), "cluster_ids": spar["cluster"],
            "feature_intron_keys": clock.junction_key(jx.iloc[fi]), "psi_fill": spar["psi_fill"],
            "splice_mu": spar["mu"], "splice_sd": spar["sd"]}
        name = m.replace("+", "_")
        np.savez(out / f"model_{name}.npz", gene_ids=gene_ids, tmm_reference_counts=gp["tmm_reference_counts"],
                 tmm_scale=gp["tmm_scale"], gene_top=gp["top"], gene_mu=gp["mu"], gene_sd=gp["sd"],
                 coef=e.coef_, intercept=e.intercept_, alpha=e.alpha, target="horvath", **extra)
        names = list(gene_ids[gp["top"]]) + ([] if m == "gene" else list(clock.junction_key(jx.iloc[fi])))
        kind = ["gene"] * n_gene + ([] if m == "gene" else ["intron"] * len(fi))
        cf = pd.DataFrame({"type": kind, "feature": names, "coef": e.coef_})
        cf = cf[cf["coef"] != 0]
        cf.reindex(cf["coef"].abs().sort_values(ascending=False).index).to_csv(out / f"model_{name}_coefficients.csv", index=False)
        print(f"final {m}: alpha {e.alpha:.4f}, {int((cf['type'] == 'gene').sum())} genes, "
              f"{int((cf['type'] == 'intron').sum())} introns -> {out / f'model_{name}.npz'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results")
    ap.add_argument("--jobs", type=int, default=5, help="parallel CV folds (5 is the useful maximum)")
    ap.add_argument("--no-loso", action="store_true", help="only fit the final models")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(a.data)
    print("samples:", data[0]["study"].value_counts().to_dict(), flush=True)
    if not a.no_loso:
        res = loso(*data, a.jobs, out)
        pd.set_option("display.width", 200)
        print(res.round(2).to_string(index=False))
    final_models(*data, a.jobs, out)


if __name__ == "__main__":
    main()
