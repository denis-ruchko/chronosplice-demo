"""Shared functions: recount3 I/O, gene normalisation, GTEx covariate handling, LeafCutter, elastic net.

Every function that learns something takes the training row indices and uses only those rows.
"""
import gzip
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import StratifiedKFold

# ----------------------------------------------------------------------------- data sources
RECOUNT3 = "https://duffel.rail.bio/recount3/human"
GTF_URL = f"{RECOUNT3}/annotations/gene_sums/human.gene_sums.G026.gtf.gz"
GEO_STUDIES = {"GSE94438": "SRP098758", "GSE79362": "SRP071965", "GSE134080": "SRP214077", "GSE122485": "SRP168555"}
GTEX = f"{RECOUNT3}/data_sources/gtex"


def geo_urls(srp):
    sra, d = f"{RECOUNT3}/data_sources/sra", srp[-2:]
    return {"meta": f"{sra}/metadata/{d}/{srp}/sra.sra.{srp}.MD.gz",
            "qc": f"{sra}/metadata/{d}/{srp}/sra.recount_qc.{srp}.MD.gz",
            "gene_sums": f"{sra}/gene_sums/{d}/{srp}/sra.gene_sums.{srp}.G026.gz",
            "ID": f"{sra}/junctions/{d}/{srp}/sra.junctions.{srp}.ALL.ID.gz",
            "RR": f"{sra}/junctions/{d}/{srp}/sra.junctions.{srp}.ALL.RR.gz",
            "MM": f"{sra}/junctions/{d}/{srp}/sra.junctions.{srp}.ALL.MM.gz"}


GTEX_URLS = {"meta": f"{GTEX}/metadata/OD/BLOOD/gtex.gtex.BLOOD.MD.gz",
             "qc": f"{GTEX}/metadata/OD/BLOOD/gtex.recount_qc.BLOOD.MD.gz",
             "gene_sums": f"{GTEX}/gene_sums/OD/BLOOD/gtex.gene_sums.BLOOD.G026.gz",
             "ID": f"{GTEX}/junctions/OD/BLOOD/gtex.junctions.BLOOD.ALL.ID.gz",
             "RR": f"{GTEX}/junctions/OD/BLOOD/gtex.junctions.BLOOD.ALL.RR.gz",
             "MM": f"{GTEX}/junctions/OD/BLOOD/gtex.junctions.BLOOD.ALL.MM.gz"}


def raw_paths(raw):
    """Local file for every download: {"GTF": path, srp: {kind: path}, "GTEX": {kind: path}}."""
    raw = Path(raw)
    p = {"GTF": raw / "human.gene_sums.G026.gtf.gz", "GTEX": {k: raw / "gtex" / f"BLOOD.{k}.gz" for k in GTEX_URLS}}
    for srp in GEO_STUDIES.values():
        p[srp] = {k: raw / srp / f"{srp}.{k}.gz" for k in geo_urls(srp)}
    return p


def download(url, dest, retries=4):
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest.with_suffix(".part"), "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            dest.with_suffix(".part").rename(dest)
            return dest
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def read_tsv(path, **kw):
    return pd.read_csv(path, sep="\t", low_memory=False, **kw)


def read_gene_sums(path):
    """recount3 gene x sample base-coverage sums (header lines start with '#')."""
    return pd.read_csv(path, sep="\t", comment="#", index_col=0)


def read_gtf_genes(path):
    g = read_tsv(path, comment="#", header=None, usecols=[0, 2, 3, 4, 6, 8],
                 names=["chrom", "feature", "start", "end", "strand", "attr"])
    g = g[g["feature"] == "gene"]
    attr = lambda k: g["attr"].str.extract(rf'{k} "([^"]+)"')[0]
    return pd.DataFrame({"gene_id": attr("gene_id"), "gene_name": attr("gene_name"), "gene_type": attr("gene_type"),
                         "chrom": g["chrom"].to_numpy(), "start": g["start"].to_numpy(), "end": g["end"].to_numpy(),
                         "strand": g["strand"].to_numpy()})


def mm_header(path):
    """(lines to skip, (n_junctions, n_samples, nnz)) of a recount3 MatrixMarket file."""
    with gzip.open(path, "rt") as f:
        for n, line in enumerate(f, 1):
            if not line.startswith("%"):
                return n, tuple(map(int, line.split()))


def mm_chunks(path, skip, chunk=5_000_000):
    """Stream (junction i, sample j, count v) triplets, 1-based, in chunks."""
    yield from pd.read_csv(path, sep=r"\s+", header=None, skiprows=skip, names=["i", "j", "v"], chunksize=chunk,
                           dtype={"i": np.int32, "j": np.int32, "v": np.float32})


def junction_key(t):
    return (t["chrom"].astype(str) + ":" + t["start"].astype(str) + "-" + t["end"].astype(str) + ":"
            + t["strand"].astype(str)).to_numpy()


# ----------------------------------------------------------------------------- target
def horvath(age):
    """Horvath (2013) age transform: log(age+1) - log(21) up to 20 years, (age-20)/21 above."""
    age = np.asarray(age, float)
    return np.where(age <= 20, np.log(age + 1) - np.log(21), (age - 20) / 21)


def horvath_inv(x):
    x = np.asarray(x, float)
    return np.where(x <= 0, 21 * np.exp(x) - 1, 21 * x + 20)


# ----------------------------------------------------------------------------- gene features
GLOBIN = {"HBA1", "HBA2", "HBB", "HBD", "HBG1", "HBG2", "HBM", "HBQ1", "HBZ", "HBE1"}
LNCRNA = {"lincRNA", "antisense", "sense_intronic", "sense_overlapping", "processed_transcript",
          "3prime_overlapping_ncRNA", "bidirectional_promoter_lncRNA", "macro_lncRNA", "non_coding"}
CPM_MIN, CPM_FRAC, N_GENES = 1.0, 0.25, 5000
GTEX_COVS = ["rin", "ischemic_min", "centre_c1", "log_depth", "male"]


def gene_filter(G, genes, train):
    """Protein-coding + lncRNA genes, no globin/chrM/chrY/rRNA, CPM >= 1 in >= 25% of training samples."""
    typ = genes["gene_type"].isin(LNCRNA | {"protein_coding"})
    drop = genes["gene_name"].isin(GLOBIN) | genes["chrom"].isin(["chrM", "chrY"]) | genes["gene_type"].isin(["rRNA", "Mt_rRNA"])
    Gt = G[train]
    cpm = Gt.multiply(1e6 / np.asarray(Gt.sum(axis=1))).tocsc()
    expressed = np.asarray((cpm >= CPM_MIN).mean(axis=0)).ravel() >= CPM_FRAC
    return np.flatnonzero(typ.to_numpy() & ~drop.to_numpy() & expressed)


def tmm_factor(obs, ref, trim_m=0.3, trim_a=0.05):
    """edgeR TMM factor of one sample against the reference sample."""
    n_o, n_r = obs.sum(), ref.sum()
    ok = (obs > 0) & (ref > 0)
    o, r = obs[ok] / n_o, ref[ok] / n_r
    m, a = np.log2(o / r), (np.log2(o) + np.log2(r)) / 2
    v = (n_o - obs[ok]) / n_o / obs[ok] + (n_r - ref[ok]) / n_r / ref[ok]
    rank = lambda x: x.argsort().argsort()
    n = len(m)
    keep = ((rank(m) >= np.floor(n * trim_m)) & (rank(m) < n - np.floor(n * trim_m))
            & (rank(a) >= np.floor(n * trim_a)) & (rank(a) < n - np.floor(n * trim_a)))
    return 2 ** (np.sum(m[keep] / v[keep]) / np.sum(1 / v[keep]))


def fit_tmm(C, train):
    """Reference = training sample whose upper quartile is closest to the training mean (edgeR default)."""
    uq = np.array([np.percentile(c, 75) for c in C[train] / C[train].sum(axis=1, keepdims=True)])
    ref = train[np.argmin(np.abs(uq - uq.mean()))]
    f_train = np.array([tmm_factor(C[i], C[ref]) for i in train])
    return {"reference": int(ref), "scale": float(np.exp(np.mean(np.log(f_train))))}


def tmm_logcpm(C, tmm, ref_counts=None):
    ref = C[tmm["reference"]] if ref_counts is None else ref_counts
    f = np.array([tmm_factor(c, ref) for c in C]) / tmm["scale"]
    eff = C.sum(axis=1) * f
    return np.log2((C + 0.5) / (eff[:, None] + 1) * 1e6).astype(np.float32)


def residualize_gtex(X, s, train):
    """GTEx training rows: remove the technical-covariate part of  x ~ age-bin dummies + GTEX_COVS  (the age
    effect is protected by the dummies). Other rows are left as they are."""
    g = train[s["study"].to_numpy()[train] == "GTEx"]
    if len(g) < 30:
        return X
    Cv = s.loc[g, GTEX_COVS].astype(float)
    Cv = Cv.fillna(Cv.median()).to_numpy()
    Cv = Cv - Cv.mean(0)
    D = np.column_stack([pd.get_dummies(s.loc[g, "age"].astype(int)).to_numpy(float), Cv])
    B, *_ = np.linalg.lstsq(D, X[g], rcond=None)
    X = X.copy()
    X[g] -= Cv @ B[-Cv.shape[1]:]
    return X


def study_gene_filters(X, s, train, fc_max=1.0):
    """Keep genes whose within-study age correlation has the same sign in GTEx and GSE94438, and drop genes
    differing by |log2 FC| > fc_max between GTEx < 40 and GSE94438 aged 20-39. Needs both studies in training."""
    study, age = s["study"].to_numpy(), s["age"].to_numpy()
    g, h = train[study[train] == "GTEx"], train[study[train] == "GSE94438"]
    if len(g) < 30 or len(h) < 30:
        return np.ones(X.shape[1], bool)
    corr = lambda rows: ((X[rows] - X[rows].mean(0)) * (age[rows] - age[rows].mean())[:, None]).sum(0)
    same_sign = np.sign(corr(g)) * np.sign(corr(h)) > 0
    mg, mh = g[age[g] < 40], h[(age[h] >= 20) & (age[h] < 40)]
    return same_sign & ~(np.abs(X[mg].mean(0) - X[mh].mean(0)) > fc_max)


def gene_features(G, genes, s, train):
    """z-scored log-CPM of the top N_GENES genes (all rows) and the fitted parameters."""
    cols = gene_filter(G, genes, train)
    C = G[:, cols].toarray()
    tmm = fit_tmm(C, train)
    ref_counts = C[tmm["reference"]].copy()
    X = tmm_logcpm(C, tmm)
    del C
    X = residualize_gtex(X, s, train)
    idx = np.flatnonzero(study_gene_filters(X, s, train))
    top = idx[np.argsort(X[train][:, idx].var(0))[::-1][:N_GENES]]
    mu, sd = X[train][:, top].mean(0), X[train][:, top].std(0) + 1e-6
    return (X[:, top] - mu) / sd, {"gene_cols": cols, "tmm_scale": tmm["scale"], "tmm_reference_counts": ref_counts,
                                   "top": top, "mu": mu, "sd": sd}


# ----------------------------------------------------------------------------- LeafCutter splicing
MAX_INTRON, MIN_CLUREADS, MIN_RATIO = 100_000, 30, 0.001
JX_MIN_READS, JX_MIN_FRAC, PSI_MIN_READS, N_INTRONS = 5, 0.05, 20, 5000


def _components(jt):
    n = len(jt)
    base = jt["chrom"].astype(str) + ":" + jt["strand"].astype(str)
    left = pd.factorize(base + ":L" + jt["start"].astype(str))[0]
    right = pd.factorize(base + ":R" + jt["end"].astype(str))[0]
    sites = np.r_[left, right + left.max() + 1] + n
    graph = sp.csr_matrix((np.ones(2 * n), (np.r_[np.arange(n), np.arange(n)], sites)), shape=(sites.max() + 1,) * 2)
    return pd.factorize(connected_components(graph, directed=False)[1][:n])[0]


def fit_clusters(J, junctions, train):
    """LeafCutter rules on training rows: stranded introns <= 100 kb with >= 5 reads in >= 5% of samples;
    cluster by shared donor/acceptor; repeatedly drop introns with < 0.1% of their cluster's reads and
    re-cluster; drop singleton clusters and clusters with < 30 pooled reads."""
    Jt = J[train]
    hit = np.bincount(Jt.indices[Jt.data >= JX_MIN_READS], minlength=J.shape[1])
    length = (junctions["end"] - junctions["start"]).abs().to_numpy()
    idx = np.flatnonzero((hit >= JX_MIN_FRAC * len(train)) & junctions["strand"].isin(["+", "-"]).to_numpy()
                         & (length <= MAX_INTRON))
    pooled = np.asarray(Jt.sum(0)).ravel()
    while True:
        cid = _components(junctions.iloc[idx].reset_index(drop=True))
        ok = pooled[idx] >= MIN_RATIO * np.bincount(cid, weights=pooled[idx])[cid]
        if ok.all():
            break
        idx = idx[ok]
    keep = (np.bincount(cid)[cid] >= 2) & (np.bincount(cid, weights=pooled[idx])[cid] >= MIN_CLUREADS)
    return idx[keep], pd.factorize(cid[keep])[0]


def psi(J, introns, cluster, rows):
    """rows x introns PSI = intron reads / cluster reads; NaN where the cluster has < 20 reads."""
    K = J[rows][:, introns].toarray().astype(np.float32)
    S = sp.csr_matrix((np.ones(len(introns), np.float32), (np.arange(len(introns)), cluster)),
                      shape=(len(introns), cluster.max() + 1))
    N = (sp.csr_matrix(K) @ S).toarray()[:, cluster]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(N >= PSI_MIN_READS, K / np.maximum(N, 1), np.nan).astype(np.float32)


def splice_features(J, junctions, train):
    """z-scored PSI of the top N_INTRONS introns (all rows); missing PSI = training mean."""
    introns, cluster = fit_clusters(J, junctions, train)
    P = psi(J, introns, cluster, np.arange(J.shape[0]))
    has = np.flatnonzero(np.isfinite(P[train]).any(0))
    P = P[:, has]
    fill = np.nanmean(P[train], 0)
    P = np.where(np.isfinite(P), P, fill)
    top = np.argsort(P[train].var(0))[::-1][:N_INTRONS]
    mu, sd = P[train][:, top].mean(0), P[train][:, top].std(0) + 1e-6
    return (P[:, top] - mu) / sd, {"introns": introns, "cluster": cluster, "feature_introns": introns[has][top],
                                   "psi_fill": fill[top], "mu": mu, "sd": sd}


# ----------------------------------------------------------------------------- elastic net
L1_RATIO, N_ALPHAS, N_FOLDS, SEED = 0.5, 60, 5, 0
AGE_BINS = ([0, 18, 30, 50, np.inf], ["<18", "18-29", "30-49", "50+"])


def cv_splits(study, age, seed=SEED, n_folds=N_FOLDS):
    """Stratified K-fold on study x age bin (strata smaller than n_folds pooled as 'rare')."""
    lab = pd.Series(np.asarray(study).astype(str)) + "|" + pd.cut(np.asarray(age, float), AGE_BINS[0], right=False,
                                                                   labels=AGE_BINS[1]).astype(str)
    lab[lab.map(lab.value_counts()) < n_folds] = "rare"
    return list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(len(lab)), lab.to_numpy()))


def alpha_grid(X, y, eps=1e-3):
    a_max = np.abs((X - X.mean(0)).T @ (y - y.mean())).max() / (len(y) * L1_RATIO)
    return np.logspace(np.log10(a_max), np.log10(a_max * eps), N_ALPHAS)


def fit_enet(X, y, splits, n_jobs=5):
    """Elastic net (l1_ratio 0.5) with the one-standard-error rule: the largest alpha whose mean CV error is
    within one SE of the minimum."""
    cv = ElasticNetCV(l1_ratio=L1_RATIO, alphas=alpha_grid(X, y), cv=splits, max_iter=20000, selection="cyclic",
                      n_jobs=n_jobs).fit(X, y)
    mean, se = cv.mse_path_.mean(1), cv.mse_path_.std(1, ddof=1) / np.sqrt(cv.mse_path_.shape[1])
    i = int(np.argmin(mean))
    ok = np.flatnonzero(mean <= mean[i] + se[i])
    alpha = float(cv.alphas_[ok[np.argmax(cv.alphas_[ok])]])
    return ElasticNet(alpha=alpha, l1_ratio=L1_RATIO, max_iter=20000, selection="cyclic").fit(X, y)
