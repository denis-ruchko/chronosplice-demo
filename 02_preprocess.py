"""Step 2: samples, sample QC, gene counts and junction counts -> data/processed.

    python 02_preprocess.py --raw data/raw --out data/processed

Samples
  GEO     runs listed in metadata/geo_samples_li2025.csv (ages and sex from Li et al. 2025, Table S1) that
          are in recount3, in that file's order within each study.
  GTEx    whole blood (SMTSD) in the v8 release freeze (SMAFRZE == RNASEQ), one sample per subject (most mapped
          reads), sorted by subject. Age = midpoint of the public 10-year bin (25, 35, ... 75).
  QC      drop samples with < 3 M gene reads, or whose labelled sex disagrees with XIST vs RPS4Y1 expression.
Genes     read counts = round(base coverage / average mapped length), as recount3::compute_read_counts.
Junctions split-read counts. Kept: stranded junctions <= 100 kb with >= 5 reads in >= 5% of the samples of at
          least one study with >= 20 samples (a superset of any training-row filter used later), and every
          source contributes its counts for all kept junctions. Junction files are streamed twice.
Outputs   samples.csv, genes.csv.gz, gene_counts.npz, junctions.csv.gz, junction_counts.npz
          (count matrices are samples x features, rows in samples.csv order).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import clock

MIN_GENE_READS, MIN_STUDY = 3e6, 20
SHEET = Path(__file__).resolve().parent / "metadata" / "geo_samples_li2025.csv"


def gene_counts(gene_sums, qc, rail_ids, external_ids):
    q = clock.read_tsv(qc, dtype={"rail_id": str}).set_index("rail_id")
    mapped = q.loc[rail_ids, "star.average_mapped_length"].to_numpy(float)
    cov = clock.read_gene_sums(gene_sums)[list(external_ids)]
    return cov.index, np.round(cov.to_numpy(np.float64) / mapped).T.astype(np.float32)


def geo_samples(paths):
    sheet = pd.read_csv(SHEET)
    rows, mats = [], []
    for gse, srp in clock.GEO_STUDIES.items():
        meta = clock.read_tsv(paths[srp]["meta"], dtype={"rail_id": str})[["rail_id", "external_id"]]
        s = sheet[sheet["Dataset"] == gse].merge(meta, left_on="Run", right_on="external_id")
        print(f"{gse} ({srp}): {len(s)} of {(sheet['Dataset'] == gse).sum()} listed runs in recount3", flush=True)
        gid, M = gene_counts(paths[srp]["gene_sums"], paths[srp]["qc"], s["rail_id"], s["external_id"])
        rows.append(pd.DataFrame({"rail_id": s["rail_id"], "run": s["Run"], "study": gse, "source": srp,
                                  "age": s["Age"].astype(float), "age_binned": False, "sex": s["Sex"].str.lower()}))
        mats.append(M)
    return pd.concat(rows, ignore_index=True), np.vstack(mats), gid


def gtex_samples(paths):
    m = clock.read_tsv(paths["GTEX"]["meta"], dtype={"rail_id": str})
    q = clock.read_tsv(paths["GTEX"]["qc"], dtype={"rail_id": str}).set_index("rail_id")
    m = m[(m["SMTSD"] == "Whole Blood") & (m["SMAFRZE"] == "RNASEQ")].copy()
    m["mapped_reads"] = q.loc[m["rail_id"], "star.all_mapped_reads"].to_numpy()
    m = m.sort_values(["SUBJID", "mapped_reads"], ascending=[True, False]).drop_duplicates("SUBJID").reset_index(drop=True)
    s = pd.DataFrame({"rail_id": m["rail_id"], "run": m["external_id"], "study": "GTEx", "source": "GTEX_BLOOD",
                      "age": m["AGE"].str.split("-").str[0].astype(int) + 5.0, "age_binned": True,
                      "sex": m["SEX"].map({1: "male", 2: "female"}), "death_hardy": m["DTHHRDY"], "rin": m["SMRIN"],
                      "ischemic_min": m["SMTSISCH"], "centre_c1": (m["SMCENTER"] == "C1").astype(float)})
    gid, M = gene_counts(paths["GTEX"]["gene_sums"], paths["GTEX"]["qc"], s["rail_id"], s["run"])
    print(f"GTEx: {len(s)} subjects", flush=True)
    return s, M, gid


def sample_qc(s, G, genes):
    lib = G.sum(1)
    cpm = lambda name: G[:, np.flatnonzero(genes["gene_name"].to_numpy() == name)].sum(1) / lib * 1e6
    predicted = np.where(np.log2(cpm("RPS4Y1") + 1) > np.log2(cpm("XIST") + 1), "male", "female")
    low = lib < MIN_GENE_READS
    mismatch = s["sex"].isin(["male", "female"]).to_numpy() & (s["sex"].to_numpy() != predicted)
    return ~low & ~mismatch, low, mismatch


# ----------------------------------------------------------------------------- junctions (two streaming passes)
def source_columns(id_file, rail_ids):
    """MM column (1-based) -> position among `rail_ids`, -1 for samples not used."""
    ids = clock.read_tsv(id_file, dtype={"rail_id": str})["rail_id"].tolist()
    col = {r: k for k, r in enumerate(ids)}
    want = np.full(len(ids) + 1, -1, np.int32)
    for k, r in enumerate(rail_ids):
        want[col[r] + 1] = k
    return want


def coverage_pass(p, rail_ids):
    """Keys of the junctions with >= 5 reads in >= 5% of this source's samples (stranded, <= 100 kb)."""
    want = source_columns(p["ID"], rail_ids)
    skip, (n_rows, _, _) = clock.mm_header(p["MM"])
    hit = np.zeros(n_rows + 1, np.int64)
    for c in clock.mm_chunks(p["MM"], skip):
        ok = (want[c["j"].to_numpy()] >= 0) & (c["v"].to_numpy() >= clock.JX_MIN_READS)
        hit += np.bincount(c["i"].to_numpy()[ok], minlength=n_rows + 1)
    rr = clock.read_tsv(p["RR"], dtype={"chromosome": str}).rename(columns={"chromosome": "chrom"})
    usable = rr["strand"].isin(["+", "-"]).to_numpy() & ((rr["end"] - rr["start"]).abs() <= clock.MAX_INTRON).to_numpy()
    keep = (hit[1:] >= clock.JX_MIN_FRAC * len(rail_ids)) & usable          # RR row k = MM row k + 1
    return clock.junction_key(rr[keep])


def count_pass(p, rail_ids, column_of_key, n_cols):
    """samples x union-junction counts for this source, built chunk by chunk."""
    want = source_columns(p["ID"], rail_ids)
    rr = clock.read_tsv(p["RR"], dtype={"chromosome": str}).rename(columns={"chromosome": "chrom"})
    newcol = np.r_[-1, column_of_key.reindex(clock.junction_key(rr)).fillna(-1).to_numpy(np.int64)]
    skip, _ = clock.mm_header(p["MM"])
    M = sp.csr_matrix((len(rail_ids), n_cols), dtype=np.float32)
    for c in clock.mm_chunks(p["MM"], skip):
        i, j = c["i"].to_numpy(), want[c["j"].to_numpy()]
        ok = (j >= 0) & (newcol[i] >= 0)
        M = M + sp.csr_matrix((c["v"].to_numpy()[ok], (j[ok], newcol[i[ok]])), shape=M.shape, dtype=np.float32)
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = clock.raw_paths(a.raw)

    s_geo, G_geo, gid = geo_samples(paths)
    s_gtex, G_gtex, gid2 = gtex_samples(paths)
    if not gid.equals(gid2):
        raise RuntimeError("GEO and GTEx gene orders differ")
    genes = clock.read_gtf_genes(paths["GTF"]).set_index("gene_id").reindex(gid).reset_index()
    s = pd.concat([s_geo, s_gtex], ignore_index=True)
    G = np.vstack([G_geo, G_gtex])
    ok, low, mismatch = sample_qc(s, G, genes)
    for name, m in (("low depth", low), ("sex mismatch", mismatch)):
        print(f"QC {name}: {s.loc[m, 'study'].value_counts().to_dict()}")
    s, G = s[ok].reset_index(drop=True), G[ok]
    s["male"] = (s["sex"] == "male").astype(float)
    s["gene_reads"] = G.sum(1)
    s["log_depth"] = np.log(s["gene_reads"])
    print("samples after QC:", s["study"].value_counts().to_dict(), flush=True)

    sources = [(gse, paths[srp]) for gse, srp in clock.GEO_STUDIES.items()] + [("GTEx", paths["GTEX"])]
    keys = []
    for name, p in sources:
        rid = s.loc[s["study"] == name, "rail_id"].tolist()
        if len(rid) >= MIN_STUDY:
            keys.append(coverage_pass(p, rid))
            print(f"junctions passing coverage in {name}: {len(keys[-1]):,}", flush=True)
    union = pd.Index(np.concatenate(keys)).unique().sort_values()
    column_of_key = pd.Series(np.arange(len(union)), index=union)
    blocks = []
    for name, p in sources:
        rid = s.loc[s["study"] == name, "rail_id"].tolist()
        blocks.append(count_pass(p, rid, column_of_key, len(union)))
        print(f"junction counts loaded for {name}: {blocks[-1].nnz:,} non-zeros", flush=True)
    J = sp.vstack(blocks, format="csr")
    parts = union.str.extract(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+):(?P<strand>.)$")
    junctions = parts.assign(start=parts["start"].astype(int), end=parts["end"].astype(int))

    s.to_csv(out / "samples.csv", index=False)
    genes.to_csv(out / "genes.csv.gz", index=False)
    sp.save_npz(out / "gene_counts.npz", sp.csr_matrix(G))
    junctions.to_csv(out / "junctions.csv.gz", index=False)
    sp.save_npz(out / "junction_counts.npz", J)
    print(f"done: {len(s)} samples, {G.shape[1]:,} genes, {J.shape[1]:,} junctions -> {out}")


if __name__ == "__main__":
    main()
