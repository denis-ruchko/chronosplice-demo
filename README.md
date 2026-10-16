# ChronoSplice demo

Blood transcriptomic age clocks built from RNA-seq in [recount3](https://rna.recount.bio/):
an elastic net on gene expression (log-CPM), and the same model with LeafCutter splicing (PSI) added.

## Data

| study | recount3 project | samples | ages |
|---|---|---|---|
| GSE94438 | SRP098758 | 327 listed | 8-60 |
| GSE79362 | SRP071965 | 245 | 13-18 |
| GSE134080 | SRP214077 | 100 | 18-73 |
| GSE122485 | SRP168555 | 3 | 25-35 |
| GTEx v8 whole blood | GTEx `BLOOD` | 755 subjects | 20-79 |

Ages and sex for the GEO samples come from Li et al. 2025, Table S1 (`metadata/geo_samples_li2025.csv`).
GTEx ages are the bin midpoints (25, 35, ... 75); exact GTEx ages are access-controlled.

## Run

```bash
pip install -r requirements.txt
python 01_download.py   --raw data/raw                         # about 1.3 GB from recount3
python 02_preprocess.py --raw data/raw --out data/processed    # samples, QC, gene and junction counts
python 03_train.py      --data data/processed --out results    # leave-one-study-out + final models
```

```
clock.py            shared functions (recount3 I/O, normalisation, LeafCutter, elastic net)
01_download.py      recount3 downloads
02_preprocess.py    samples, QC, gene and junction counts
03_train.py         leave-one-study-out evaluation and final models
metadata/           GEO sample sheet (ages, sex)
```

## What each step does

**01_download.py.** Downloads recount3 metadata, QC tables, gene coverage sums and junction counts for the
four GEO studies and GTEx whole blood, plus the GENCODE v26 annotation recount3 uses.

**02_preprocess.py.**
- **Samples:** GEO runs from the sample sheet that are in recount3. For GTEx, whole blood in the v8
  release freeze, one sample per subject.
- **Sample QC:** drops samples with fewer than 3 M gene reads, or whose labelled sex disagrees with
  XIST/RPS4Y1 expression.
- **Gene counts:** coverage divided by mean mapped read length, as in `recount3::compute_read_counts`.
- **Junction counts:**
  - Kept: stranded junctions up to 100 kb with ≥ 5 reads in ≥ 5% of the samples of at least one study
    (≥ 20 samples).
  - Every study contributes its counts for every kept junction, so splicing ratios are comparable between
    studies.

**03_train.py.** Every fitted quantity uses training rows only.

- **Evaluation:** leave-one-study-out, holding out GSE94438, GSE79362, GSE134080 and GTEx in turn. Within
  each training set:
  - **Genes:**
    - protein-coding and lncRNA genes, excluding globin, chrM, chrY and rRNA, with CPM ≥ 1 in ≥ 25% of
      samples;
    - edgeR-style TMM log-CPM;
    - for GTEx rows, RIN, ischemic time, centre, depth and sex regressed out, with age-bin terms in the model
      so the age effect is kept;
    - genes kept only if their age correlation has the same sign in GTEx and GSE94438;
    - genes dropped if they differ by more than 1 log2 unit between the two studies at matched ages (20-39);
    - top 5,000 genes by variance, z-scored.
  - **LeafCutter splicing:**
    - introns sharing a donor or acceptor are clustered;
    - introns with < 0.1% of their cluster's reads are dropped and the clusters re-formed;
    - clusters need ≥ 2 introns and ≥ 30 reads;
    - PSI = intron reads / cluster reads (missing when the cluster has < 20 reads, then filled with the
      training mean);
    - top 5,000 introns by variance, z-scored.
  - **Model:** elastic net (l1_ratio 0.5) on Horvath-transformed age. Alpha is the largest within one standard
    error of the best 5-fold CV error, with folds stratified by study × age bin.
- **Final models:** both models refitted on all samples and saved to `results/` with everything needed to
  score new samples (gene IDs, the TMM reference, scalers, intron clusters, coefficients).

## Outputs (`results/`)
- `loso_metrics.csv`: MAE, r, slope and bias, plus MAE under 30, 30-49 and 50+, per held-out study and pooled
  over the GEO studies.
- `loso_predictions.csv`: held-out predictions.
- `model_gene.npz`, `model_gene_leafcutter.npz`: the final models.
- `model_gene_coefficients.csv`, `model_gene_leafcutter_coefficients.csv`: their non-zero coefficients.

## Limits
- The GEO studies are mostly young (GSE79362 is 13-18; GSE94438 is mostly under 35), so ages over 50 come mainly from GTEx, which is postmortem blood with binned ages.
- Held-out GTEx MAE is measured against bin midpoints.

## References
- Wilks C. et al. recount3: summaries and queries for large-scale RNA-seq expression and splicing. *Genome Biol* 2021.
- Li Y.I. et al. Annotation-free quantification of RNA splicing using LeafCutter. *Nat Genet* 2018.
- Horvath S. DNA methylation age of human tissues and cell types. *Genome Biol* 2013.
- GTEx Consortium. The GTEx Consortium atlas of genetic regulatory effects across human tissues. *Science* 2020.
- Li et al. 2025, Table S1: source of the GEO sample ages (add the full citation).
