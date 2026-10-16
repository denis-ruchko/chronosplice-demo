"""Step 1: download the recount3 files (about 1.3 GB in total; existing files are skipped).

    python 01_download.py --raw data/raw

GEO studies (recount3 SRA projects): GSE94438 SRP098758, GSE79362 SRP071965, GSE134080 SRP214077,
GSE122485 SRP168555 -- metadata, QC, gene sums, junctions. GTEx v8 whole blood (recount3 GTEx 'BLOOD')
-- the same files. Plus the GENCODE v26 gene annotation used by recount3 (G026).
"""
import argparse

import clock


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    a = ap.parse_args()
    paths = clock.raw_paths(a.raw)
    jobs = [(clock.GTF_URL, paths["GTF"])]
    for gse, srp in clock.GEO_STUDIES.items():
        jobs += [(u, paths[srp][k]) for k, u in clock.geo_urls(srp).items()]
    jobs += [(u, paths["GTEX"][k]) for k, u in clock.GTEX_URLS.items()]
    for url, dest in jobs:
        clock.download(url, dest)
        print(f"{dest}  {dest.stat().st_size / 1e6:8.1f} MB", flush=True)


if __name__ == "__main__":
    main()
