"""
Download the #Secim2023 files used in this project from Harvard Dataverse (doi:10.7910/DVN/QJA1ZW).

    python src/download_data.py                 # monthly profile snapshots (~6 MB)  -> data/raw/profiles/
    python src/download_data.py --trends        # + city-level trending topics (~100 MB) -> data/raw/trends/

The tweet-ID files (~5 GB) and follower networks (~8 GB) are NOT needed for this project.
"""
import argparse
import json
import urllib.request
from pathlib import Path

DOI = "doi:10.7910/DVN/QJA1ZW"
API = "https://dataverse.harvard.edu/api"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def file_list():
    url = f"{API}/datasets/:persistentId/?persistentId={DOI}"
    with urllib.request.urlopen(url) as r:
        return json.load(r)["data"]["latestVersion"]["files"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trends", action="store_true", help="also download trending-topic files")
    args = ap.parse_args()
    folders = {"profiles"} | ({"trends"} if args.trends else set())

    for f in file_list():
        folder = f.get("directoryLabel", "")
        if folder not in folders:
            continue
        df = f["dataFile"]
        dest = RAW / folder / df["filename"]
        if dest.exists() and dest.stat().st_size == df["filesize"]:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {folder}/{df['filename']} ({df['filesize'] / 1e6:.1f} MB)")
        urllib.request.urlretrieve(f"{API}/access/datafile/{df['id']}", dest)


if __name__ == "__main__":
    main()
