"""
Build analysis-ready tables from the raw #Secim2023 profile snapshots.

Inputs
  data/raw/profiles/account_statistics_YYYY-MM.jsons.gz   (Harvard Dataverse, doi:10.7910/DVN/QJA1ZW)
  data/raw/TwitterAccountList.csv                          (ViralLab/Secim2023_Dataset on GitHub)
  data/external/mps_28th_term_wikipedia.csv                (MPs elected on 14 May 2023, Turkish Wikipedia)

Outputs
  data/processed/daily_panel.csv.gz      one row per (account, day): followers, friends, tweets, listed, favourites
  data/processed/accounts.csv            one row per political account with party, role, election outcome
  data/processed/account_features.csv    per-account growth features for the hypothesis tests

Run:  python src/build_dataset.py
"""
import glob
import gzip
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, EXT, OUT = ROOT / "data/raw", ROOT / "data/external", ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)

# Key dates of the 2023 Turkish elections
EARTHQUAKE = pd.Timestamp("2023-02-06")
CAMPAIGN_START = pd.Timestamp("2023-03-10")  # Erdoğan signs the election decree
ELECTION_R1 = pd.Timestamp("2023-05-14")
ELECTION_R2 = pd.Timestamp("2023-05-28")

PARTY_FIX = {  # unify spelling variants in the account list
    "GELECEK": "GELECEK", "ZAFER": "ZAFER", "MEMLEKET": "MEMLEKET",
}
ALLIANCE = {  # alliances as registered for the 14 May 2023 election
    "AKP": "Cumhur", "MHP": "Cumhur", "BBP": "Cumhur", "YRP": "Cumhur",
    "CHP": "Millet", "IYI": "Millet", "DEVA": "Millet", "GELECEK": "Millet",
    "SAADET": "Millet", "DP": "Millet",
    "HDP": "Emek ve Özgürlük", "TİP": "Emek ve Özgürlük",
    "ZAFER": "ATA", "MEMLEKET": "Other",
}


def norm(s: str) -> str:
    """Lower-case, strip Turkish diacritics and punctuation for fuzzy-safe name matching."""
    if not isinstance(s, str):
        return ""
    s = s.replace("İ", "i").replace("I", "ı").lower()
    s = s.translate(str.maketrans("çğıöşüâîû", "cgiosuaiu"))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", s).strip()


def load_profiles():
    rows, meta = [], {}
    for path in sorted(glob.glob(str(RAW / "profiles" / "*.jsons.gz"))):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                meta[rec["id"]] = (rec["name"], rec["screen_name"], rec["created_at"])
                for day, s in rec["stats"].items():
                    rows.append({"id": rec["id"], "date": day, **s})
    daily = (pd.DataFrame(rows)
             .assign(date=lambda d: pd.to_datetime(d["date"]))
             .drop_duplicates(["id", "date"])
             .sort_values(["id", "date"]))
    meta = pd.DataFrame([(k, *v) for k, v in meta.items()],
                        columns=["id", "name", "screen_name", "created_at"])
    meta["created_at"] = pd.to_datetime(meta["created_at"], format="%a %b %d %H:%M:%S +0000 %Y")
    return daily, meta


def build_accounts(meta):
    acc = pd.read_csv(RAW / "TwitterAccountList.csv")
    acc["sn"] = acc["Twitter"].str.rstrip("/").str.split("/").str[-1].str.lower()
    acc = acc.drop_duplicates("sn")
    acc["party"] = acc["Party Short"].replace(PARTY_FIX)
    notes = acc["Notes"].fillna("")
    acc["role"] = np.select(
        [notes.str.contains("National|Local", case=False),
         notes.str.contains("Başkan|Sözcü|Genel Sekreter", case=False),
         acc["party"].notna() & (notes != "")],
        ["media", "party_executive", "district_politician"], default="other_politician")
    acc["district"] = np.where(acc["role"] == "district_politician", notes, None)

    m = meta.assign(sn=meta["screen_name"].str.lower()).merge(
        acc[["sn", "Full Name", "party", "role", "district"]], on="sn", how="left")
    m["alliance"] = m["party"].map(ALLIANCE)

    mps = pd.read_csv(EXT / "mps_28th_term_wikipedia.csv")
    elected = set(mps["name"].map(norm))
    m["elected_2023"] = (m["Full Name"].map(norm).isin(elected)
                         | m["name"].map(norm).isin(elected))
    m["account_age_years"] = (ELECTION_R1 - m["created_at"].dt.tz_localize(None)).dt.days / 365.25
    return m


def growth(daily, start, end):
    """Relative follower growth of each account between the snapshots closest to start/end."""
    a = daily[daily["date"].between(start - pd.Timedelta(days=7), start)].groupby("id").last()
    b = daily[daily["date"].between(end - pd.Timedelta(days=7), end)].groupby("id").last()
    j = a[["followers", "tweets"]].join(b[["followers", "tweets"]], lsuffix="_0", rsuffix="_1", how="inner")
    return pd.DataFrame({
        "followers_start": j["followers_0"],
        "follower_growth_pct": 100 * (j["followers_1"] - j["followers_0"]) / j["followers_0"].clip(lower=1),
        "tweets_in_period": j["tweets_1"] - j["tweets_0"],
    })


def main():
    daily, meta = load_profiles()
    accounts = build_accounts(meta)

    camp = growth(daily, CAMPAIGN_START, ELECTION_R1).add_prefix("campaign_")
    base = growth(daily, pd.Timestamp("2022-12-01"), pd.Timestamp("2023-02-05")).add_prefix("baseline_")
    feats = accounts.set_index("id").join([camp, base], how="left")

    daily.to_csv(OUT / "daily_panel.csv.gz", index=False, compression="gzip")
    accounts.to_csv(OUT / "accounts.csv", index=False)
    feats.reset_index().to_csv(OUT / "account_features.csv", index=False)
    print(f"daily rows: {len(daily):,} | accounts: {len(accounts):,} | "
          f"party-matched: {accounts['party'].notna().sum():,} | elected 2023: {accounts['elected_2023'].sum():,}")


if __name__ == "__main__":
    main()
