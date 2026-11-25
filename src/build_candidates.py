"""
Link the official 2023 candidate lists to (a) the MPs who were elected and (b) the #Secim2023 Twitter accounts.

Inputs
  data/external/candidates_2023.csv            OCR'd from Resmî Gazete 19.04.2023, No. 32168 (Mükerrer)
                                               (see src/ocr/ for the OCR + parsing pipeline)
  data/external/mps_28th_term_wikipedia.csv    600 MPs elected on 14 May 2023
  data/processed/accounts.csv                  from src/build_dataset.py

Output
  data/processed/candidates_linked.csv   one row per candidate with list rank, seats won by the list,
                                         margin to the cutoff, elected flag and matched Twitter account id

Key idea (closed-list PR): a candidate is elected iff  rank <= seats won by their list in the district.
    margin = rank - list_seats   ->   margin <= 0 : elected,   margin >= 1 : not elected
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT, OUT = ROOT / "data/external", ROOT / "data/processed"

MAJOR_LISTS = {  # lists that won seats in 2023 (small allies ran on these lists)
    "ADALET VE KALKINMA PARTİSİ": "AKP", "CUMHURİYET HALK PARTİSİ": "CHP",
    "YEŞİLLER VE SOL GELECEK PARTİSİ": "YSGP", "MİLLİYETÇİ HAREKET PARTİSİ": "MHP",
    "İYİ PARTİ": "IYI", "YENİDEN REFAH PARTİSİ": "YRP", "TÜRKİYE İŞÇİ PARTİSİ": "TİP",
}


def key(s):
    s = str(s).replace("İ", "I").replace("ı", "i").upper()
    s = s.translate(str.maketrans("ÇĞÖŞÜÂÎÛ", "CGOSUAIU"))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z ]", "", s).split()


def same_person(short, full):
    """True if every token of the commonly used name appears, in order, in the official full name
    (official lists use full legal names, e.g. 'YILDIRIM TUĞRUL TÜRKEŞ' for 'Tuğrul Türkeş')."""
    a, b = key(short), "".join(key(full))
    if not a or not b.endswith(a[-1]):
        return False
    pos = 0
    for t in a:
        pos = b.find(t, pos)
        if pos < 0:
            return False
        pos += len(t)
    return True


def match(names_df, name_col, cands, district_col=None):
    """Return candidate index for each row in names_df (None if no unique match)."""
    by_surname = {}
    for i, full in cands["full_name"].items():
        k = key(full)
        if k:
            by_surname.setdefault(k[-1], []).append(i)
    out = []
    for _, r in names_df.iterrows():
        k = key(r[name_col])
        pool = [i for i in by_surname.get(k[-1] if k else "", []) if same_person(r[name_col], cands.at[i, "full_name"])]
        if district_col and len(pool) > 1:
            same_d = [i for i in pool if cands.at[i, "district"] == r[district_col]]
            pool = same_d or pool
        if len(pool) > 1:  # prefer a list that won seats; drop if still ambiguous
            major = [i for i in pool if cands.at[i, "party"] in MAJOR_LISTS]
            pool = major if len(major) == 1 else pool
        out.append(pool[0] if len(pool) == 1 else None)
    return out


def main():
    cands = pd.read_csv(EXT / "candidates_2023.csv")
    cands["education"] = cands["education"].replace({"YUKSEK": "YÜKSEK", "ILK": "İLK"})
    mps = pd.read_csv(EXT / "mps_28th_term_wikipedia.csv")
    accounts = pd.read_csv(OUT / "accounts.csv", dtype={"id": str})

    # 1) elected MPs -> candidate rows (same district)
    mps["cand_idx"] = match(mps, "name", cands, district_col="province")
    cands["elected"] = cands.index.isin(mps["cand_idx"].dropna().astype(int))
    print(f"elected MPs found on the official lists: {mps.cand_idx.dropna().nunique()} / {len(mps)}")

    # 2) seats won per list and distance to the cutoff
    seats = cands[cands.elected].groupby(["district", "party"]).size().rename("list_seats")
    cands = cands.join(seats, on=["district", "party"]).fillna({"list_seats": 0})
    cands["margin"] = cands["rank"] - cands["list_seats"]
    consistent = ((cands.margin <= 0) == cands.elected).mean()
    print(f"rank-cutoff rule reproduces the elected flag for {consistent:.1%} of candidates")

    # 3) Twitter accounts -> candidates
    accounts["display"] = accounts["Full Name"].fillna(accounts["name"])
    accounts["cand_idx"] = match(accounts, "display", cands)
    linked = accounts.dropna(subset=["cand_idx"]).astype({"cand_idx": int})
    linked = linked.drop_duplicates("cand_idx", keep=False)  # two accounts -> one candidate: ambiguous, drop
    cands = cands.join(linked.set_index("cand_idx")[["id", "screen_name"]].rename(columns={"id": "twitter_id"}))
    cands["list"] = cands["party"].map(MAJOR_LISTS).fillna("other")
    cands.to_csv(OUT / "candidates_linked.csv", index_label="cand_idx")
    print(f"Twitter accounts linked to a 2023 candidacy: {cands.twitter_id.notna().sum()}")


if __name__ == "__main__":
    main()
