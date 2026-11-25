"""Step 2 of the OCR pipeline (see run_ocr.sh): clean OCR rows -> canonical 87 districts, 24 parties + independents, repaired list ranks."""
import difflib
import re
import sys
import unicodedata

import pandas as pd


def asc(s):
    s = str(s).replace("İ", "I").replace("ı", "i").upper()
    s = s.translate(str.maketrans("ÇĞÖŞÜÂÎÛ", "CGOSUAIU"))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z0-9]", "", s)


raw, mps_csv, out = sys.argv[1:4]
d = pd.read_csv(raw)
canon = sorted(pd.read_csv(mps_csv).province.unique())
prov = {}  # ascii province -> list of canonical district names
for c in canon:
    base = re.sub(r"\s*\(.*\)", "", c)
    prov.setdefault(asc(base), []).append(c)
roman = {"1": "I", "2": "II", "3": "III"}

# --- districts: resolve once per raw header string, using page order to fill missing numbers
page_district, last = {}, None
for page, hdr in d.groupby("page").district.first().items():
    s = asc(hdr)
    head = s.split("SECIM")[0]
    hits = [p for p in prov if p in head]
    p = max(hits, key=len) if hits else None
    if p is None:
        page_district[page] = last
        continue
    opts = prov[p]
    if len(opts) == 1:
        res = opts[0]
    else:
        m = re.search(p + r"([123])", s)
        if m:
            res = f"{re.sub(r' .*', '', opts[0])} ({roman[m.group(1)]})"
        elif last in opts:
            res = last
        else:
            res = opts[0]
    page_district[page] = last = res
d["district"] = d.page.map(page_district)

# --- parties: canonical = frequent labels; others fuzzy-matched; non-party lines (occupation wraps) carried forward
counts = d.party.value_counts()
parties = [p for p, n in counts.items() if n >= 150]
fixed, cur = [], None
prev_raw = None
for raw_p in d.party:
    if raw_p != prev_raw:
        m = difflib.get_close_matches(str(raw_p), parties, n=1, cutoff=0.85)
        if m:
            cur = m[0]
        prev_raw = raw_p
    fixed.append(cur)
d["party"] = fixed

# --- ranks: trust OCR rank only if it continues the list plausibly, else previous + 1
d["rank_ocr"] = d["rank"]
fixed_rank, prev, key_prev = [], 0, None
for key, r in zip(zip(d.district, d.party), d["rank"]):
    if key != key_prev:
        prev, key_prev = 0, key
    r = int(r) if pd.notna(r) else None
    new = r if (r is not None and prev < r <= prev + 3) else prev + 1
    fixed_rank.append(new); prev = new
d["rank"] = fixed_rank
d = d.drop_duplicates(["district", "party", "rank"])
d.to_csv(out, index=False)
print(len(d), "rows |", d.district.nunique(), "districts |", d.party.nunique(), "parties")
print(d.groupby("party").size().sort_values(ascending=False).to_string())
