"""Step 1 of the OCR pipeline (see run_ocr.sh): parse tesseract TSV output of the Resmî Gazete 19.04.2023 (No. 32168 Mükerrer) candidate lists into a table."""
import csv
import glob
import re
import sys

import pandas as pd

EDU = {"YÜKSEK", "ORTA", "İLK", "ILK", "YUKSEK", "LİSANS", "ÖNLİSANS"}


def squash(s):
    return re.sub(r"[^A-ZÇĞİÖŞÜÂÎÛ0-9.]", "", s.upper())


def page_lines(path):
    d = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, keep_default_na=False, dtype={"text": str})
    d = d[(d.level == 5) & (d.text.str.strip() != "")]
    out = []
    for _, g in d.groupby(["block_num", "par_num", "line_num"]):
        g = g.sort_values("left")
        out.append((g.top.min(), g.left.min(), list(zip(g.text, g.left))))
    return sorted(out)


def main(tsv_dir, out_csv):
    rows, district, party, page_no = [], None, None, None
    for path in sorted(glob.glob(f"{tsv_dir}/*.tsv")):
        page_no = int(re.search(r"(\d+)\.tsv", path).group(1))
        for top, left, words in page_lines(path):
            text = " ".join(w for w, _ in words)
            sq = squash(text)
            if "SEÇİMÇEVRESİ" in sq:
                m = re.search(r"([A-ZÇĞİÖŞÜÂÎ]+?)İLİ(\d*)?(?:NUMARALI|NOLU|NO\.?LU)?(?:\.?BÖLGE)?SEÇİMÇEVRESİ", sq)
                if m:
                    district = (m.group(1) + (f"-{m.group(2)}" if m.group(2) else ""))
                else:
                    district = "?" + sq[:40]
                continue
            if "KESİNADAYLİSTESİ" in sq or "RESMİGAZETE" in sq or sq.startswith("S.NO") or sq.startswith("SNO"):
                continue
            first = words[0][0].rstrip(".")
            if left > 500 and not first.isdigit():
                if top < 300 or len(sq) < 3:
                    continue
                party = text.strip(" |©—-")
                continue
            has_edu = any(w.upper() in EDU for w, _ in words)
            if left < 300 and (first.isdigit() or has_edu):
                toks = [w for w, _ in (words[1:] if first.isdigit() else words) if re.search(r"\w", w)]
                toks = [t for t in toks if not (len(t) <= 2 and not t.isalpha())]  # stray OCR marks
                ei = next((i for i, t in enumerate(toks) if t.upper() in EDU), None)
                if ei is None or ei == 0:
                    rows.append(dict(page=page_no, district=district, party=party, rank=int(first) if first.isdigit() else None,
                                     full_name=" ".join(toks), education=None, occupation=None, parse_ok=False))
                    continue
                rows.append(dict(page=page_no, district=district, party=party, rank=int(first) if first.isdigit() else None,
                                 full_name=" ".join(toks[:ei]), education=toks[ei].upper().replace("ILK", "İLK"),
                                 occupation=" ".join(toks[ei + 1:]), parse_ok=True))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    df = main(sys.argv[1], sys.argv[2])
    print(len(df), df.parse_ok.mean())
