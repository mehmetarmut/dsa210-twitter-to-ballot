#!/usr/bin/env bash
# Rebuild data/external/candidates_2023.csv from the official Resmî Gazete PDF.
# The PDF pages are scanned images, so they are OCR'd with tesseract (Turkish model).
# Requires: poppler-utils (pdftoppm), tesseract + tesseract-ocr-tur, python3 + pandas.
set -euo pipefail
cd "$(dirname "$0")/../.."
PDF=data/raw/resmi_gazete_20230419M1-1.pdf
[ -f "$PDF" ] || curl -L -o "$PDF" https://www.resmigazete.gov.tr/eskiler/2023/04/20230419M1-1.pdf
mkdir -p build/ocr_img build/ocr_tsv
pdftoppm -r 300 -gray -f 4 -l 425 -png "$PDF" build/ocr_img/p
ls build/ocr_img/*.png | xargs -P 4 -I{} sh -c 'tesseract {} build/ocr_tsv/$(basename {} .png) -l tur --psm 4 tsv >/dev/null 2>&1'
python3 src/ocr/parse_candidates.py build/ocr_tsv build/candidates_raw.csv
python3 src/ocr/clean_candidates.py build/candidates_raw.csv data/external/mps_28th_term_wikipedia.csv data/external/candidates_2023.csv
