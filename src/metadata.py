from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw_pdf"
METADATA_DIR = BASE_DIR / "data" / "metadata"

METADATA_DIR.mkdir(parents=True, exist_ok=True)

docs = []

for filename in RAW_DIR.iterdir():

    if filename.suffix != ".pdf":
        continue

    name = filename.name

    qb_match = re.match(r"quartbul_(\d{4})_(\d)_komplett\.(\w+)\.pdf", name)
    if qb_match:
        year = int(qb_match.group(1))
        issue = int(qb_match.group(2))
        lang = qb_match.group(3).upper()

        docs.append({
            "doc_id": f"QB_{year}_Q{issue}_{lang}",
            "doc_type": "QB",
            "year": year,
            "issue": issue,
            "language": lang,
            "local_path": str(filename),
            "title": f"Quarterly Bulletin {issue}/{year}",
        })
        continue

    fsr_match = re.match(r"stabrep_(\d{4})\.(\w+)\.pdf", name)
    if fsr_match:
        year = int(fsr_match.group(1))
        lang = fsr_match.group(2).upper()

        docs.append({
            "doc_id": f"FSR_{year}_{lang}",
            "doc_type": "FSR",
            "year": year,
            "issue": None,
            "language": lang,
            "local_path": str(filename),
            "title": f"Financial Stability Report {year}",
        })

df = pd.DataFrame(docs)

df.to_csv(METADATA_DIR / "docs.csv", index=False)

print("Saved to:", METADATA_DIR / "docs.csv")