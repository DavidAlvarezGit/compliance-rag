from pathlib import Path
import re
import pandas as pd
import fitz  # pymupdf

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw_pdf"
METADATA_DIR = BASE_DIR / "data" / "metadata"

METADATA_DIR.mkdir(parents=True, exist_ok=True)

TOPIC_PATTERNS = [
    ("credit_risk_standardized_approach", [r"credit risk standardized approach", r"credit risk standardised approach", r"\bsa[- ]?cr\b", r"approche standard.*risque de cr[ée]dit"]),
    ("irb_framework", [r"internal ratings[- ]based", r"\birb\b", r"notations? internes?"]),
    ("liquidity_coverage_ratio_lcr", [r"liquidity coverage ratio", r"\blcr\b"]),
    ("net_stable_funding_ratio_nsfr", [r"net stable funding ratio", r"\bnsfr\b"]),
    ("leverage_ratio_rules", [r"leverage ratio rules", r"leverage ratio", r"ratio de levier"]),
    ("operational_risk_framework", [r"operational risk framework", r"operational risk", r"risques? op[ée]rationnels?", r"r[ée]silience op[ée]rationnelle"]),
    ("capital_requirements_framework", [r"capital requirements framework", r"capital requirements", r"fonds propres", r"capital adequacy", r"basel framework"]),
    ("corporate_governance_internal_controls", [r"corporate governance", r"governance d.?entreprise", r"contr[oô]les? internes?"]),
    ("market_conduct_rules", [r"securities trading", r"r[èe]gles? de conduite sur le march[ée]", r"market conduct"]),
    ("climate_nature_related_financial_risks", [r"climate.*nature[- ]related financial risks", r"risques financiers li[ée]s?.*nature", r"climate"]),
    ("liquidity_risk_management", [r"liquidity management", r"risque de liquidit[ée]", r"gestion du risque de liquidit[ée]"]),
]


def normalize_text(s: str) -> str:
    return " ".join(s.lower().replace("_", " ").replace("-", " ").split())


def sample_pdf_text(pdf_path: Path, max_pages: int = 6) -> str:
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return ""
    parts = []
    for i in range(min(max_pages, len(doc))):
        try:
            parts.append(doc.load_page(i).get_text("text"))
        except Exception:
            continue
    return normalize_text(" ".join(parts))


def infer_topic(name_norm: str, text_norm: str) -> str:
    # Prefer deterministic filename cues when available.
    filename_overrides = [
        ("capital_requirements_framework", ["capital requirements framework"]),
        ("credit_risk_standardized_approach", ["credit risk standardized approach", "credit risk standardised approach", "sa-cr"]),
        ("irb_framework", ["irb framework", "internal ratings based"]),
        ("operational_risk_framework", ["operational risks resilience", "operational risk framework"]),
        ("liquidity_coverage_ratio_lcr", ["liquidity coverage ratio", "lcr"]),
        ("net_stable_funding_ratio_nsfr", ["net stable funding ratio", "nsfr"]),
        ("leverage_ratio_rules", ["leverage ratio rules", "leverage ratio"]),
        ("corporate_governance_internal_controls", ["corporate governance", "internal controls"]),
        ("market_conduct_rules", ["securities trading", "market conduct"]),
        ("climate_nature_related_financial_risks", ["climate", "nature related financial risks"]),
        ("liquidity_risk_management", ["liquidity management", "risk management reporting"]),
    ]
    for topic_id, hints in filename_overrides:
        if all(h in name_norm for h in hints):
            return topic_id
        if any(h in name_norm for h in hints):
            return topic_id

    combined = f"{name_norm} {text_norm}".strip()
    for topic_id, patterns in TOPIC_PATTERNS:
        for pat in patterns:
            if re.search(pat, combined):
                return topic_id
    return "other"


def infer_year(name_norm: str, text_norm: str) -> int:
    m = re.search(r"\b(20\d{2})\b", name_norm)
    if m:
        return int(m.group(1))
    m = re.search(r"circulaire\s*(20\d{2})\s*/\s*\d+", text_norm)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(20\d{2})\b", text_norm)
    return int(m.group(1)) if m else 0


def infer_language(name_norm: str, text_norm: str) -> str:
    if "_en" in name_norm:
        return "EN"
    fr_hits = sum(int(w in text_norm) for w in ["circulaire", "risque", "liquidité", "liquidite", "banques", "gouvernance"])
    en_hits = sum(int(w in text_norm) for w in ["framework", "capital", "requirements", "committee", "banking"])
    return "FR" if fr_hits >= en_hits else "EN"


docs = []

for filename in sorted(RAW_DIR.iterdir()):
    if filename.suffix.lower() != ".pdf":
        continue

    name_norm = normalize_text(filename.name)
    text_norm = sample_pdf_text(filename)

    topic = infer_topic(name_norm=name_norm, text_norm=text_norm)
    year = infer_year(name_norm=name_norm, text_norm=text_norm)
    language = infer_language(name_norm=name_norm, text_norm=text_norm)

    docs.append(
        {
            "doc_id": f"REG_BANK_{topic}_{year}_{filename.stem}",
            "doc_type": "REG_BANK",
            "topic": topic,
            "year": year,
            "issue": None,
            "language": language,
            "local_path": str(filename),
            "title": filename.stem.replace("_", " ").replace("-", " ").strip(),
        }
    )

df = pd.DataFrame(docs)
df.to_csv(METADATA_DIR / "docs.csv", index=False)

print("Saved to:", METADATA_DIR / "docs.csv")
print("Total documents:", len(df))
