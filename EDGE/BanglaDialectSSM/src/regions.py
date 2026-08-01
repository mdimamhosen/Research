"""
regions.py
----------
Dialect AREA / region labels for BanglaDialectSSM.

Each training row can say where the dialect comes from, e.g.:
  sylheti, chittagonian, barishal, noakhali, ...

Format in pairs.tsv (3 columns):
  dialect_sentence <TAB> standard_sentence <TAB> area
"""

from __future__ import annotations

# Canonical ids used in code + TSV (lowercase, underscore-free or simple)
AREAS: list[str] = [
    "unspecified",   # 0 — unknown / not labeled yet
    "standard",      # 1 — standard Bangla (for the standard side)
    "sylheti",       # 2 — সিলেটি / Sylhetia
    "chittagonian",  # 3 — চাটগাঁইয়া / Chittagong
    "barishal",      # 4 — বরিশালিয়া / Barishal
    "noakhali",      # 5 — নোয়াখাইল্যা
    "mymensingh",    # 6 — ময়মনসিংহ
    "rangpur",       # 7 — রংপুর
    "rajshahi",      # 8 — রাজশাহী
    "khulna",        # 9
    "dhaka",         # 10 — ঢাকায়া / Dhaka regional
    "tangail",       # 11
    "kishoreganj",   # 12
    "narail",        # 13
    "narsingdi",     # 14
]

# Friendly spellings people might type -> canonical id
ALIASES: dict[str, str] = {
    "unknown": "unspecified",
    "none": "unspecified",
    "": "unspecified",
    "sylhet": "sylheti",
    "sylhetia": "sylheti",
    "sylhetiya": "sylheti",
    "siloti": "sylheti",
    "chittagong": "chittagonian",
    "chatgaiya": "chittagonian",
    "ctg": "chittagonian",
    "barisal": "barishal",
    "barishalia": "barishal",
    "barishaliya": "barishal",
    "borishailla": "barishal",
    "noakhailla": "noakhali",
    "mymensing": "mymensingh",
    "std": "standard",
    "bangla": "standard",
    "standard_bangla": "standard",
}


def normalize_area(name: str) -> str:
    """Map any user spelling to a canonical area id."""
    key = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    if key in ALIASES:
        key = ALIASES[key]
    if key not in AREAS:
        return "unspecified"
    return key


def area_to_id(name: str) -> int:
    return AREAS.index(normalize_area(name))


def id_to_area(area_id: int) -> str:
    if 0 <= area_id < len(AREAS):
        return AREAS[area_id]
    return "unspecified"


def num_areas() -> int:
    return len(AREAS)
