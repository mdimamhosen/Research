#!/usr/bin/env python3
"""Batch CLI: scanned Bengali PDF → UTF-8 .txt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from src.pipeline import (
    ENGINES,
    discover_pdfs,
    ocr_pdf_to_file,
    resolve_default_engine,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OCR scanned Bengali book PDFs to UTF-8 .txt files."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=ROOT / "data" / "pdfs",
        help="PDF file or directory of PDFs (default: data/pdfs)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=ROOT / "data" / "txt",
        help="Output directory for .txt files (default: data/txt)",
    )
    parser.add_argument(
        "--engine",
        "-e",
        choices=ENGINES,
        default=None,
        help="OCR engine (default: openai if key set, else claude, else tesseract)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI (default: 300)")
    parser.add_argument("--page-start", type=int, default=None, help="First page (1-based)")
    parser.add_argument("--page-end", type=int, default=None, help="Last page (1-based)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .txt files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = args.engine or resolve_default_engine()

    try:
        pdfs = discover_pdfs(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not pdfs:
        print(f"No PDFs found under {args.input}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Engine: {engine} | DPI: {args.dpi} | PDFs: {len(pdfs)}")

    for pdf in pdfs:
        out = args.output / f"{pdf.stem}.txt"
        if out.exists() and not args.force:
            print(f"Skip (exists): {out.name}")
            continue

        print(f"Processing: {pdf.name} → {out.name}")
        with tqdm(desc=pdf.stem, unit="page") as bar:

            def on_progress(page_number: int, idx: int, total: int) -> None:
                bar.total = total
                bar.n = idx
                bar.set_postfix(page=page_number)
                bar.refresh()

            try:
                ocr_pdf_to_file(
                    pdf,
                    out,
                    engine=engine,
                    dpi=args.dpi,
                    page_start=args.page_start,
                    page_end=args.page_end,
                    on_progress=on_progress,
                    force=True,
                )
            except Exception as exc:  # noqa: BLE001 — surface to user, continue batch
                print(f"\nFailed {pdf.name}: {exc}", file=sys.stderr)
                continue

        print(f"Wrote: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
