#!/usr/bin/env python3
"""Automated CLI: scanned Bengali PDFs → UTF-8 .txt (open-source OCR by default)."""

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
    OPEN_SOURCE_ENGINES,
    discover_pdfs,
    ocr_pdf_to_file,
    resolve_default_engine,
    unique_path,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description=(
            "Batch-OCR scanned Bengali book PDFs into UTF-8 .txt files. "
            "Default engines are open-source (tesseract, paddle)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=ROOT / "data" / "pdfs",
        help="PDF file or directory of PDFs",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=ROOT / "data" / "txt",
        help="Output directory for .txt files",
    )
    parser.add_argument(
        "--engine",
        "-e",
        choices=ENGINES,
        default=None,
        help=(
            "OCR engine. Open-source: "
            + ", ".join(OPEN_SOURCE_ENGINES)
            + ". Default from OCR_ENGINE env or tesseract."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300, help="PDF render DPI")
    parser.add_argument("--page-start", type=int, default=None, help="First page (1-based)")
    parser.add_argument("--page-end", type=int, default=None, help="Last page (1-based)")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip PDFs that already have output/<stem>.txt (resume-friendly automation)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output/<stem>.txt if it exists",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.force and args.skip_existing:
        print("Error: use only one of --force or --skip-existing", file=sys.stderr)
        return 2

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
    print(
        f"Engine: {engine} | DPI: {args.dpi} | PDFs: {len(pdfs)} | "
        f"open-source default: {engine in OPEN_SOURCE_ENGINES}"
    )

    ok = skipped = failed = 0

    for pdf in pdfs:
        exact = args.output / f"{pdf.stem}.txt"
        if args.skip_existing and exact.exists():
            print(f"Skip (exists): {exact.name}")
            skipped += 1
            continue

        if args.force:
            out = exact
        elif exact.exists():
            out = unique_path(args.output, pdf.stem, ".txt")
        else:
            out = exact

        print(f"Processing: {pdf.name} → {out.name}")
        with tqdm(desc=pdf.stem[:40], unit="page") as bar:

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
            except Exception as exc:  # noqa: BLE001
                print(f"\nFailed {pdf.name}: {exc}", file=sys.stderr)
                failed += 1
                continue

        print(f"Wrote: {out}")
        ok += 1

    print(f"\nDone. ok={ok} skipped={skipped} failed={failed}")
    return 1 if failed and not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
