#!/usr/bin/env python3
"""Automated CLI: scanned Bengali PDFs → UTF-8 .txt via PaddleOCR (batch-ready)."""

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
            "Batch-OCR scanned Bengali book PDFs into UTF-8 .txt files "
            "using PaddleOCR-VL. Pass one PDF file OR a folder of PDFs — "
            "each PDF becomes its own .txt (same stem name)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python cli.py                         # all PDFs in data/pdfs/\n"
            "  python cli.py -i book.pdf             # one file\n"
            "  python cli.py -i data/pdfs --recursive\n"
            "  python cli.py -i data/pdfs --skip-existing\n"
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=ROOT / "data" / "pdfs",
        help="One PDF file, OR a directory containing PDFs (auto-detects 1..N)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=ROOT / "data" / "txt",
        help="Output directory — one .txt per PDF",
    )
    parser.add_argument(
        "--engine",
        "-e",
        choices=ENGINES,
        default=None,
        help="OCR engine (only paddle). Default from OCR_ENGINE env or paddle.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PDF render DPI")
    parser.add_argument("--page-start", type=int, default=None, help="First page (1-based)")
    parser.add_argument("--page-end", type=int, default=None, help="Last page (1-based)")
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="When -i is a folder, also find PDFs in subfolders",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip PDFs that already have output/<stem>.txt (resume-friendly)",
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
        pdfs = discover_pdfs(args.input, recursive=args.recursive)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not pdfs:
        print(
            f"No PDFs found under {args.input}\n"
            "Drop .pdf files into data/pdfs/ (or pass -i path/to/file.pdf).",
            file=sys.stderr,
        )
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Engine: {engine} | DPI: {args.dpi} | Discovered PDFs: {len(pdfs)}")
    for i, pdf in enumerate(pdfs, start=1):
        print(f"  [{i}/{len(pdfs)}] {pdf}")

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
