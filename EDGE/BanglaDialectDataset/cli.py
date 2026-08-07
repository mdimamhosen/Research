#!/usr/bin/env python3
"""Automated CLI: scanned Bengali PDFs → UTF-8 .txt via PaddleOCR (batch-ready)."""

from __future__ import annotations

import argparse
import os
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
            "  python cli.py --fast --skip-existing\n"
            "  python cli.py -e tesseract --skip-existing\n"
            "  python cli.py -i data/pdfs --recursive\n"
            "  # Kaggle: no copy needed — point at dataset folder:\n"
            "  python cli.py --kaggle -o /kaggle/working/txt --skip-existing\n"
            "  python cli.py -i /kaggle/input/datasets/USER/NAME -o /kaggle/working/txt -r\n"
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="One PDF file, OR a directory of PDFs (auto-detects 1..N). Default: data/pdfs",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory — one .txt per PDF. Default: data/txt",
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help=(
            "Kaggle mode: find ALL PDFs under /kaggle/input (recursive). "
            "No need to copy PDFs one-by-one. Default out: /kaggle/working/txt"
        ),
    )
    parser.add_argument(
        "--engine",
        "-e",
        choices=ENGINES,
        default=None,
        help=(
            "OCR engine: paddle (Bangla quality, slower on CPU) or "
            "tesseract (fast CPU). Default from OCR_ENGINE env or paddle."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="PDF render DPI (default 300, or 150 with --fast)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Faster CPU mode: DPI 150, skip layout detection, shrink large pages. "
            "Slightly lower quality; much better than ~90s/page default on CPU."
        ),
    )
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

    # Quieter Paddle logs / Windows "pattern" noise
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("FLAGS_minloglevel", "2")
    import warnings

    warnings.filterwarnings("ignore", message=".*ccache.*")
    warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")

    if args.fast:
        os.environ["PADDLE_OCR_FAST"] = "1"
        os.environ["PADDLE_OCR_USE_LAYOUT"] = "0"
        os.environ.setdefault("PADDLE_OCR_MAX_SIDE", "1280")

    # Defaults + Kaggle convenience (read PDFs in place — no manual copy)
    if args.kaggle:
        input_path = args.input or Path("/kaggle/input")
        output_path = args.output or Path("/kaggle/working/txt")
        recursive = True
    else:
        input_path = args.input or (ROOT / "data" / "pdfs")
        output_path = args.output or (ROOT / "data" / "txt")
        # Auto-recurse when pointing at typical Kaggle input trees
        recursive = args.recursive or str(input_path).startswith("/kaggle/input")

    dpi = args.dpi if args.dpi is not None else (150 if args.fast else 300)
    engine = args.engine or resolve_default_engine()

    try:
        pdfs = discover_pdfs(input_path, recursive=recursive)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not pdfs:
        print(
            f"No PDFs found under {input_path}\n"
            "Pass a folder of PDFs with -i, or on Kaggle use --kaggle "
            "(scans /kaggle/input recursively — no copy needed).",
            file=sys.stderr,
        )
        return 1

    output_path.mkdir(parents=True, exist_ok=True)
    mode = "fast" if args.fast else "quality"
    print(
        f"Engine: {engine} | mode: {mode} | DPI: {dpi} | "
        f"recursive: {recursive} | Discovered PDFs: {len(pdfs)}"
    )
    for i, pdf in enumerate(pdfs, start=1):
        print(f"  [{i}/{len(pdfs)}] {pdf}")

    ok = skipped = failed = 0

    for pdf in pdfs:
        exact = output_path / f"{pdf.stem}.txt"
        if args.skip_existing and exact.exists():
            print(f"Skip (exists): {exact.name}")
            skipped += 1
            continue

        if args.force:
            out = exact
        elif exact.exists():
            out = unique_path(output_path, pdf.stem, ".txt")
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
                    dpi=dpi,
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
