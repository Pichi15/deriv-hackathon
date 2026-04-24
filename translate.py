"""Deriv multilingual translation pipeline — entry point.

Run:
    uv run translate.py --language ar
"""
from __future__ import annotations

import argparse

from pipeline.extract import extract_all_pages
from pipeline.qa import run_qa
from pipeline.report import print_cost_report
from pipeline.terms import identify_terms
from pipeline.translate import translate_all

PAGES = [
    "https://deriv.com/",
    "https://deriv.com/markets/forex/",
    "https://deriv.com/blog/posts/eur-usd-rebounds-dollar-demand-fades/",
    "https://deriv.com/regulatory/",
    "https://deriv.com/trading-platforms/deriv-bot/",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Deriv multilingual translation pipeline")
    parser.add_argument("--language", required=True, help="Target language code (e.g. ar, fr, es)")
    parser.add_argument(
        "--pages",
        nargs="*",
        help="Optional subset of page URLs (default: all five). Useful for the 2-page core build.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Ignore the Step 1 cache and re-scrape every page (otherwise pages whose "
        "segments JSON + screenshot both already exist on disk are skipped).",
    )
    args = parser.parse_args()

    pages = args.pages if args.pages else PAGES

    extract_all_pages(pages, force=args.force_extract)
    identify_terms(pages)
    translate_all(pages, args.language)
    run_qa(pages, args.language)
    print_cost_report()


if __name__ == "__main__":
    main()
