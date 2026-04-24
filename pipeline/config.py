"""Shared configuration — env loading, paths, pricing constants, shared cost_log."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"
SEGMENTS_DIR = OUTPUT_DIR / "segments"
TERMS_DIR = OUTPUT_DIR / "terms"
TRANSLATIONS_DIR = OUTPUT_DIR / "translations"
QA_DIR = OUTPUT_DIR / "qa"
RUN_LOG_PATH = OUTPUT_DIR / "run.log"
COST_REPORT_PATH = OUTPUT_DIR / "cost_report.json"

for d in (OUTPUT_DIR, SCREENSHOTS_DIR, SEGMENTS_DIR, TERMS_DIR, TRANSLATIONS_DIR, QA_DIR):
    d.mkdir(parents=True, exist_ok=True)

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "").rstrip("/")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")

TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "claude-4.6-sonnet")
TERM_ID_MODEL = os.getenv("TERM_ID_MODEL", "gemini-3.1-flash-lite-preview")
QA_MODEL = os.getenv("QA_MODEL", "gemini-3.1-pro-preview")

COST_PER_INPUT_TOKEN = float(os.getenv("COST_PER_INPUT_TOKEN", "0.000003"))
COST_PER_OUTPUT_TOKEN = float(os.getenv("COST_PER_OUTPUT_TOKEN", "0.000015"))
GEMINI_FLASH_INPUT_COST = float(os.getenv("GEMINI_FLASH_INPUT_COST", "0.0000001"))
GEMINI_FLASH_OUTPUT_COST = float(os.getenv("GEMINI_FLASH_OUTPUT_COST", "0.0000004"))
GEMINI_PRO_INPUT_COST = float(os.getenv("GEMINI_PRO_INPUT_COST", "0.00000125"))
GEMINI_PRO_OUTPUT_COST = float(os.getenv("GEMINI_PRO_OUTPUT_COST", "0.00001"))

TRANSLATION_BATCH_SIZE = int(os.getenv("TRANSLATION_BATCH_SIZE", "5"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "5"))

# Shared cost log populated by every LLM-calling step. `list.append` is atomic
# under CPython's GIL, so concurrent workers can append without a lock.
cost_log: list[dict] = []


def page_slug(url: str) -> str:
    """Stable, filesystem-safe slug for a page URL.

    deriv.com/                                     -> deriv_home
    deriv.com/markets/forex/                       -> deriv_forex
    deriv.com/blog/posts/eur-usd-rebounds-...      -> deriv_blog_eur_usd_rebounds_dollar_demand_fades
    deriv.com/regulatory/                          -> deriv_regulatory
    deriv.com/trading-platforms/deriv-bot/         -> deriv_bot
    """
    path = urlparse(url).path.strip("/")
    if not path:
        return "deriv_home"
    parts = [p for p in path.split("/") if p]
    # Special-case short, recognisable slugs
    if parts[0] == "markets" and len(parts) >= 2:
        return f"deriv_{parts[1].replace('-', '_')}"
    if parts[0] == "trading-platforms" and len(parts) >= 2:
        return parts[1].replace("-", "_")
    if parts[0] == "blog":
        tail = parts[-1].replace("-", "_")
        return f"deriv_blog_{tail}"
    return "deriv_" + "_".join(p.replace("-", "_") for p in parts)
