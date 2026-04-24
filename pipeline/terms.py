"""Step 2 — Protected term identification (Gemini Flash Lite, structured output).

Runs once per page. Output is language-agnostic and cached to output/terms/<slug>_terms.json.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from .config import (
    GEMINI_FLASH_INPUT_COST,
    GEMINI_FLASH_OUTPUT_COST,
    MAX_CONCURRENCY,
    TERM_ID_MODEL,
    TERMS_DIR,
    cost_log,
    page_slug,
)
from .extract import load_segments
from .llm import extract_usage, make_chat
from .logging_utils import info

# Regexes from Step 1 — reused here so dynamic tokens are captured in the unified list.
TOKEN_RX = [
    re.compile(r"\{\{[^{}]+\}\}"),
    re.compile(r"\{\d+\}"),
    re.compile(r"%[sdif]"),
]


class TermCategory(str, Enum):
    brand = "brand"
    product = "product"
    platform = "platform"
    market = "market"
    regulatory = "regulatory"
    token = "token"


class ProtectedTerm(BaseModel):
    term: str = Field(..., description="Exact string as it appears on the page")
    category: TermCategory


class ProtectedTermsOutput(BaseModel):
    protected_terms: list[ProtectedTerm]


SYSTEM_PROMPT = """You are a term protection assistant for a translation pipeline.

Given text segments from a financial trading website, identify EVERY term that
must NOT be translated. Categories and examples (return only terms actually
present in the input):

- brand       : Deriv, Deriv Group, Deriv.com Limited, Trustpilot, UF Awards
- product     : Deriv Academy, Deriv Blog, Deriv API
- platform    : Deriv MT5, Deriv Bot, Deriv Trader, Deriv cTrader, Deriv Nakala,
                Deriv GO, SmartTrader, Deriv P2P, Deriv app, MetaTrader 5, MT5, cTrader
- market      : currency pairs (EUR/USD, GBP/USD), asset class names (CFDs, Forex,
                ETFs, Stocks, Commodities, Cryptocurrencies), indices, company names
                used as tradable symbols (Apple, Tesla, NVIDIA, Bitcoin, Ethereum)
- regulatory  : regulatory bodies and legal entities (Malta Financial Services
                Authority, MFSA, Labuan FSA, Deriv (BVI) Ltd, Deriv (FX) Ltd, ...)

Rules:
1. Include the exact string as it appears in the segments (match casing and punctuation).
2. Order results LONGEST TO SHORTEST to prevent partial-match errors during
   substitution. "Deriv MT5" must come BEFORE "Deriv".
3. Deduplicate — each distinct term appears once.
4. Do NOT include generic verbs or UI words ("Trading", "Markets", "Platforms",
   "Login", "Open account") unless they are part of a branded name.
5. Do NOT invent terms that are not present in the input."""


USER_TEMPLATE = """Identify the protected terms in these segments from {page_url}.

Segments:
{segments_block}
"""


def _detect_dynamic_tokens_in_page(segments: list[dict]) -> list[str]:
    found: list[str] = []
    for seg in segments:
        for rx in TOKEN_RX:
            found.extend(rx.findall(seg["text"]))
    # Dedupe preserving order
    return list(dict.fromkeys(found))


def _build_token_map(terms: list[ProtectedTerm], dynamic_tokens: list[str]) -> dict[str, str]:
    """Assign opaque __T{n}__ tokens, longest-first so substitution is safe."""
    # Merge LLM-identified terms with regex-detected dynamic tokens (category=token).
    all_items: list[tuple[str, str]] = []
    seen: set[str] = set()

    # Dynamic tokens go last numerically but are deduplicated against LLM output first.
    for t in terms:
        if t.term not in seen:
            all_items.append((t.term, t.category.value))
            seen.add(t.term)
    for dt in dynamic_tokens:
        if dt not in seen:
            all_items.append((dt, "token"))
            seen.add(dt)

    # Sort longest-first (ties broken alphabetically for determinism).
    all_items.sort(key=lambda x: (-len(x[0]), x[0]))

    token_map = {f"__T{i}__": term for i, (term, _) in enumerate(all_items)}
    categories = {term: cat for term, cat in all_items}
    return token_map, categories


def _identify_one(page_url: str, sllm) -> None:
    """Run term identification for a single page and write its cache file."""
    slug = page_slug(page_url)
    cache_path = TERMS_DIR / f"{slug}_terms.json"
    if cache_path.exists():
        info("step2", f"{page_url} — cache hit, skipping", page=page_url, cached=True)
        return

    segments_payload = load_segments(page_url)
    segments = segments_payload["segments"]
    segments_block = "\n".join(f"- {s['text']}" for s in segments)
    user_msg = USER_TEMPLATE.format(page_url=page_url, segments_block=segments_block)

    t0 = time.perf_counter()
    result = sllm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    parsed: ProtectedTermsOutput = result["parsed"]
    raw = result["raw"]
    in_tok, out_tok = extract_usage(raw)

    dynamic_tokens = _detect_dynamic_tokens_in_page(segments)
    token_map, categories = _build_token_map(parsed.protected_terms, dynamic_tokens)

    # Invert for substitution: term -> token (longest-first preserved since
    # we built the map in that order).
    term_to_token = {term: tok for tok, term in token_map.items()}

    cost_log.append(
        {
            "step": "step2_term_id",
            "page": page_url,
            "language": None,
            "model": TERM_ID_MODEL,
            "batch_index": None,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "estimated_usd": in_tok * GEMINI_FLASH_INPUT_COST + out_tok * GEMINI_FLASH_OUTPUT_COST,
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    )

    payload = {
        "page": page_url,
        "protected_terms": [
            {"term": term, "category": categories[term], "token": term_to_token[term]}
            for term in term_to_token
        ],
        "token_map": token_map,  # token -> term
        "term_to_token": term_to_token,  # term -> token (longest-first order)
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    info(
        "step2",
        f"{page_url} — identified {len(term_to_token)} protected terms "
        f"({len(parsed.protected_terms)} from LLM, {len(dynamic_tokens)} dynamic)",
        page=page_url,
        count=len(term_to_token),
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_ms=latency_ms,
    )


def identify_terms(pages: list[str]) -> None:
    """Run term identification for every page in parallel (up to MAX_CONCURRENCY).

    The LLM handle is shared across threads — ChatOpenAI's invoke() is stateless
    per call, so there's no contention beyond the underlying HTTP client's
    connection pool.
    """
    llm = make_chat(TERM_ID_MODEL, temperature=0.0)
    sllm = llm.with_structured_output(ProtectedTermsOutput, include_raw=True)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futures = [pool.submit(_identify_one, url, sllm) for url in pages]
        for fut in futures:
            fut.result()  # surface exceptions


def load_terms(page_url: str) -> dict:
    slug = page_slug(page_url)
    return json.loads((TERMS_DIR / f"{slug}_terms.json").read_text())
