"""Step 3b — programmatic QA; Step 5 — LLM QA scoring with Gemini Pro."""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from .config import (
    GEMINI_PRO_INPUT_COST,
    GEMINI_PRO_OUTPUT_COST,
    MAX_CONCURRENCY,
    QA_DIR,
    QA_MODEL,
    cost_log,
    page_slug,
)
from .extract import load_segments, screenshot_path
from .llm import encode_image, extract_usage, make_chat
from .logging_utils import info, qa, warn
from .terms import load_terms
from .translate import _substitute_tokens, load_translation

TOKEN_RX = re.compile(r"__T\d+__")

# Content types where source == translated is legitimate (numbers, symbols, acronyms).
_EXEMPT_TYPES = {"alt"}  # alt text may legitimately be the same in some cases


class QAResult(BaseModel):
    score: int = Field(..., ge=0, le=100, description="0–100 quality score")
    feedback: str = Field(..., description="Actionable, freeform evaluation")


SYSTEM_PROMPT_QA = """You are a professional translation quality evaluator for a regulated financial trading platform.

Score the translation from 0 to 100 where:
  100 = perfect fidelity, tone, and term protection
   60 = any protected brand/product/market name is mistranslated (HARD CAP)
    0 = completely broken or untranslated

Judgement rubric (apply in order):
1. Protected term fidelity — any translated brand/product/market term caps the
   score at 60. List every violation in the feedback with the segment id.
2. Completeness — flag segments that were silently dropped or echoed back in
   the source language.
3. Fluency & register — evaluate natural phrasing, tone, and alignment with
   element type (punchy hero h1 vs formal footer legal).
4. Length discipline — for buttons and nav labels, the translation should not
   balloon significantly past the source's character budget.

Feedback should be concise, actionable, and reference specific segment ids."""


USER_TEMPLATE_QA = """Evaluate the {language_name} ({language_code}) translation of this page.

Protected terms that must appear UNCHANGED in every segment that contained them:
{protected_terms_block}

Source → translation pairs:
{pairs_block}
"""


_LATIN_LETTER_RX = re.compile(r"[A-Za-zÀ-ÿ]")


def _has_translatable_content(source: str, term_to_token: dict[str, str]) -> bool:
    """True iff the source has translatable text after protected terms are removed.

    A segment that IS a protected term (e.g. 'Forex', 'Deriv MT5') legitimately
    round-trips unchanged. A segment that's pure numbers/units ('168M+', '3M+')
    also has no translatable content.
    """
    cleaned = source
    for term in term_to_token:
        if term in cleaned:
            cleaned = cleaned.replace(term, "")
    # After removing terms and digits/symbols/whitespace, 2+ consecutive letters
    # still standing means there is a real word to translate.
    letters_only = "".join(ch for ch in cleaned if _LATIN_LETTER_RX.match(ch))
    return len(letters_only) >= 2


def _check_untranslated(source: str, translated: str, term_to_token: dict[str, str]) -> bool:
    if source.strip() != translated.strip():
        return False
    return _has_translatable_content(source, term_to_token)


def _check_token_corruption(tokenised_source: str, tokenised_output: str) -> list[str]:
    """Compare __T{n}__ counts between source and PRE-restoration output."""
    issues: list[str] = []
    src_tokens = set(TOKEN_RX.findall(tokenised_source))
    for tok in src_tokens:
        src_count = tokenised_source.count(tok)
        out_count = tokenised_output.count(tok) if tokenised_output else 0
        if src_count != out_count:
            issues.append(f"{tok}: expected {src_count}, found {out_count}")
    return issues


def programmatic_qa(page_url: str, language: str) -> dict:
    """Step 3b — deterministic, no-LLM QA."""
    translation = load_translation(page_url, language)
    terms = load_terms(page_url)
    term_to_token: dict[str, str] = terms["term_to_token"]
    slug = page_slug(page_url)

    issues = []
    untranslated_count = 0
    corruption_count = 0

    for seg in translation["segments"]:
        src = seg["source_text"]
        out_restored = seg["translated_text"]
        out_tokenised = seg.get("translated_text_tokenised") or ""

        # Re-tokenise the source so we can compare token counts pre-restoration.
        # Must use the same word-boundary substitution as Step 3 or we'll
        # phantom-tokenise substrings ('Deriv' inside 'Derived').
        tokenised_src = _substitute_tokens(src, term_to_token)

        if (
            seg["type"] not in _EXEMPT_TYPES
            and _check_untranslated(src, out_restored, term_to_token)
        ):
            issues.append(
                {
                    "segment_id": seg["id"],
                    "type": "untranslated",
                    "source_text": src,
                    "translated_text": out_restored,
                }
            )
            untranslated_count += 1
            warn(
                "step3b",
                f"{page_url} → {language} — {seg['id']}: untranslated (source == output)",
                page=page_url,
                language=language,
                segment_id=seg["id"],
            )

        for corruption in _check_token_corruption(tokenised_src, out_tokenised):
            issues.append(
                {
                    "segment_id": seg["id"],
                    "type": "token_corruption",
                    "detail": corruption,
                    "source_text": src,
                    "translated_text": out_restored,
                }
            )
            corruption_count += 1
            warn(
                "step3b",
                f"{page_url} → {language} — {seg['id']}: token corruption — {corruption}",
                page=page_url,
                language=language,
                segment_id=seg["id"],
            )

    report = {
        "page": page_url,
        "language": language,
        "total_segments": len(translation["segments"]),
        "issues": issues,
        "untranslated_count": untranslated_count,
        "corruption_count": corruption_count,
        "pass": untranslated_count == 0 and corruption_count == 0,
    }

    lang_dir = QA_DIR / language
    lang_dir.mkdir(parents=True, exist_ok=True)
    (lang_dir / f"{slug}_programmatic_qa.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    info(
        "step3b",
        f"{page_url} → {language} — {len(translation['segments'])} segments checked, "
        f"{len(issues)} issues found",
        page=page_url,
        language=language,
        total=len(translation["segments"]),
        issues=len(issues),
    )
    return report


def _llm_qa_with_client(page_url: str, language: str, sllm) -> dict:
    """Step 5 body — accepts a pre-built structured-output client so the
    concurrent dispatcher can share one across threads."""
    from .translate import LANGUAGE_NAMES  # local import to avoid cycle at import time

    translation = load_translation(page_url, language)
    terms = load_terms(page_url)
    slug = page_slug(page_url)
    language_name = LANGUAGE_NAMES.get(language, language)

    image_data_url = encode_image(screenshot_path(page_url))

    protected_terms_block = "\n".join(
        f"- {pt['term']} ({pt['category']})" for pt in terms["protected_terms"]
    )
    pairs_block = "\n".join(
        f"[{seg['id']}] ({seg['type']}, {seg['section']})\n"
        f"  SRC: {seg['source_text']}\n"
        f"  OUT: {seg['translated_text']}"
        for seg in translation["segments"]
    )

    user_text = USER_TEMPLATE_QA.format(
        language_name=language_name,
        language_code=language,
        protected_terms_block=protected_terms_block,
        pairs_block=pairs_block,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_QA},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    t0 = time.perf_counter()
    result = sllm.invoke(messages)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    parsed: QAResult = result["parsed"]
    raw = result["raw"]
    in_tok, out_tok = extract_usage(raw)

    cost_log.append(
        {
            "step": "step5_qa",
            "page": page_url,
            "language": language,
            "model": QA_MODEL,
            "batch_index": None,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "estimated_usd": in_tok * GEMINI_PRO_INPUT_COST + out_tok * GEMINI_PRO_OUTPUT_COST,
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    )

    payload = {
        "page": page_url,
        "language": language,
        "score": parsed.score,
        "feedback": parsed.feedback,
    }

    lang_dir = QA_DIR / language
    lang_dir.mkdir(parents=True, exist_ok=True)
    (lang_dir / f"{slug}_llm_qa.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    # One-line summary to stdout.
    first_line = parsed.feedback.splitlines()[0] if parsed.feedback else ""
    short = (first_line[:120] + "…") if len(first_line) > 120 else first_line
    qa(
        "step5",
        f"{page_url} → {language}   {parsed.score}/100   {short}",
        page=page_url,
        language=language,
        score=parsed.score,
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_ms=latency_ms,
    )
    return payload


def llm_qa(page_url: str, language: str) -> dict:
    """Single-page LLM QA entry point — builds its own client."""
    llm = make_chat(QA_MODEL, temperature=0.0)
    sllm = llm.with_structured_output(QAResult, include_raw=True)
    return _llm_qa_with_client(page_url, language, sllm)


def run_qa(pages: list[str], language: str) -> None:
    """Programmatic QA sequentially, then LLM QA in parallel across pages."""
    for url in pages:
        programmatic_qa(url, language)

    if not pages:
        return

    llm = make_chat(QA_MODEL, temperature=0.0)
    sllm = llm.with_structured_output(QAResult, include_raw=True)

    info(
        "step5",
        f"dispatching LLM QA for {len(pages)} page(s) with {MAX_CONCURRENCY} workers",
        language=language,
        pages=len(pages),
    )

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futures = [pool.submit(_llm_qa_with_client, url, language, sllm) for url in pages]
        for fut in as_completed(futures):
            fut.result()  # surface exceptions
