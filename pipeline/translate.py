"""Step 3 — Multilingual translation (LiteLLM proxy, vision + structured output).

One LLM call per batch of TRANSLATION_BATCH_SIZE segments. The full-page
screenshot is attached to every batch as visual context — tone and register
cues are easier to get right with the page in view.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from .config import (
    COST_PER_INPUT_TOKEN,
    COST_PER_OUTPUT_TOKEN,
    MAX_CONCURRENCY,
    TRANSLATION_BATCH_SIZE,
    TRANSLATION_MODEL,
    TRANSLATIONS_DIR,
    cost_log,
    page_slug,
)
from .extract import load_segments, screenshot_path
from .llm import encode_image, extract_usage, make_chat
from .logging_utils import error, info, warn
from .terms import load_terms

LANGUAGE_NAMES = {
    "ar": "Arabic",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "de": "German",
    "it": "Italian",
    "ru": "Russian",
    "tr": "Turkish",
    "pl": "Polish",
    "ko": "Korean",
    "zh": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "vi": "Vietnamese",
    "bn": "Bengali",
    "sw": "Swahili",
}


class TranslatedSegment(BaseModel):
    segment_id: str
    translated_text: str


class TranslationBatchOutput(BaseModel):
    language: str
    translations: list[TranslatedSegment]


SYSTEM_PROMPT = """You are a professional translator for a regulated financial trading platform.
Translate UI copy accurately, preserving tone, intent, and register.

CRITICAL — Protected tokens:
Any token matching the pattern __T{n}__ is a protected brand name, product name,
or market term. You MUST:
1. Never translate these tokens — keep them exactly as-is (e.g. `__T3__` stays `__T3__`).
2. Preserve the exact count — if __T0__ appears twice in the source, it must
   appear exactly twice in the translation. No additions, no omissions.
3. Never split or modify the token markers themselves (no spaces inside `__T3__`).

Element type and section are provided as translation hints:
- h1/h2 in hero: punchy, short — do not expand
- button / cta: imperative, concise — match approximate character count
- p in footer / legal: formal register
- nav / nav-link: single words or short phrases only
- alt (image alt text): descriptive, concise
- li (list item): short, parallel structure with siblings

Return the translation for EACH segment keyed by its exact segment_id.
Do not merge, split, drop, or add segments."""


USER_TEMPLATE = """Translate the following {n} segments into {language_name} ({language_code}).

{token_hint}

Segments (JSON):
{segments_json}
"""


_WORD_CHAR = re.compile(r"\w")


def _term_regex(term: str) -> re.Pattern:
    """Compile a regex that matches a protected term without swallowing
    adjacent word characters.

    `Deriv` must match in `Deriv Bot` but not in `Derived`. Word boundaries are
    added at each end only where the term itself starts/ends with a word char
    (so terms containing parens/slashes still match at their edges).
    """
    prefix = r"\b" if _WORD_CHAR.match(term[0]) else ""
    suffix = r"\b" if _WORD_CHAR.match(term[-1]) else ""
    return re.compile(prefix + re.escape(term) + suffix)


def _substitute_tokens(text: str, term_to_token: dict[str, str]) -> str:
    """Replace every occurrence of each protected term with its opaque token.

    term_to_token is in longest-first order, which is essential: `Deriv MT5`
    must match and consume before the shorter `Deriv` sees its substring.
    """
    for term, token in term_to_token.items():
        pattern = _term_regex(term)
        text = pattern.sub(token, text)
    return text


def _restore_tokens(text: str, token_map: dict[str, str]) -> str:
    """Inverse of _substitute_tokens — token -> term."""
    for token, term in token_map.items():
        if token in text:
            text = text.replace(token, term)
    return text


_TOKEN_RX = re.compile(r"__T\d+__")


def _tokens_present(text: str) -> set[str]:
    """Set of __T{n}__ tokens appearing in a string."""
    return set(_TOKEN_RX.findall(text))


def _build_token_hint(batch_tokens: set[str], token_map: dict[str, str], categories: dict[str, str]) -> str:
    if not batch_tokens:
        return "No protected tokens appear in this batch."
    lines = ["Tokens present in this batch that must remain unchanged:"]
    # Sorted by numeric token id for readability.
    for tok in sorted(batch_tokens, key=lambda t: int(t.strip("_T"))):
        term = token_map.get(tok, "?")
        cat = categories.get(term, "")
        suffix = f" ({cat})" if cat else ""
        lines.append(f"- {tok} = {term}{suffix}")
    return "\n".join(lines)


def _chunks(seq: list, size: int) -> list[list]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _prepare_page(page_url: str, language: str, sllm) -> dict:
    """Load segments/terms/screenshot and build per-page context + batches.

    Kept separate from batch execution so we can flatten batches across pages
    into a single thread pool without blocking on each page's preparation.
    """
    slug = page_slug(page_url)
    segments_payload = load_segments(page_url)
    segments = segments_payload["segments"]

    terms_payload = load_terms(page_url)
    token_map: dict[str, str] = terms_payload["token_map"]
    term_to_token: dict[str, str] = terms_payload["term_to_token"]
    categories = {pt["term"]: pt["category"] for pt in terms_payload["protected_terms"]}

    language_name = LANGUAGE_NAMES.get(language, language)
    image_data_url = encode_image(screenshot_path(page_url))

    prepared = []
    for seg in segments:
        tokenised = _substitute_tokens(seg["text"], term_to_token)
        prepared.append(
            {
                "id": seg["id"],
                "text_original": seg["text"],
                "text_tokenised": tokenised,
                "type": seg["type"],
                "section": seg["section"],
                "width_px": seg["width_px"],
                "href": seg.get("href"),
            }
        )

    batches = _chunks(prepared, TRANSLATION_BATCH_SIZE)
    info(
        "step3",
        f"{page_url} → {language} — {len(segments)} segments in {len(batches)} batch(es)",
        page=page_url,
        language=language,
        batches=len(batches),
    )

    return {
        "page_url": page_url,
        "language": language,
        "language_name": language_name,
        "slug": slug,
        "segments": segments,
        "token_map": token_map,
        "term_to_token": term_to_token,
        "categories": categories,
        "image_data_url": image_data_url,
        "batches": batches,
        "sllm": sllm,
        # Populated by batch workers. dict.__setitem__ is atomic under CPython's
        # GIL, and each batch owns a disjoint set of segment_ids, so no lock needed.
        "translations": {},
    }


def _translate_batch(ctx: dict, bi: int) -> None:
    """Run one LLM call for batch ``bi`` of ``ctx`` and update ctx['translations']."""
    page_url = ctx["page_url"]
    language = ctx["language"]
    batches = ctx["batches"]
    batch = batches[bi]
    token_map = ctx["token_map"]
    categories = ctx["categories"]
    image_data_url = ctx["image_data_url"]
    sllm = ctx["sllm"]

    batch_tokens: set[str] = set()
    for item in batch:
        batch_tokens.update(_tokens_present(item["text_tokenised"]))
    token_hint = _build_token_hint(batch_tokens, token_map, categories)

    payload_segments = [
        {
            "id": item["id"],
            "text": item["text_tokenised"],
            "type": item["type"],
            "section": item["section"],
            "width_px": item["width_px"],
        }
        for item in batch
    ]

    user_text = USER_TEMPLATE.format(
        n=len(batch),
        language_name=ctx["language_name"],
        language_code=language,
        token_hint=token_hint,
        segments_json=json.dumps(payload_segments, ensure_ascii=False, indent=2),
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    t0 = time.perf_counter()
    try:
        result = sllm.invoke(messages)
    except Exception as e:
        error(
            "step3",
            f"{page_url} → {language} — batch {bi + 1}/{len(batches)} failed: {e}",
            page=page_url,
            language=language,
            batch_index=bi,
        )
        raise
    latency_ms = int((time.perf_counter() - t0) * 1000)

    parsed: TranslationBatchOutput = result["parsed"]
    raw = result["raw"]
    in_tok, out_tok = extract_usage(raw)

    cost_log.append(
        {
            "step": "step3_translation",
            "page": page_url,
            "language": language,
            "model": TRANSLATION_MODEL,
            "batch_index": bi,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "estimated_usd": in_tok * COST_PER_INPUT_TOKEN + out_tok * COST_PER_OUTPUT_TOKEN,
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    )

    returned_ids = {t.segment_id for t in parsed.translations}
    expected_ids = {item["id"] for item in batch}
    missing = expected_ids - returned_ids
    extra = returned_ids - expected_ids
    if missing:
        warn(
            "step3",
            f"{page_url} → {language} — batch {bi + 1}: missing segments {sorted(missing)}",
            page=page_url,
            language=language,
            batch_index=bi,
            missing=sorted(missing),
        )
    if extra:
        warn(
            "step3",
            f"{page_url} → {language} — batch {bi + 1}: unexpected segments {sorted(extra)}",
            page=page_url,
            language=language,
            batch_index=bi,
            extra=sorted(extra),
        )

    translations = ctx["translations"]
    for t in parsed.translations:
        src_tokenised = next(
            (b["text_tokenised"] for b in batch if b["id"] == t.segment_id), ""
        )
        for tok in _tokens_present(src_tokenised):
            src_count = src_tokenised.count(tok)
            out_count = t.translated_text.count(tok)
            if src_count != out_count:
                warn(
                    "step3",
                    f"{page_url} → {language} — {t.segment_id}: token {tok} "
                    f"count mismatch (src={src_count}, out={out_count})",
                    page=page_url,
                    language=language,
                    segment_id=t.segment_id,
                    token=tok,
                )
        translations[t.segment_id] = t.translated_text

    info(
        "step3",
        f"{page_url} → {language} — batch {bi + 1}/{len(batches)} complete "
        f"({len(parsed.translations)} segments, {in_tok}+{out_tok} tok, {latency_ms}ms)",
        page=page_url,
        language=language,
        batch_index=bi,
        returned=len(parsed.translations),
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_ms=latency_ms,
    )


def _finalize_page(ctx: dict) -> dict:
    """Restore tokens and write the per-page translation JSON."""
    page_url = ctx["page_url"]
    language = ctx["language"]
    segments = ctx["segments"]
    token_map = ctx["token_map"]
    translations = ctx["translations"]

    out_segments = []
    for seg in segments:
        tokenised = translations.get(seg["id"])
        if tokenised is None:
            warn(
                "step3",
                f"{page_url} → {language} — {seg['id']}: no translation returned, "
                f"falling back to source text",
                page=page_url,
                language=language,
                segment_id=seg["id"],
            )
            restored = seg["text"]
        else:
            restored = _restore_tokens(tokenised, token_map)

        out_segments.append(
            {
                "id": seg["id"],
                "source_text": seg["text"],
                "translated_text": restored,
                "translated_text_tokenised": tokenised,
                "type": seg["type"],
                "section": seg["section"],
                "width_px": seg["width_px"],
                "href": seg.get("href"),
            }
        )

    lang_dir = TRANSLATIONS_DIR / language
    lang_dir.mkdir(parents=True, exist_ok=True)
    out_path = lang_dir / f"{ctx['slug']}.json"
    payload = {
        "page": page_url,
        "language": language,
        "segments": out_segments,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    info(
        "step3",
        f"{page_url} → {language} — wrote {out_path.relative_to(TRANSLATIONS_DIR.parent.parent)}",
        page=page_url,
        language=language,
        out_path=str(out_path),
    )
    return payload


def translate_page(page_url: str, language: str) -> dict:
    """Translate a single page end-to-end (batches run sequentially here).

    Kept for ad-hoc / single-page callers. The concurrent pipeline uses
    ``translate_all`` which flattens batches across pages.
    """
    llm = make_chat(TRANSLATION_MODEL, temperature=0.2)
    sllm = llm.with_structured_output(TranslationBatchOutput, include_raw=True)
    ctx = _prepare_page(page_url, language, sllm)
    for bi in range(len(ctx["batches"])):
        _translate_batch(ctx, bi)
    return _finalize_page(ctx)


def translate_all(pages: list[str], language: str) -> None:
    """Translate every page in parallel at the batch level.

    Pages' batches are flattened into one work queue so a 2-batch page does not
    idle 3 workers while a 10-batch page is still running — whoever frees a
    worker slot next picks up the next pending batch regardless of which page
    it belongs to.
    """
    llm = make_chat(TRANSLATION_MODEL, temperature=0.2)
    sllm = llm.with_structured_output(TranslationBatchOutput, include_raw=True)

    contexts = [_prepare_page(url, language, sllm) for url in pages]

    work_items: list[tuple[dict, int]] = [
        (ctx, bi) for ctx in contexts for bi in range(len(ctx["batches"]))
    ]
    if not work_items:
        return

    info(
        "step3",
        f"dispatching {len(work_items)} batch(es) across {len(contexts)} page(s) "
        f"with {MAX_CONCURRENCY} workers",
        language=language,
        total_batches=len(work_items),
        pages=len(contexts),
    )

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futures = [pool.submit(_translate_batch, ctx, bi) for ctx, bi in work_items]
        for fut in as_completed(futures):
            fut.result()  # surface exceptions

    for ctx in contexts:
        _finalize_page(ctx)


def load_translation(page_url: str, language: str) -> dict:
    slug = page_slug(page_url)
    return json.loads((TRANSLATIONS_DIR / language / f"{slug}.json").read_text())
