# Deriv Multilingual Translation Pipeline — Design Spec
> Hackathon build. 60-minute window. Python. LiteLLM + LangChain.

---

## Logging Standards (applies to all steps)

Every significant event in the pipeline writes a structured log entry. All log entries share a base shape and are written to both stdout (human-readable via `rich`) and a JSONL file (`output/run.log`) for machine consumption.

### LLM Call Log Schema

Every LLM call across every step appends to a shared `cost_log` list with consistent tags:

```python
{
    "step":          "step2_term_id | step3_translation | step5_qa",
    "page":          "https://deriv.com/",          # always present
    "language":      "ar",                           # None for step 2 (language-agnostic)
    "model":         "gemini-3.1-flash-lite",        # actual model used
    "batch_index":   0,                              # None for single-call steps
    "input_tokens":  1240,
    "output_tokens": 380,
    "estimated_usd": 0.00091,
    "latency_ms":    1340,
    "timestamp":     "2026-04-24T10:32:01Z",
}
```

Tagging `step`, `page`, and `language` on every entry is what makes Step 4's grouped cost report possible — any entry can be filtered and aggregated along any axis without post-hoc inference.

### Pipeline Event Log

Non-LLM events log at INFO/WARNING level to stdout only:

```
[INFO]  [step1] deriv.com — extracted 84 segments, screenshot saved
[INFO]  [step2] deriv.com — identified 23 protected terms
[INFO]  [step3] deriv.com → ar — batch 1/17 complete (5 segments)
[WARN]  [step3] deriv.com → ar — seg_042: token __T3__ missing from translation
[INFO]  [step5] deriv.com → ar — QA score: 87/100
[ERROR] [step3] deriv.com → fr — batch 3 failed: LiteLLM timeout, retrying
```

---

## Project Structure & Entry Point

### Running the pipeline

```bash
uv run translate.py --language ar
```

A single language code is passed per run. To translate into multiple languages, run the script once per language. Each run is fully self-contained and saves all intermediate outputs — re-running the same language overwrites previous outputs for that language.

### Entry point: `translate.py`

```python
import argparse
from pipeline.extract   import extract_all_pages
from pipeline.terms     import identify_terms
from pipeline.translate import translate_all
from pipeline.qa        import run_qa
from pipeline.report    import print_cost_report

PAGES = [
    "https://deriv.com/",
    "https://deriv.com/markets/forex/",
    "https://deriv.com/blog/posts/eur-usd-rebounds-dollar-demand-fades/",
    "https://deriv.com/regulatory/",
    "https://deriv.com/trading-platforms/deriv-bot/",
]

def main():
    parser = argparse.ArgumentParser(description="Deriv multilingual translation pipeline")
    parser.add_argument("--language", required=True, help="Target language code (e.g. ar, fr, es)")
    args = parser.parse_args()

    extract_all_pages(PAGES)          # Step 1 — saves to output/segments/ and output/screenshots/
    identify_terms(PAGES)             # Step 2 — saves to output/terms/
    translate_all(PAGES, args.language)  # Step 3 — saves to output/translations/<lang>/
    run_qa(PAGES, args.language)      # Step 3b + Step 5 — saves to output/qa/<lang>/
    print_cost_report()               # Step 4 — prints table, saves output/cost_report.json

if __name__ == "__main__":
    main()
```

### `pyproject.toml` (uv)

```toml
[project]
name = "deriv-translation-pipeline"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "playwright>=1.43",
    "langchain>=0.2",
    "langchain-community>=0.2",
    "langchain-google-genai>=1.0",
    "litellm>=1.40",
    "pydantic>=2.0",
    "rich>=13.0",
    "python-dotenv>=1.0",
    "pillow>=10.0",
]

[project.scripts]
translate = "translate:main"
```

Install and first run:
```bash
uv sync
uv run playwright install chromium
uv run translate.py --language ar
```

### Output Directory Structure

All intermediate and final outputs are saved to `output/`. Every step writes before the next begins — a crash mid-run leaves all completed work intact.

```
output/
├── screenshots/                        # Step 1 — one PNG per page
│   ├── deriv_home.png
│   ├── deriv_forex.png
│   └── ...
├── segments/                           # Step 1 — extracted segments per page
│   ├── deriv_home_segments.json
│   ├── deriv_forex_segments.json
│   └── ...
├── terms/                              # Step 2 — protected terms per page (cached)
│   ├── deriv_home_terms.json
│   └── ...
├── translations/                       # Step 3 — translated segments per language
│   └── ar/
│       ├── deriv_home.json
│       ├── deriv_forex.json
│       └── ...
├── qa/                                 # Step 3b + Step 5 — QA results per language
│   └── ar/
│       ├── deriv_home_programmatic_qa.json
│       ├── deriv_home_llm_qa.json
│       └── ...
├── cost_report.json                    # Step 4 — full cost log + summary
└── run.log                             # All steps — structured JSONL event log
```

### `.env` (full reference)

```
# LiteLLM routing
LITELLM_BASE_URL=https://your-custom-endpoint
LITELLM_MODEL=claude-sonnet-4-6

# Gemini
GOOGLE_API_KEY=your-key

# Pricing (set to model's published rates)
COST_PER_INPUT_TOKEN=0.000003
COST_PER_OUTPUT_TOKEN=0.000015

# Pipeline tuning
TRANSLATION_BATCH_SIZE=5        # segments per translation call
```

---

## Step 1: Content Extraction

### Tool: Playwright (Python)

Deriv's site is a React/Next.js app. A raw HTTP request (`requests`, `curl`) returns a near-empty HTML shell — the content is injected by JavaScript after page load. Playwright runs a headless Chromium browser, waits for the DOM to fully hydrate, then reads the rendered output. This is the only reliable way to get the actual page content.

### Input Pages

```
https://deriv.com/
https://deriv.com/markets/forex/
https://deriv.com/blog/posts/eur-usd-rebounds-dollar-demand-fades/
https://deriv.com/regulatory/
https://deriv.com/trading-platforms/deriv-bot/
```

For the core build, extract from 2 pages: the homepage and one content-heavy page (forex or blog). The others are stretch.

### Page Priority

| Priority | Page | Why |
|---|---|---|
| Core | `deriv.com/` | Highest traffic, all element types present, nav + hero + CTAs |
| Core | `deriv.com/markets/forex/` | Market-specific copy, good CTA density, named terms (EUR/USD, etc.) |
| Stretch | `deriv.com/trading-platforms/deriv-bot/` | Product page — heavy on brand terms, good term protection test |
| Stretch | `deriv.com/blog/posts/eur-usd-...` | Long-form prose, tests fluency at sentence level |
| Stretch | `deriv.com/regulatory/` | Legal/compliance language — hardest to translate correctly, good QA stress test |

### What to Extract

Target every element that carries translatable content or UI meaning. The hackathon spec explicitly requires: headings, body text, CTAs, navigation, links, formatting, and placeholders/tokens.

| Element type | CSS selector targets | Why |
|---|---|---|
| Headings | `h1, h2, h3, h4` | Primary communicative content |
| Body text | `p` | Core copy |
| List items | `ul li, ol li` | Used heavily in nav dropdowns, feature lists, regulatory lists |
| CTAs / buttons | `button, a[role="button"], [class*="cta"]` | High-impact short strings |
| Navigation labels | `nav a` | Appear on every page — high reuse value |
| Inline links | `main a:not([role="button"])` | Hackathon spec explicitly calls out link preservation; `href` must be retained |
| Alt text | `img[alt]` | Accessibility requirement; often missed |

**Link handling:** For inline `<a>` tags, preserve the `href` attribute in the output schema — it must not be translated and must survive the round-trip unchanged. Treat it like a protected token (see Step 2).

**Dynamic tokens/placeholders:** The spec requires preserving placeholders/tokens. Scan extracted text for patterns like `{{variable}}`, `{0}`, `%s`, or `__placeholder__`. These must be identified, protected before translation, and restored after — same mechanism as brand term protection.

Exclude: `<script>`, `<style>`, `<meta>`, data attributes, cookie banners, Trustpilot widget content (third-party iframe, not part of Deriv's translatable surface).

### Output Format

Each extracted segment is stored as a structured object — not raw HTML, not plain text:

```json
{
  "page": "https://deriv.com/",
  "screenshot_path": "output/screenshots/deriv_home.png",
  "segments": [
    {
      "id": "seg_001",
      "type": "h1",
      "text": "Trading for Anyone Anywhere Anytime",
      "html": "<h1>Trading for <span>Anyone</span><br>Anywhere<br>Anytime</h1>",
      "xpath": "/html/body/main/section[1]/h1",
      "section": "hero",
      "width_px": 600
    }
  ]
}
```

| Field | Source | Purpose |
|---|---|---|
| `text` | Inner text of element | What gets translated |
| `html` | `element.inner_html()` | Structural reassembly after translation |
| `xpath` | Computed from DOM position | Reinjection anchor |
| `section` | Nearest ancestor landmark tag or class | Translation hint — tone and register |
| `width_px` | `element.bounding_box()["width"]` | Character budget hint for CTAs and nav |
| `screenshot_path` | Page-level field, not per-segment | Passed to translation and QA calls |

`id` is stable across runs — generated as `seg_{page_slug}_{index}` — used as the key for cache lookups and QA mapping.

### Why preserve HTML alongside text?

Translation operates on `text`. But the output must be injected back into the page structure — inline tags (`<span>`, `<strong>`, `<br>`) must survive the round-trip. Stripping to plain text then translating loses this. The pipeline protects inner tags as placeholders during translation (covered in Step 2).

---

## Step 2: Term Identification & Protection

### Tool: LangChain + Gemini 3.1 Flash Lite + Pydantic

Term identification is cheap work — it requires pattern recognition across short strings, not nuanced multilingual reasoning. Gemini Flash Lite is fast and inexpensive, runs once per page (not per language), and its output is cached and reused across all translation calls in Step 3. Using a frontier model here would waste budget on a task that doesn't need it.

LangChain's `.with_structured_output()` is used with a bounded Pydantic schema to enforce the response shape at the framework level — the LLM cannot return malformed JSON, missing fields, or values outside the allowed category enum. This removes the need for any post-processing or defensive parsing of the model's output.

### What to Identify

The hackathon spec requires identifying brand names, product names, and market names without providing the list — finding them is part of the task. Feed all extracted `text` fields from Step 1 to the model and ask it to classify any term that should pass through translation unchanged.

| Category | Examples from Deriv's surface |
|---|---|
| Brand names | `Deriv`, `Deriv Group` |
| Product names | `Deriv MT5`, `Deriv Bot`, `Deriv Trader`, `Deriv cTrader`, `Deriv Nakala`, `Deriv GO`, `SmartTrader`, `Deriv P2P`, `Deriv app` |
| Platform names | `MetaTrader 5`, `MT5`, `cTrader` |
| Currency pairs & market names | `EUR/USD`, `GBP/USD`, `CFDs`, `Forex`, `ETFs` |
| Regulatory & legal entities | `Malta Financial Services Authority`, `MFSA`, `Labuan FSA`, `Trustpilot` |
| Dynamic tokens | `{{variable}}`, `{0}`, `%s` — detected by regex in Step 1, passed in here for unified handling |

### Pydantic Schema

```python
from enum import Enum
from pydantic import BaseModel

class TermCategory(str, Enum):
    brand      = "brand"
    product    = "product"
    platform   = "platform"
    market     = "market"
    regulatory = "regulatory"
    token      = "token"

class ProtectedTerm(BaseModel):
    term: str
    category: TermCategory

class ProtectedTermsOutput(BaseModel):
    protected_terms: list[ProtectedTerm]
```

The `TermCategory` enum bounds the model to exactly six valid values — any hallucinated category causes a validation error before it reaches the pipeline. LangChain wires this via:

```python
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
structured_llm = llm.with_structured_output(ProtectedTermsOutput)
result: ProtectedTermsOutput = structured_llm.invoke(prompt)
```

### Prompt Design

Send all segments in a single batched call — no prose in the response, the schema handles structure. Prompt shape:

```
You are a term protection assistant for a translation pipeline.

Given the following text segments from a financial trading website, identify every term that must NOT be translated: brand names, product names, platform names, currency pairs, market names, regulatory bodies, and any dynamic tokens.

Order results longest to shortest — this prevents partial-match errors during substitution.

Segments:
[...all extracted text joined...]
```

Longest-first ordering is enforced in the prompt because substitution is done left-to-right — `Deriv MT5` must be matched and replaced before `Deriv`, otherwise `Deriv` in `Deriv MT5` gets clobbered first.

### Output: Protected Terms List

The LLM returns terms and categories only. Tokens are assigned in Python immediately after:

```python
# LLM output
result: ProtectedTermsOutput = structured_llm.invoke(prompt)

# Token assignment — deterministic, unique, no LLM involvement
token_map = {
    f"__T{i}__": term.term
    for i, term in enumerate(result.protected_terms)
}
# e.g. {"__T0__": "Deriv MT5", "__T1__": "Deriv Bot", "__T2__": "EUR/USD", ...}
```

Before a segment goes to the translation LLM in Step 3, terms are replaced with their tokens. After translation, tokens are swapped back. The translated text never sees the protected strings.

### Scope & Caching

- Run once per page, not per language. The protected terms list is language-agnostic.
- Cache output to disk (`output/terms/<page_slug>_terms.json`). On re-runs, skip the Gemini call if the cache file exists.
- The same terms list is reused for all target languages in Step 3, making this the cheapest step in the pipeline per unit of value delivered.

### Cost Logging

```python
cost_log.append({
    "step":          "step2_term_id",
    "page":          page_url,
    "language":      None,                          # language-agnostic
    "model":         "gemini-3.1-flash-lite",
    "batch_index":   None,
    "input_tokens":  input_tokens,
    "output_tokens": output_tokens,
    "estimated_usd": (input_tokens  * GEMINI_FLASH_INPUT_COST)
                   + (output_tokens * GEMINI_FLASH_OUTPUT_COST),
    "latency_ms":    latency_ms,
    "timestamp":     datetime.utcnow().isoformat() + "Z",
})
```

Gemini Flash Lite pricing constants are separate from the LiteLLM model pricing and defined in `config.py`.

---

## Step 3: Multilingual Translation

### Tool: LiteLLM (custom base URL) + LangChain + Pydantic

LiteLLM is the HTTP layer — it provides a unified OpenAI-compatible interface and routes calls to the configured backend via a custom base URL. LangChain sits on top via `ChatLiteLLM`, and Pydantic enforces the response schema as in Step 2.

```python
# .env
LITELLM_BASE_URL=https://your-custom-endpoint
LITELLM_MODEL=claude-sonnet-4-6   # or whatever the custom endpoint expects
```

```python
from langchain_community.chat_models import ChatLiteLLM

llm = ChatLiteLLM(
    model=os.getenv("LITELLM_MODEL"),
    api_base=os.getenv("LITELLM_BASE_URL"),
)
structured_llm = llm.with_structured_output(TranslationBatchOutput)
```

### Configuration

`TRANSLATION_BATCH_SIZE` (from `.env`) is the primary tuning knob. Smaller batches give more focused translations; larger batches reduce API call overhead. Default 5 is conservative — increase if quality holds. All other config lives in the consolidated `.env` reference in the Project Structure section.

### Pre-translation: Token Substitution

Before any segment enters the translation call, protected terms (from Step 2) are substituted with their opaque tokens:

```python
def substitute_tokens(text: str, token_map: dict[str, str]) -> str:
    # token_map is inverted: term -> token, sorted longest-first
    for term, token in token_map.items():
        text = text.replace(term, token)
    return text
```

Only the tokens that are **present in the batch** are passed to the model in the prompt — not the full list. This keeps the instruction focused and reduces noise.

### Pydantic Schema

```python
class TranslatedSegment(BaseModel):
    segment_id: str
    translated_text: str

class TranslationBatchOutput(BaseModel):
    language: str
    translations: list[TranslatedSegment]
```

`segment_id` ties each translation back to its source segment unambiguously. The model cannot collapse, reorder, or drop segments without a validation error.

### Call Structure

Each batch call includes:

| Input | How it's passed | Why |
|---|---|---|
| Page screenshot | Image content block | Full visual context — same screenshot for every batch on that page |
| Segments (batch of N) | User message | `id`, `text` (token-substituted), `type`, `section`, `width_px` |
| Target language | System prompt + user message | Unambiguous instruction |
| Tokens present in batch | Explicit list in prompt | Constraint on what must not be translated |

### Prompt Design

**System prompt:**
```
You are a professional translator for a regulated financial trading platform.
Translate UI copy accurately, preserving tone, intent, and register.

CRITICAL — Protected tokens:
Any token matching the pattern __T{n}__ is a protected brand name, product name,
or market term. You MUST:
1. Never translate these tokens — keep them exactly as-is in English.
2. Preserve the exact count — if __T0__ appears twice in the source, it must
   appear exactly twice in the translation. No additions, no omissions.

Element type and section are provided as translation hints:
- h1/h2 in hero: punchy, short — do not expand
- button/CTA: imperative, concise — match approximate character count
- p in footer/legal: formal register
- nav: single words or short phrases only
```

**User message (per batch):**
```
Translate the following {N} segments to {language}.

Tokens present in this batch that must not be translated:
- __T0__ = [term] (brand)
- __T2__ = [term] (market)

Segments:
[
  { "id": "seg_001", "text": "__T3__ for Anyone Anywhere Anytime", "type": "h1", "section": "hero", "width_px": 600 },
  { "id": "seg_004", "text": "Explore __T0__", "type": "button", "section": "hero", "width_px": 140 },
  ...
]
```

Revealing what each token represents in the prompt (e.g. `__T0__ = Deriv MT5`) is intentional — it helps the model correctly gender, inflect, and position the surrounding translated words without mistaking the token itself for something to translate.

### Post-translation: Token Restoration

After each batch response is validated by Pydantic:

```python
def restore_tokens(text: str, token_map: dict[str, str]) -> str:
    for token, term in token_map.items():
        text = text.replace(token, term)
    return text
```

### Cost & Token Tracking

LiteLLM with a custom base URL cannot reliably auto-lookup pricing — the model name may not exist in LiteLLM's pricing database. Cost is therefore calculated manually from explicit pricing constants and the token counts LiteLLM always returns on its OpenAI-compatible response object.

**Register custom pricing at startup** so `litellm.completion_cost()` works as a convenience if called elsewhere, and so pricing is defined in one place:

```python
import litellm

litellm.register_model({
    os.getenv("LITELLM_MODEL"): {
        "input_cost_per_token":  float(os.getenv("COST_PER_INPUT_TOKEN")),
        "output_cost_per_token": float(os.getenv("COST_PER_OUTPUT_TOKEN")),
    }
})
```

**Token extraction from LiteLLM response** — LiteLLM always returns an OpenAI-compatible `usage` object regardless of backend. Access it via the LangChain response metadata:

```python
response = structured_llm.invoke(messages)

# LangChain wraps LiteLLM usage in response_metadata
usage = response.response_metadata.get("usage", {})
input_tokens  = usage.get("prompt_tokens", 0)
output_tokens = usage.get("completion_tokens", 0)

cost_log.append({
    "step":          "step3_translation",
    "page":          page_url,
    "language":      language,
    "model":         os.getenv("LITELLM_MODEL"),
    "batch_index":   batch_index,
    "input_tokens":  input_tokens,
    "output_tokens": output_tokens,
    "estimated_usd": (input_tokens  * float(os.getenv("COST_PER_INPUT_TOKEN")))
                   + (output_tokens * float(os.getenv("COST_PER_OUTPUT_TOKEN"))),
    "latency_ms":    latency_ms,     # time the invoke() call took
    "timestamp":     datetime.utcnow().isoformat() + "Z",
})
```

**Required `.env` additions:**
```
COST_PER_INPUT_TOKEN=0.000003    # set to model's published rate
COST_PER_OUTPUT_TOKEN=0.000015   # set to model's published rate
```

Costs accumulate per language and per page, then surface as a summary table at the end of the run (Step 4).

### Output

Translations written to `output/translations/<language_code>/<page_slug>.json`:

```json
{
  "page": "https://deriv.com/",
  "language": "ar",
  "segments": [
    {
      "id": "seg_001",
      "source_text": "Trading for Anyone Anywhere Anytime",
      "translated_text": "التداول للجميع في أي مكان وأي وقت",
      "type": "h1",
      "section": "hero"
    }
  ]
}
```

---

## Step 3b: Programmatic QA

No LLM. Runs immediately after Step 3 completes for a page-language pair, before the LLM-based scoring in Step 5. Two deterministic checks that directly map to the hackathon's basic QA requirement: *"Flag untranslated segments or placeholder corruption."*

### Check 1 — Untranslated Segments

A segment is flagged as untranslated if its `translated_text` is identical to `source_text` after token restoration. This catches segments the LLM silently skipped or echoed back.

```python
def check_untranslated(source: str, translated: str, language: str) -> bool:
    # Allow identical text only for segments that are pure tokens/symbols
    if source.strip() == translated.strip():
        return True  # flag it
    return False
```

### Check 2 — Placeholder Corruption

Count occurrences of every `__T{n}__` token in the restored translated text and compare against the source. A mismatch means the model dropped, duplicated, or partially corrupted a protected term.

```python
import re

def check_token_corruption(source: str, translated: str, token_map: dict) -> list[str]:
    issues = []
    for token in token_map:
        src_count = source.count(token)
        out_count = translated.count(token)
        if src_count != out_count:
            issues.append(f"{token}: expected {src_count}, found {out_count}")
    return issues
```

Note: token corruption is checked on the text **before** restoration — if `__T2__` was mangled to `T2` or `__T 2__` by the model, restoration would silently fail and the original term would appear verbatim in the output. The count check catches this.

### Output

Per-page per-language, written to `output/qa/<lang>/<page_slug>_programmatic_qa.json`:

```json
{
    "page": "https://deriv.com/",
    "language": "ar",
    "total_segments": 84,
    "issues": [
        {
            "segment_id": "seg_deriv_home_042",
            "type": "untranslated",
            "source_text": "Learn more",
            "translated_text": "Learn more"
        },
        {
            "segment_id": "seg_deriv_home_017",
            "type": "token_corruption",
            "detail": "__T1__: expected 1, found 0",
            "source_text": "Explore __T1__",
            "translated_text": "استكشف"
        }
    ],
    "untranslated_count": 1,
    "corruption_count": 1,
    "pass": false
}
```

Issues are also emitted to the pipeline event log immediately:

```
[WARN]  [step3b] deriv.com → ar — seg_042: untranslated (source == output)
[WARN]  [step3b] deriv.com → ar — seg_017: token corruption — __T1__ expected 1, found 0
[INFO]  [step3b] deriv.com → ar — 84 segments checked, 2 issues found
```

---

## Step 4: Cost Reporting

No LLM calls. Pure aggregation over the shared `cost_log` populated throughout Steps 2, 3, and 5.

### Grouping Axes

The report groups along two axes independently:

**By language** — sums all entries where `language == lang`. This includes Step 3 translation costs and Step 5 QA costs. Step 2 is language-agnostic and reported separately as a shared infrastructure cost.

**By page** — sums all entries where `page == url` across all steps. This includes Step 2 term identification, all Step 3 translation batches for that page across all languages, and Step 5 QA for that page.

### Output

Printed to terminal via `rich.table` at the end of the run, and written to `output/cost_report.json`.

**By-language table:**
```
┌─────────────┬──────────────┬───────────────┬───────────────┬──────────────┐
│ Language    │ Input Tokens │ Output Tokens │ Total Tokens  │ Cost (USD)   │
├─────────────┼──────────────┼───────────────┼───────────────┼──────────────┤
│ ar          │ 48,200       │ 12,400        │ 60,600        │ $0.0183      │
│ fr          │ 46,800       │ 11,900        │ 58,700        │ $0.0177      │
│ es          │ 47,100       │ 12,100        │ 59,200        │ $0.0179      │
│ SHARED*     │ 3,200        │ 840           │ 4,040         │ $0.0013      │
├─────────────┼──────────────┼───────────────┼───────────────┼──────────────┤
│ TOTAL       │ 145,300      │ 37,240        │ 182,540       │ $0.0552      │
└─────────────┴──────────────┴───────────────┴───────────────┴──────────────┘
* Step 2 term identification — not attributed to any language
```

**By-page table:** same shape, rows are page slugs.

**Per-model breakdown** is also logged to `cost_report.json` — useful since Step 2 uses Gemini Flash Lite, Step 3 uses the LiteLLM-routed model, and Step 5 uses Gemini Pro.

---

## Step 5: QA Scoring (good to have)

### Tool: Gemini 3.1 Pro + LangChain + Pydantic

QA uses a more capable model than term identification — it requires holistic judgement across an entire page's translation, not pattern recognition over short strings. Gemini Pro is the right tradeoff: strong multilingual evaluation capability without the cost of running it per-segment.

Run once per page per language, after all batches for that page-language pair are complete.

### Pydantic Schema

Two attributes, no more:

```python
class QAResult(BaseModel):
    score:    int   # 0–100
    feedback: str   # freeform, actionable — issues found, segments flagged, suggestions
```

`score` is bounded to `int` — the model cannot return `87.5` or `"good"`. `feedback` is intentionally freeform: structured severity levels are stretch (Step 7 in the hackathon spec), and freeform is faster to implement and still useful as evidence.

### Input to the QA Call

```
System: You are a professional translation quality evaluator for a financial trading platform.
        Score the translation from 0 to 100 where:
        100 = perfect fidelity, tone, and term protection
        0   = completely broken or untranslated
        Be strict about protected term violations — any mistranslated brand or product name
        caps the score at 60.

[full-page screenshot as image content]

User:   Evaluate the Arabic translation of the following page.

        Protected terms (must appear unchanged in every translated segment):
        [list of all protected terms from Step 2]

        Original segments:
        [all source text segments with id, type, section]

        Translated segments:
        [all translated_text values with matching ids]
```

### Logging

QA results are logged to the shared `cost_log` (token usage, tagged `step=step5_qa`) and to `output/qa/<language_code>/<page_slug>_llm_qa.json`:

```json
{
    "page":     "https://deriv.com/",
    "language": "ar",
    "score":    87,
    "feedback": "Overall strong translation. Hero headline preserves urgency. Issues: seg_042 translates 'Deriv Bot' as 'روبوت ديريف' — protected term violation. seg_018 footer legal text is overly literal; register should be more formal. Navigation labels are accurate and concise."
}
```

QA score and a one-line summary are also printed to terminal at run end alongside the cost report:

```
[QA] deriv.com → ar   87/100  ⚠ 1 protected term violation (seg_042)
[QA] deriv.com → fr   94/100  ✓
```

---

<!-- Steps 6–8 stretch — to be populated if time allows -->
