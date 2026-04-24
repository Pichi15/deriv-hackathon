"""Step 1 — Playwright extraction.

Extracts one full-page screenshot + a list of translatable segments per page,
writing to output/screenshots/<slug>.png and output/segments/<slug>_segments.json.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from playwright.async_api import async_playwright

from .config import MAX_CONCURRENCY, SCREENSHOTS_DIR, SEGMENTS_DIR, page_slug
from .logging_utils import info, warn

# Regex for dynamic tokens/placeholders the spec calls out: {{var}}, {0}, %s, __X__
TOKEN_PATTERNS = [
    re.compile(r"\{\{[^{}]+\}\}"),
    re.compile(r"\{\d+\}"),
    re.compile(r"%[sdif]"),
    re.compile(r"__[A-Za-z0-9_]+__"),
]


# JS run in the page: collect every translatable element with text + metadata.
# Done in one evaluate() call so we pay a single DOM-walk cost and keep ordering.
EXTRACT_JS = r"""
() => {
    const TRUSTPILOT_HOSTS = ['trustpilot.com', 'widget.trustpilot'];
    const EXCLUDE_TAGS = new Set(['SCRIPT', 'STYLE', 'META', 'NOSCRIPT', 'LINK', 'HEAD', 'IFRAME', 'SVG', 'TEMPLATE']);

    function xpathOf(el) {
        if (!el || el.nodeType !== 1) return '';
        if (el === document.body) return '/html/body';
        const parent = el.parentElement;
        if (!parent) return '';
        const siblings = Array.from(parent.children).filter(c => c.tagName === el.tagName);
        const idx = siblings.indexOf(el) + 1;
        const tag = el.tagName.toLowerCase();
        return xpathOf(parent) + '/' + tag + '[' + idx + ']';
    }

    function nearestSection(el) {
        let cur = el;
        while (cur && cur !== document.body) {
            const tag = cur.tagName ? cur.tagName.toLowerCase() : '';
            if (['header', 'footer', 'nav', 'main', 'section', 'aside'].includes(tag)) {
                return tag;
            }
            const cls = (cur.className && typeof cur.className === 'string') ? cur.className.toLowerCase() : '';
            if (cls) {
                if (/hero/.test(cls))        return 'hero';
                if (/footer/.test(cls))      return 'footer';
                if (/header|nav/.test(cls))  return 'header';
                if (/testimonial/.test(cls)) return 'testimonial';
                if (/cta/.test(cls))         return 'cta';
            }
            cur = cur.parentElement;
        }
        return 'body';
    }

    function inTrustpilot(el) {
        let cur = el;
        while (cur && cur !== document.body) {
            const cls = (cur.className && typeof cur.className === 'string') ? cur.className.toLowerCase() : '';
            const id  = (cur.id || '').toLowerCase();
            if (/trustpilot/.test(cls) || /trustpilot/.test(id)) return true;
            cur = cur.parentElement;
        }
        return false;
    }

    function isVisible(el) {
        const box = el.getBoundingClientRect();
        if (box.width === 0 || box.height === 0) return false;
        const style = window.getComputedStyle(el);
        if (style.visibility === 'hidden' || style.display === 'none' || parseFloat(style.opacity) === 0) return false;
        return true;
    }

    function immediateText(el) {
        // Text that belongs to THIS element (not swallowed by a child matcher).
        let s = '';
        for (const node of el.childNodes) {
            if (node.nodeType === 3) s += node.textContent;
        }
        return s.trim();
    }

    const selectors = [
        { sel: 'h1',  type: 'h1' },
        { sel: 'h2',  type: 'h2' },
        { sel: 'h3',  type: 'h3' },
        { sel: 'h4',  type: 'h4' },
        { sel: 'p',   type: 'p' },
        { sel: 'li',  type: 'li' },
        { sel: 'button', type: 'button' },
        { sel: 'a[role="button"]', type: 'button' },
        { sel: '[class*="cta" i]', type: 'cta' },
        { sel: 'nav a', type: 'nav-link' },
        { sel: 'main a:not([role="button"])', type: 'link' },
        { sel: 'img[alt]', type: 'alt' },
    ];

    const seen = new Set();
    const out = [];

    for (const {sel, type} of selectors) {
        const els = document.querySelectorAll(sel);
        for (const el of els) {
            if (EXCLUDE_TAGS.has(el.tagName)) continue;
            if (inTrustpilot(el)) continue;
            if (!isVisible(el) && type !== 'alt') continue;

            let text;
            let html = '';
            let href = null;
            if (type === 'alt') {
                text = (el.getAttribute('alt') || '').trim();
            } else {
                // For headings and paragraphs, use full inner text so we keep span/strong content.
                // For buttons/links/li/cta also full inner text.
                text = (el.innerText || '').trim();
                html = el.innerHTML || '';
                if (el.tagName === 'A') {
                    href = el.getAttribute('href');
                }
            }

            if (!text) continue;
            if (text.length > 600) continue;  // skip long prose blobs that are really containers

            // Dedup by (tag, text) — Deriv renders nav twice (desktop + mobile).
            const key = type + '\x1f' + text;
            if (seen.has(key)) continue;
            seen.add(key);

            const box = el.getBoundingClientRect();
            out.push({
                type,
                text,
                html,
                href,
                xpath: xpathOf(el),
                section: nearestSection(el),
                width_px: Math.round(box.width),
            });
        }
    }

    return out;
};
"""


def _detect_tokens(text: str) -> list[str]:
    found: list[str] = []
    for pat in TOKEN_PATTERNS:
        found.extend(pat.findall(text))
    # Dedupe, preserve order
    return list(dict.fromkeys(found))


async def _extract_one(page_url: str, browser, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        slug = page_slug(page_url)
        screenshot_path = SCREENSHOTS_DIR / f"{slug}.png"
        segments_path = SEGMENTS_DIR / f"{slug}_segments.json"

        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        try:
            await page.goto(page_url, wait_until="networkidle", timeout=45000)
        except Exception as e:
            warn(
                "step1",
                f"{page_url} — networkidle timed out, falling back to domcontentloaded: {e}",
            )
            await page.goto(page_url, wait_until="domcontentloaded", timeout=45000)

        await page.wait_for_timeout(1500)

        for label in ("Accept all", "Accept", "I agree", "Got it"):
            try:
                btn = page.get_by_role("button", name=label, exact=False)
                if await btn.count() > 0:
                    await btn.first.click(timeout=1000)
                    await page.wait_for_timeout(300)
                    break
            except Exception:
                pass

        await page.screenshot(path=str(screenshot_path), full_page=True)

        raw = await page.evaluate(EXTRACT_JS)
        await context.close()

        segments = []
        for i, row in enumerate(raw):
            seg_id = f"seg_{slug}_{i:03d}"
            tokens = _detect_tokens(row["text"])
            segments.append(
                {
                    "id": seg_id,
                    "type": row["type"],
                    "text": row["text"],
                    "html": row.get("html", ""),
                    "href": row.get("href"),
                    "xpath": row.get("xpath", ""),
                    "section": row.get("section", ""),
                    "width_px": row.get("width_px", 0),
                    "dynamic_tokens": tokens,
                }
            )

        payload = {
            "page": page_url,
            "screenshot_path": str(screenshot_path.relative_to(SCREENSHOTS_DIR.parent.parent)),
            "segments": segments,
        }
        segments_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        info(
            "step1",
            f"{page_url} — extracted {len(segments)} segments, screenshot saved",
            page=page_url,
            count=len(segments),
            screenshot=str(screenshot_path),
        )
        return payload


def _is_cached(page_url: str) -> bool:
    """Both segments JSON and screenshot PNG must exist for the cache to count."""
    slug = page_slug(page_url)
    return (SEGMENTS_DIR / f"{slug}_segments.json").exists() and (
        SCREENSHOTS_DIR / f"{slug}.png"
    ).exists()


async def _extract_all_async(to_scrape: list[str]) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
            await asyncio.gather(
                *(_extract_one(url, browser, semaphore) for url in to_scrape)
            )
        finally:
            await browser.close()


def extract_all_pages(pages: list[str], force: bool = False) -> None:
    """Scrape + screenshot each page. Skips pages with both artefacts already on disk.

    Scraping runs up to ``MAX_CONCURRENCY`` pages in parallel via async Playwright
    and an ``asyncio.Semaphore``. Cached pages don't consume a slot.

    Pass ``force=True`` to ignore the cache and re-scrape.
    """
    to_scrape = [url for url in pages if force or not _is_cached(url)]
    cached = [url for url in pages if url not in to_scrape]

    for url in cached:
        payload = load_segments(url)
        info(
            "step1",
            f"{url} — cache hit, skipping ({len(payload['segments'])} segments)",
            page=url,
            cached=True,
            count=len(payload["segments"]),
        )

    if not to_scrape:
        return

    asyncio.run(_extract_all_async(to_scrape))


def load_segments(page_url: str) -> dict:
    slug = page_slug(page_url)
    path = SEGMENTS_DIR / f"{slug}_segments.json"
    return json.loads(path.read_text())


def screenshot_path(page_url: str) -> Path:
    return SCREENSHOTS_DIR / f"{page_slug(page_url)}.png"
