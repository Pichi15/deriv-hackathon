"""Step 4 — Cost aggregation + rich tables + cost_report.json."""
from __future__ import annotations

import json
from collections import defaultdict

from rich.console import Console
from rich.table import Table

from .config import COST_REPORT_PATH, cost_log, page_slug

console = Console()


def _empty():
    return {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "calls": 0}


def _aggregate(entries: list[dict], key_fn) -> dict:
    buckets: dict[str, dict] = defaultdict(_empty)
    for e in entries:
        k = key_fn(e)
        b = buckets[k]
        b["input_tokens"] += e["input_tokens"]
        b["output_tokens"] += e["output_tokens"]
        b["cost"] += e["estimated_usd"]
        b["calls"] += 1
    return buckets


def _fmt_table(title: str, buckets: dict, first_col: str) -> Table:
    table = Table(title=title, header_style="bold")
    table.add_column(first_col)
    table.add_column("Calls", justify="right")
    table.add_column("Input Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Total Tokens", justify="right")
    table.add_column("Cost (USD)", justify="right")

    total = _empty()
    for k in sorted(buckets):
        b = buckets[k]
        total_tok = b["input_tokens"] + b["output_tokens"]
        table.add_row(
            k,
            str(b["calls"]),
            f"{b['input_tokens']:,}",
            f"{b['output_tokens']:,}",
            f"{total_tok:,}",
            f"${b['cost']:.4f}",
        )
        total["calls"] += b["calls"]
        total["input_tokens"] += b["input_tokens"]
        total["output_tokens"] += b["output_tokens"]
        total["cost"] += b["cost"]

    table.add_section()
    table.add_row(
        "TOTAL",
        str(total["calls"]),
        f"{total['input_tokens']:,}",
        f"{total['output_tokens']:,}",
        f"{total['input_tokens'] + total['output_tokens']:,}",
        f"${total['cost']:.4f}",
        style="bold",
    )
    return table


def print_cost_report() -> None:
    if not cost_log:
        console.print("[yellow]No LLM calls recorded — skipping cost report.[/]")
        COST_REPORT_PATH.write_text(json.dumps({"entries": [], "summary": {}}, indent=2))
        return

    # By-language: lumps Step 2 under "SHARED" since it's language-agnostic.
    by_lang = _aggregate(cost_log, lambda e: e["language"] if e["language"] else "SHARED*")

    # By-page: uses page slug for compact display.
    by_page = _aggregate(cost_log, lambda e: page_slug(e["page"]))

    # By-step and by-model: useful diagnostic rollups saved to JSON only.
    by_step = _aggregate(cost_log, lambda e: e["step"])
    by_model = _aggregate(cost_log, lambda e: e["model"])

    console.print()
    console.print(_fmt_table("Costs by Language", by_lang, "Language"))
    console.print()
    console.print("* SHARED = Step 2 term identification (language-agnostic)", style="dim")
    console.print()
    console.print(_fmt_table("Costs by Page", by_page, "Page"))
    console.print()
    console.print(_fmt_table("Costs by Step", by_step, "Step"))
    console.print()
    console.print(_fmt_table("Costs by Model", by_model, "Model"))
    console.print()

    summary = {
        "totals": {
            "calls": len(cost_log),
            "input_tokens": sum(e["input_tokens"] for e in cost_log),
            "output_tokens": sum(e["output_tokens"] for e in cost_log),
            "cost_usd": sum(e["estimated_usd"] for e in cost_log),
        },
        "by_language": {k: v for k, v in by_lang.items()},
        "by_page": {k: v for k, v in by_page.items()},
        "by_step": {k: v for k, v in by_step.items()},
        "by_model": {k: v for k, v in by_model.items()},
    }
    COST_REPORT_PATH.write_text(
        json.dumps({"entries": cost_log, "summary": summary}, ensure_ascii=False, indent=2)
    )
    console.print(f"[green]Wrote {COST_REPORT_PATH}[/]")
