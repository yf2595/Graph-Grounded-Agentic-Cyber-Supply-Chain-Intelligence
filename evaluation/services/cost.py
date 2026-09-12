"""Token-cost helpers for WAITI evaluation."""
from __future__ import annotations


DEFAULT_PRICING = {
    "gpt-5.4": {"input_per_1m": 2.50, "output_per_1m": 15.00},
    "gpt-4.1": {"input_per_1m": 2.00, "output_per_1m": 8.00},
}


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing: dict | None = None,
) -> float:
    """Estimate USD cost from token counts using per-1M rates."""
    table = pricing or DEFAULT_PRICING
    rates = table.get(model) or table.get("default") or DEFAULT_PRICING["gpt-5.4"]
    inp = float(rates.get("input_per_1m", 2.50))
    out = float(rates.get("output_per_1m", 15.00))
    return round((input_tokens / 1_000_000.0) * inp + (output_tokens / 1_000_000.0) * out, 8)
