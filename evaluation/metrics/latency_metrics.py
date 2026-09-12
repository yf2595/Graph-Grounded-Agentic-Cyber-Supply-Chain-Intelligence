"""Latency / resource / cost aggregates."""
from __future__ import annotations

import statistics


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 6) if xs else None


def _std(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return 0.0 if xs else None
    return round(statistics.stdev(xs), 6)


def latency_metrics(
    latencies_ms: list[float],
    llm_calls: list[int],
    graph_queries: list[int],
    in_tok: list[int] | None = None,
    out_tok: list[int] | None = None,
    costs: list[float] | None = None,
) -> dict:
    if not latencies_ms:
        return {
            "n": 0,
            "mean_latency_ms": None,
            "std_latency_ms": None,
            "median_latency_ms": None,
            "p95_latency_ms": None,
            "mean_llm_calls": None,
            "std_llm_calls": None,
            "mean_graph_queries": None,
            "mean_input_tokens": None,
            "std_input_tokens": None,
            "mean_output_tokens": None,
            "std_output_tokens": None,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "mean_cost_usd": None,
            "std_cost_usd": None,
            "total_cost_usd": 0.0,
        }
    xs = sorted(latencies_ms)
    p95 = xs[min(len(xs) - 1, int(0.95 * (len(xs) - 1)))]
    in_tok = in_tok or []
    out_tok = out_tok or []
    costs = costs or []
    return {
        "n": len(xs),
        "mean_latency_ms": _mean(latencies_ms),
        "std_latency_ms": _std(latencies_ms),
        "median_latency_ms": round(statistics.median(xs), 2),
        "p95_latency_ms": round(p95, 2),
        "mean_llm_calls": _mean([float(x) for x in llm_calls]) if llm_calls else 0.0,
        "std_llm_calls": _std([float(x) for x in llm_calls]) if llm_calls else 0.0,
        "mean_graph_queries": _mean([float(x) for x in graph_queries]) if graph_queries else 0.0,
        "mean_input_tokens": _mean([float(x) for x in in_tok]) if in_tok else 0.0,
        "std_input_tokens": _std([float(x) for x in in_tok]) if in_tok else 0.0,
        "mean_output_tokens": _mean([float(x) for x in out_tok]) if out_tok else 0.0,
        "std_output_tokens": _std([float(x) for x in out_tok]) if out_tok else 0.0,
        "total_input_tokens": int(sum(in_tok)) if in_tok else 0,
        "total_output_tokens": int(sum(out_tok)) if out_tok else 0,
        "mean_cost_usd": _mean(costs) if costs else 0.0,
        "std_cost_usd": _std(costs) if costs else 0.0,
        "total_cost_usd": round(sum(costs), 6) if costs else 0.0,
    }
