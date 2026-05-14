"""
Shared helpers used by every category calculator. Internal to the
`trading_state.analyzer.calculators` package — not part of the public
analyzer surface.
"""
from typing import Callable, Optional

from ..metrics_cache import AnalysisContext
from ..metrics_models import (
    MetricSeriesPoint,
    MetricWindow,
    BenchmarkSeries,
    WindowData,
    TradeWindowData,
    TradeSummary,
)


CALENDAR_DAYS = 365


def paired_daily_returns(
    window: WindowData,
    benchmark: BenchmarkSeries,
) -> tuple[list[float], list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    weights: list[float] = []
    for index, rp in enumerate(window.return_points):
        bench_rp = benchmark.return_points[window.start_index + index]
        if rp.daily_return is None or rp.days is None:
            continue
        if bench_rp.daily_return is None or bench_rp.days is None:
            continue
        xs.append(rp.daily_return)
        ys.append(bench_rp.daily_return)
        weights.append(rp.days)
    return xs, ys, weights


def benchmark_for(
    context: AnalysisContext,
    benchmark: str,
) -> Optional[BenchmarkSeries]:
    if not benchmark:
        return None
    return context.benchmarks.get(benchmark.lower())


def window_results(
    windows: list[WindowData],
    calculator: Callable[[WindowData], Optional[float]],
) -> list[MetricWindow]:
    results: list[MetricWindow] = []
    for window in windows:
        value = calculator(window)
        if value is None:
            continue
        results.append(
            MetricWindow(
                label=window.label,
                start=window.start_time,
                end=window.end_time,
                value=value,
            )
        )
    return results


def trade_window_results(
    windows: list[TradeWindowData],
    calculator: Callable[[TradeWindowData], Optional[float]],
) -> list[MetricWindow]:
    results: list[MetricWindow] = []
    for window in windows:
        value = calculator(window)
        if value is None:
            continue
        results.append(
            MetricWindow(
                label=window.label,
                start=window.start,
                end=window.end,
                value=value,
            )
        )
    return results


def total_return_series(
    context: AnalysisContext,
) -> list[MetricSeriesPoint]:
    return [
        MetricSeriesPoint(time=time, value=value)
        for time, value in zip(context.times, context.cumulative_returns)
    ]


def trade_series(points: list) -> list[MetricSeriesPoint]:
    return [
        MetricSeriesPoint(time=point.time, value=point.pnl)
        for point in points
    ]


def trade_extras(summary: TradeSummary) -> Optional[dict[str, float]]:
    extras: dict[str, float] = {
        'trade_count': float(summary.total),
        'win_count': float(summary.wins),
        'loss_count': float(summary.losses),
    }
    if summary.avg_win is not None:
        extras['avg_win'] = summary.avg_win
    if summary.avg_loss is not None:
        extras['avg_loss'] = summary.avg_loss
    if summary.total_profit is not None:
        extras['total_profit'] = summary.total_profit
    if summary.total_loss is not None:
        extras['total_loss'] = summary.total_loss
    return extras or None
