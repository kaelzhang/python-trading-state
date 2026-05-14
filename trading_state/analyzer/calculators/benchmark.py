"""Benchmark-Relative & Attribution calculators."""
from math import sqrt
from typing import Optional

from ..metrics_cache import AnalysisContext
from ..metrics_models import (
    BenchmarkSeries,
    MetricResult,
    SkippedResult,
    WindowData,
)
from ..metrics_stats import (
    compound_returns,
    risk_free_daily,
    weighted_correlation,
    weighted_covariance,
    weighted_mean,
    weighted_std,
    weighted_variance,
)
from ..types import ParamsBenchmarkRelative
from ._common import benchmark_for, paired_daily_returns, window_results


def calc_alpha(
    context: AnalysisContext, params: ParamsBenchmarkRelative
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    benchmark = benchmark_for(context, params.benchmark)
    if benchmark is None:
        return SkippedResult(f'benchmark {params.benchmark} not available')
    value = _alpha(full, benchmark, params.window, params.trading_days)
    windows = window_results(
        context.windows(),
        lambda w: _alpha(w, benchmark, params.window, params.trading_days),
    )
    return MetricResult(value, full.end_time, windows)


def calc_jensen_alpha(
    context: AnalysisContext, params: ParamsBenchmarkRelative
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    benchmark = benchmark_for(context, params.benchmark)
    if benchmark is None:
        return SkippedResult(f'benchmark {params.benchmark} not available')
    rf_daily = risk_free_daily(
        params.risk_free_rate, params.trading_days
    )
    value = _jensen_alpha(
        full, benchmark, rf_daily, params.window, params.trading_days
    )
    windows = window_results(
        context.windows(),
        lambda w: _jensen_alpha(
            w, benchmark, rf_daily, params.window, params.trading_days
        ),
    )
    return MetricResult(value, full.end_time, windows)


def calc_beta(
    context: AnalysisContext, params: ParamsBenchmarkRelative
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    benchmark = benchmark_for(context, params.benchmark)
    if benchmark is None:
        return SkippedResult(f'benchmark {params.benchmark} not available')
    value = beta(full, benchmark, params.window)
    windows = window_results(
        context.windows(),
        lambda w: beta(w, benchmark, params.window),
    )
    return MetricResult(value, full.end_time, windows)


def calc_correlation(
    context: AnalysisContext, params: ParamsBenchmarkRelative
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    benchmark = benchmark_for(context, params.benchmark)
    if benchmark is None:
        return SkippedResult(f'benchmark {params.benchmark} not available')
    value = _correlation(full, benchmark, params.window)
    windows = window_results(
        context.windows(),
        lambda w: _correlation(w, benchmark, params.window),
    )
    return MetricResult(value, full.end_time, windows)


def calc_tracking_error(
    context: AnalysisContext, params: ParamsBenchmarkRelative
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    benchmark = benchmark_for(context, params.benchmark)
    if benchmark is None:
        return SkippedResult(f'benchmark {params.benchmark} not available')
    value = _tracking_error(
        full, benchmark, params.window, params.trading_days
    )
    windows = window_results(
        context.windows(),
        lambda w: _tracking_error(
            w, benchmark, params.window, params.trading_days
        ),
    )
    return MetricResult(value, full.end_time, windows)


# Helpers reused by other categories (risk_adjusted's treynor / m2).

def beta(
    window: WindowData, benchmark: BenchmarkSeries, lookback: int
) -> Optional[float]:
    port, bench, weights = paired_daily_returns(window, benchmark)
    if lookback > 0 and len(port) > lookback:
        port = port[-lookback:]
        bench = bench[-lookback:]
        weights = weights[-lookback:]
    return _weighted_beta(port, bench, weights)


def benchmark_window(
    window: WindowData, benchmark: BenchmarkSeries
) -> Optional[WindowData]:
    if not benchmark.return_points:
        return None
    times = window.times
    values: list[float] = []
    for idx in range(window.start_index, window.end_index + 1):
        cum = benchmark.cumulative_returns[idx]
        if cum is None:
            return None
        values.append(1.0 + cum)
    return_points = benchmark.return_points[
        window.start_index:window.end_index
    ]
    period_returns = [
        rp.period_return for rp in return_points
        if rp.period_return is not None
    ]
    daily_returns = [
        rp.daily_return for rp in return_points
        if rp.daily_return is not None and rp.days is not None
    ]
    daily_weights = [
        rp.days for rp in return_points
        if rp.daily_return is not None and rp.days is not None
    ]
    cumulative_return = compound_returns(period_returns)
    return WindowData(
        label=window.label,
        start_index=window.start_index,
        end_index=window.end_index,
        start_time=window.start_time,
        end_time=window.end_time,
        times=times,
        values=values,
        return_points=return_points,
        period_returns=period_returns,
        daily_returns=daily_returns,
        daily_weights=daily_weights,
        cumulative_return=cumulative_return,
    )


def _alpha(
    window: WindowData,
    benchmark: BenchmarkSeries,
    lookback: int,
    trading_days: int,
) -> Optional[float]:
    port, bench, weights = paired_daily_returns(window, benchmark)
    if lookback > 0 and len(port) > lookback:
        port = port[-lookback:]
        bench = bench[-lookback:]
        weights = weights[-lookback:]
    mean_port = weighted_mean(port, weights)
    mean_bench = weighted_mean(bench, weights)
    if mean_port is None or mean_bench is None:
        return None
    return (mean_port - mean_bench) * trading_days


def _jensen_alpha(
    window: WindowData,
    benchmark: BenchmarkSeries,
    rf_daily: float,
    lookback: int,
    trading_days: int,
) -> Optional[float]:
    port, bench, weights = paired_daily_returns(window, benchmark)
    if lookback > 0 and len(port) > lookback:
        port = port[-lookback:]
        bench = bench[-lookback:]
        weights = weights[-lookback:]
    b = _weighted_beta(port, bench, weights)
    if b is None:
        return None
    mean_port = weighted_mean(port, weights)
    mean_bench = weighted_mean(bench, weights)
    if mean_port is None or mean_bench is None:
        return None
    alpha_daily = (mean_port - rf_daily) - b * (mean_bench - rf_daily)
    return alpha_daily * trading_days


def _correlation(
    window: WindowData, benchmark: BenchmarkSeries, lookback: int
) -> Optional[float]:
    port, bench, weights = paired_daily_returns(window, benchmark)
    if lookback > 0 and len(port) > lookback:
        port = port[-lookback:]
        bench = bench[-lookback:]
        weights = weights[-lookback:]
    return weighted_correlation(port, bench, weights)


def _tracking_error(
    window: WindowData,
    benchmark: BenchmarkSeries,
    lookback: int,
    trading_days: int,
) -> Optional[float]:
    port, bench, weights = paired_daily_returns(window, benchmark)
    active = [p - b for p, b in zip(port, bench)]
    if lookback > 0 and len(active) > lookback:
        active = active[-lookback:]
        weights = weights[-lookback:]
    std = weighted_std(active, weights)
    if std is None:
        return None
    return std * sqrt(trading_days)


def _weighted_beta(
    xs: list[float], ys: list[float], weights: list[float]
) -> Optional[float]:
    cov = weighted_covariance(xs, ys, weights)
    if cov is None:
        return None
    var = weighted_variance(ys, weights)
    if var is None or var == 0:
        return None
    return cov / var
