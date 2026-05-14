"""Risk-Adjusted Performance Ratio calculators."""
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
    annualize_cagr,
    get_trading_days,
    mean,
    risk_free_daily,
    weighted_mean,
    weighted_std,
)
from ..types import (
    Params,
    ParamsCalmarRatio,
    ParamsInformationRatio,
    ParamsM2,
    ParamsSharpeRatio,
    ParamsSortinoRatio,
    ParamsTreynorRatio,
)
from ._common import (
    CALENDAR_DAYS,
    benchmark_for,
    paired_daily_returns,
    window_results,
)
from .benchmark import benchmark_window, beta
from .volatility import annualized_std, downside_deviation_daily, downside_threshold


def calc_sharpe_ratio(
    context: AnalysisContext, params: ParamsSharpeRatio
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    trading_days = get_trading_days(params)
    rf_daily = risk_free_daily(params.risk_free_rate, trading_days)
    value = _sharpe_ratio(
        full.daily_returns, full.daily_weights, rf_daily, trading_days
    )
    windows = window_results(
        context.windows(),
        lambda w: _sharpe_ratio(
            w.daily_returns, w.daily_weights, rf_daily, trading_days
        ),
    )
    return MetricResult(value, full.end_time, windows)


def calc_sortino_ratio(
    context: AnalysisContext, params: ParamsSortinoRatio
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    trading_days = get_trading_days(params)
    threshold = downside_threshold(
        params.minimum_acceptable_return,
        params.downside_threshold,
        trading_days,
    )
    value = _sortino_ratio(
        full.daily_returns, full.daily_weights, threshold, trading_days
    )
    windows = window_results(
        context.windows(),
        lambda w: _sortino_ratio(
            w.daily_returns, w.daily_weights, threshold, trading_days
        ),
    )
    return MetricResult(value, full.end_time, windows)


def calc_treynor_ratio(
    context: AnalysisContext, params: ParamsTreynorRatio
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    bench = benchmark_for(context, params.benchmark)
    if bench is None:
        return SkippedResult(f'benchmark {params.benchmark} not available')
    rf_daily = risk_free_daily(
        params.risk_free_rate, params.trading_days
    )
    value = _treynor_ratio(full, bench, rf_daily, params.trading_days)
    windows = window_results(
        context.windows(),
        lambda w: _treynor_ratio(w, bench, rf_daily, params.trading_days),
    )
    return MetricResult(value, full.end_time, windows)


def calc_information_ratio(
    context: AnalysisContext, params: ParamsInformationRatio
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    bench = benchmark_for(context, params.benchmark)
    if bench is None:
        return SkippedResult(f'benchmark {params.benchmark} not available')
    value = _information_ratio(
        full, bench, params.tracking_error_window, params.trading_days
    )
    windows = window_results(
        context.windows(),
        lambda w: _information_ratio(
            w, bench, params.tracking_error_window, params.trading_days
        ),
    )
    return MetricResult(value, full.end_time, windows)


def calc_m2(
    context: AnalysisContext, params: ParamsM2
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    bench = benchmark_for(context, params.benchmark)
    if bench is None:
        return SkippedResult(f'benchmark {params.benchmark} not available')
    rf_daily = risk_free_daily(
        params.risk_free_rate, params.trading_days
    )
    value = _m2(full, bench, rf_daily, params.trading_days)
    windows = window_results(
        context.windows(),
        lambda w: _m2(w, bench, rf_daily, params.trading_days),
    )
    return MetricResult(value, full.end_time, windows)


def calc_calmar_ratio(
    context: AnalysisContext, params: ParamsCalmarRatio
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _calmar_ratio(context, full, params.risk_free_rate)
    windows = window_results(
        context.windows(),
        lambda w: _calmar_ratio(context, w, params.risk_free_rate),
    )
    return MetricResult(value, full.end_time, windows)


def calc_mar_ratio(
    context: AnalysisContext, _params: Params | None
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _mar_ratio(context, full)
    windows = window_results(
        context.windows(), lambda w: _mar_ratio(context, w)
    )
    return MetricResult(value, full.end_time, windows)


def calc_upi_martin_ratio(
    context: AnalysisContext, _params: Params | None
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _upi_martin_ratio(context, full)
    windows = window_results(
        context.windows(), lambda w: _upi_martin_ratio(context, w)
    )
    return MetricResult(value, full.end_time, windows)


def calc_sterling_ratio(
    context: AnalysisContext, _params: Params | None
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _sterling_ratio(context, full)
    windows = window_results(
        context.windows(), lambda w: _sterling_ratio(context, w)
    )
    return MetricResult(value, full.end_time, windows)


def calc_burke_ratio(
    context: AnalysisContext, _params: Params | None
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _burke_ratio(context, full)
    windows = window_results(
        context.windows(), lambda w: _burke_ratio(context, w)
    )
    return MetricResult(value, full.end_time, windows)


def calc_pain_ratio(
    context: AnalysisContext, _params: Params | None
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _pain_ratio(context, full)
    windows = window_results(
        context.windows(), lambda w: _pain_ratio(context, w)
    )
    return MetricResult(value, full.end_time, windows)


def _sharpe_ratio(
    daily_returns: list[float],
    daily_weights: list[float],
    rf_daily: float,
    trading_days: int,
) -> Optional[float]:
    mean_daily = weighted_mean(daily_returns, daily_weights)
    if mean_daily is None:
        return None
    std = weighted_std(daily_returns, daily_weights)
    if std is None or std == 0:
        return None
    return (mean_daily - rf_daily) / std * sqrt(trading_days)


def _sortino_ratio(
    daily_returns: list[float],
    daily_weights: list[float],
    threshold: float,
    trading_days: int,
) -> Optional[float]:
    mean_daily = weighted_mean(daily_returns, daily_weights)
    if mean_daily is None:
        return None
    downside = downside_deviation_daily(
        daily_returns, daily_weights, threshold
    )
    if downside is None or downside == 0:
        return None
    return (mean_daily - threshold) / downside * sqrt(trading_days)


def _treynor_ratio(
    window: WindowData,
    bench: BenchmarkSeries,
    rf_daily: float,
    trading_days: int,
) -> Optional[float]:
    b = beta(window, bench, 0)
    if b is None or b == 0:
        return None
    mean_daily = weighted_mean(window.daily_returns, window.daily_weights)
    if mean_daily is None:
        return None
    excess = mean_daily - rf_daily
    return excess * trading_days / b


def _information_ratio(
    window: WindowData,
    bench: BenchmarkSeries,
    lookback: int,
    trading_days: int,
) -> Optional[float]:
    port, b, weights = paired_daily_returns(window, bench)
    active = [p - bb for p, bb in zip(port, b)]
    if lookback > 0 and len(active) > lookback:
        active = active[-lookback:]
        weights = weights[-lookback:]
    mean_active = weighted_mean(active, weights)
    std_active = weighted_std(active, weights)
    if mean_active is None or std_active is None or std_active == 0:
        return None
    return mean_active / std_active * sqrt(trading_days)


def _m2(
    window: WindowData,
    bench: BenchmarkSeries,
    rf_daily: float,
    trading_days: int,
) -> Optional[float]:
    sharpe = _sharpe_ratio(
        window.daily_returns, window.daily_weights, rf_daily, trading_days
    )
    if sharpe is None:
        return None
    bw = benchmark_window(window, bench)
    if bw is None:
        return None
    bench_vol = annualized_std(bw, trading_days)
    if bench_vol is None:
        return None
    return (rf_daily * trading_days) + sharpe * bench_vol


def _calmar_ratio(
    context: AnalysisContext,
    window: WindowData,
    risk_free_rate: float,
) -> Optional[float]:
    stats = context.drawdown_stats(window)
    if stats.max_drawdown is None or stats.max_drawdown == 0:
        return None
    days = (window.end_time - window.start_time).total_seconds() / 86400.0
    cagr = annualize_cagr(window.cumulative_return, days, CALENDAR_DAYS)
    if cagr is None:
        return None
    return (cagr - risk_free_rate) / stats.max_drawdown


def _mar_ratio(
    context: AnalysisContext, window: WindowData
) -> Optional[float]:
    stats = context.drawdown_stats(window)
    if stats.max_drawdown is None or stats.max_drawdown == 0:
        return None
    days = (window.end_time - window.start_time).total_seconds() / 86400.0
    cagr = annualize_cagr(window.cumulative_return, days, CALENDAR_DAYS)
    if cagr is None:
        return None
    return cagr / stats.max_drawdown


def _upi_martin_ratio(
    context: AnalysisContext, window: WindowData
) -> Optional[float]:
    stats = context.drawdown_stats(window)
    if stats.ulcer_index is None or stats.ulcer_index == 0:
        return None
    days = (window.end_time - window.start_time).total_seconds() / 86400.0
    cagr = annualize_cagr(window.cumulative_return, days, CALENDAR_DAYS)
    if cagr is None:
        return None
    return cagr / stats.ulcer_index


def _sterling_ratio(
    context: AnalysisContext, window: WindowData
) -> Optional[float]:
    stats = context.drawdown_stats(window)
    if stats.avg_drawdown is None or stats.avg_drawdown == 0:
        return None
    days = (window.end_time - window.start_time).total_seconds() / 86400.0
    cagr = annualize_cagr(window.cumulative_return, days, CALENDAR_DAYS)
    if cagr is None:
        return None
    return cagr / stats.avg_drawdown


def _burke_ratio(
    context: AnalysisContext, window: WindowData
) -> Optional[float]:
    stats = context.drawdown_stats(window)
    if not stats.episodes:
        return None
    squared = [ep.depth ** 2 for ep in stats.episodes]
    denom = sqrt(mean(squared)) if squared else None
    if denom is None or denom == 0:
        return None
    days = (window.end_time - window.start_time).total_seconds() / 86400.0
    cagr = annualize_cagr(window.cumulative_return, days, CALENDAR_DAYS)
    if cagr is None:
        return None
    return cagr / denom


def _pain_ratio(
    context: AnalysisContext, window: WindowData
) -> Optional[float]:
    stats = context.drawdown_stats(window)
    if stats.pain_index is None or stats.pain_index == 0:
        return None
    days = (window.end_time - window.start_time).total_seconds() / 86400.0
    cagr = annualize_cagr(window.cumulative_return, days, CALENDAR_DAYS)
    if cagr is None:
        return None
    return cagr / stats.pain_index
