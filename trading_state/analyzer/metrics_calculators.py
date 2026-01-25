from math import sqrt
from typing import Callable, Optional

from .types import (
    AnalyzerType,
    Params,
    ParamsAnnualizedReturn,
    ParamsSharpeRatio,
    ParamsSortinoRatio,
    ParamsTreynorRatio,
    ParamsInformationRatio,
    ParamsM2,
    ParamsCalmarRatio,
    ParamsDownsideDeviation,
    ParamsSemiVariance,
    ParamsVaR,
    ParamsTailRatio,
    ParamsBenchmarkRelative
)
from .metrics_cache import AnalysisContext
from .metrics_models import (
    MetricSeriesPoint,
    MetricWindow,
    MetricResult,
    SkippedResult,
    TradeWindowData,
    TradeSummary,
    WindowData,
    BenchmarkSeries
)
from .metrics_stats import (
    annualize_from_daily,
    annualize_cagr,
    compound_returns,
    mean,
    weighted_mean,
    weighted_variance,
    weighted_std,
    weighted_covariance,
    weighted_correlation,
    quantile,
    risk_free_daily,
    daily_threshold,
    get_trading_days
)

CALENDAR_DAYS = 365


def _paired_daily_returns(
    window: WindowData,
    benchmark: BenchmarkSeries
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


def _benchmark_for(
    context: AnalysisContext,
    benchmark: str
) -> Optional[BenchmarkSeries]:
    if not benchmark:
        return None
    return context.benchmarks.get(benchmark.lower())


def _window_results(
    windows: list[WindowData],
    calculator: Callable[[WindowData], Optional[float]]
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
                value=value
            )
        )
    return results


def _trade_window_results(
    windows: list[TradeWindowData],
    calculator: Callable[[TradeWindowData], Optional[float]]
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
                value=value
            )
        )
    return results


def _total_return_series(context: AnalysisContext) -> list[MetricSeriesPoint]:
    points: list[MetricSeriesPoint] = []
    for time, value in zip(context.times, context.cumulative_returns):
        points.append(MetricSeriesPoint(time=time, value=value))
    return points


def _trade_series(points: list) -> list[MetricSeriesPoint]:
    return [
        MetricSeriesPoint(time=point.time, value=point.pnl)
        for point in points
    ]


def _trade_extras(summary: TradeSummary) -> Optional[dict[str, float]]:
    extras: dict[str, float] = {
        'trade_count': float(summary.total),
        'win_count': float(summary.wins),
        'loss_count': float(summary.losses)
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


def calc_total_return(context: AnalysisContext, _params: Params | None) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = full.cumulative_return
    windows = _window_results(context.windows(), lambda w: w.cumulative_return)
    series = _total_return_series(context)
    return MetricResult(value, full.end_time, windows, series)


def calc_annualized_return(context: AnalysisContext, params: ParamsAnnualizedReturn) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    trading_days = get_trading_days(params)
    mean_daily = weighted_mean(full.daily_returns, full.daily_weights)
    value = annualize_from_daily(mean_daily, trading_days)
    windows = _window_results(
        context.windows(),
        lambda w: annualize_from_daily(
            weighted_mean(w.daily_returns, w.daily_weights),
            trading_days
        )
    )
    return MetricResult(value, full.end_time, windows)


def calc_cagr(context: AnalysisContext, _params: Params | None) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    days = (full.end_time - full.start_time).total_seconds() / 86400.0
    value = annualize_cagr(full.cumulative_return, days, CALENDAR_DAYS)
    windows = _window_results(
        context.windows(),
        lambda w: annualize_cagr(
            w.cumulative_return,
            (w.end_time - w.start_time).total_seconds() / 86400.0,
            CALENDAR_DAYS
        )
    )
    return MetricResult(value, full.end_time, windows)


def calc_mean_return(context: AnalysisContext, _params: Params | None) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = weighted_mean(full.daily_returns, full.daily_weights)
    windows = _window_results(
        context.windows(),
        lambda w: weighted_mean(w.daily_returns, w.daily_weights)
    )
    return MetricResult(value, full.end_time, windows, notes='daily')


def calc_geometric_mean_return(context: AnalysisContext, _params: Params | None) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _geometric_mean_daily(full)
    windows = _window_results(
        context.windows(),
        lambda w: _geometric_mean_daily(w)
    )
    return MetricResult(value, full.end_time, windows, notes='daily')


def calc_volatility(context: AnalysisContext, params: ParamsAnnualizedReturn) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    std = weighted_std(full.daily_returns, full.daily_weights)
    trading_days = get_trading_days(params)
    value = std * sqrt(trading_days) if std is not None else None
    windows = _window_results(
        context.windows(),
        lambda w: _annualized_std(w, trading_days)
    )
    return MetricResult(value, full.end_time, windows)


def calc_downside_deviation(context: AnalysisContext, params: ParamsDownsideDeviation) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    trading_days = get_trading_days(params)
    threshold = _downside_threshold(
        params.minimum_acceptable_return,
        params.downside_threshold,
        trading_days
    )
    value = _downside_deviation(full.daily_returns, full.daily_weights, threshold, trading_days)
    windows = _window_results(
        context.windows(),
        lambda w: _downside_deviation(w.daily_returns, w.daily_weights, threshold, trading_days)
    )
    return MetricResult(value, full.end_time, windows)


def calc_semi_variance(context: AnalysisContext, params: ParamsSemiVariance) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    trading_days = get_trading_days(params)
    threshold = _downside_threshold(
        params.minimum_acceptable_return,
        params.downside_threshold,
        trading_days
    )
    value = _semi_variance(full.daily_returns, full.daily_weights, threshold, trading_days)
    windows = _window_results(
        context.windows(),
        lambda w: _semi_variance(w.daily_returns, w.daily_weights, threshold, trading_days)
    )
    return MetricResult(value, full.end_time, windows, notes='annualized')


def calc_sharpe_ratio(context: AnalysisContext, params: ParamsSharpeRatio) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    trading_days = get_trading_days(params)
    rf_daily = risk_free_daily(params.risk_free_rate, trading_days)
    value = _sharpe_ratio(full.daily_returns, full.daily_weights, rf_daily, trading_days)
    windows = _window_results(
        context.windows(),
        lambda w: _sharpe_ratio(w.daily_returns, w.daily_weights, rf_daily, trading_days)
    )
    return MetricResult(value, full.end_time, windows)


def calc_sortino_ratio(context: AnalysisContext, params: ParamsSortinoRatio) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    trading_days = get_trading_days(params)
    threshold = _downside_threshold(
        params.minimum_acceptable_return,
        params.downside_threshold,
        trading_days
    )
    value = _sortino_ratio(full.daily_returns, full.daily_weights, threshold, trading_days)
    windows = _window_results(
        context.windows(),
        lambda w: _sortino_ratio(w.daily_returns, w.daily_weights, threshold, trading_days)
    )
    return MetricResult(value, full.end_time, windows)


def calc_treynor_ratio(context: AnalysisContext, params: ParamsTreynorRatio) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    benchmark = _benchmark_for(context, params.benchmark)
    if benchmark is None:
        return SkippedResult(f'benchmark {params.benchmark} not available')
    rf_daily = risk_free_daily(params.risk_free_rate, 252)
    value = _treynor_ratio(full, benchmark, rf_daily)
    windows = _window_results(
        context.windows(),
        lambda w: _treynor_ratio(w, benchmark, rf_daily)
    )
    return MetricResult(value, full.end_time, windows)


def calc_information_ratio(context: AnalysisContext, params: ParamsInformationRatio) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    benchmark = _benchmark_for(context, params.benchmark)
    if benchmark is None:
        return SkippedResult(f'benchmark {params.benchmark} not available')
    value = _information_ratio(full, benchmark, params.tracking_error_window)
    windows = _window_results(
        context.windows(),
        lambda w: _information_ratio(w, benchmark, params.tracking_error_window)
    )
    return MetricResult(value, full.end_time, windows)


def calc_m2(context: AnalysisContext, params: ParamsM2) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    benchmark = _benchmark_for(context, params.benchmark)
    if benchmark is None:
        return SkippedResult(f'benchmark {params.benchmark} not available')
    rf_daily = risk_free_daily(params.risk_free_rate, params.trading_days)
    value = _m2(full, benchmark, rf_daily, params.trading_days)
    windows = _window_results(
        context.windows(),
        lambda w: _m2(w, benchmark, rf_daily, params.trading_days)
    )
    return MetricResult(value, full.end_time, windows)


def calc_calmar_ratio(context: AnalysisContext, params: ParamsCalmarRatio) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _calmar_ratio(context, full, params.risk_free_rate)
    windows = _window_results(
        context.windows(),
        lambda w: _calmar_ratio(context, w, params.risk_free_rate)
    )
    return MetricResult(value, full.end_time, windows)


def calc_mar_ratio(context: AnalysisContext, _params: Params | None) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _mar_ratio(context, full)
    windows = _window_results(
        context.windows(),
        lambda w: _mar_ratio(context, w)
    )
    return MetricResult(value, full.end_time, windows)


def calc_upi_martin_ratio(context: AnalysisContext, _params: Params | None) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _upi_martin_ratio(context, full)
    windows = _window_results(
        context.windows(),
        lambda w: _upi_martin_ratio(context, w)
    )
    return MetricResult(value, full.end_time, windows)


def calc_sterling_ratio(context: AnalysisContext, _params: Params | None) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _sterling_ratio(context, full)
    windows = _window_results(
        context.windows(),
        lambda w: _sterling_ratio(context, w)
    )
    return MetricResult(value, full.end_time, windows)


def calc_burke_ratio(context: AnalysisContext, _params: Params | None) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _burke_ratio(context, full)
    windows = _window_results(
        context.windows(),
        lambda w: _burke_ratio(context, w)
    )
    return MetricResult(value, full.end_time, windows)


def calc_pain_ratio(context: AnalysisContext, _params: Params | None) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _pain_ratio(context, full)
    windows = _window_results(
        context.windows(),
        lambda w: _pain_ratio(context, w)
    )
    return MetricResult(value, full.end_time, windows)


def calc_mdd(context: AnalysisContext, _params: Params | None) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    stats = context.drawdown_stats(full)
    windows = _window_results(
        context.windows(),
        lambda w: context.drawdown_stats(w).max_drawdown
    )
    return MetricResult(stats.max_drawdown, full.end_time, windows, series=stats.series)


def calc_average_drawdown(context: AnalysisContext, _params: Params | None) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    stats = context.drawdown_stats(full)
    windows = _window_results(
        context.windows(),
        lambda w: context.drawdown_stats(w).avg_drawdown
    )
    return MetricResult(stats.avg_drawdown, full.end_time, windows)


def calc_tuw(context: AnalysisContext, _params: Params | None) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    stats = context.drawdown_stats(full)
    series = [
        MetricSeriesPoint(time=episode.recovery_time or full.end_time, value=episode.duration_days)
        for episode in stats.episodes
    ]
    extras = {}
    if stats.tuw_avg_days is not None:
        extras['average_days'] = stats.tuw_avg_days
    if stats.tuw_current_days is not None:
        extras['current_days'] = stats.tuw_current_days
    return MetricResult(stats.tuw_max_days, full.end_time, [], series=series, extras=extras or None)


def calc_ulcer_index(context: AnalysisContext, _params: Params | None) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    stats = context.drawdown_stats(full)
    windows = _window_results(
        context.windows(),
        lambda w: context.drawdown_stats(w).ulcer_index
    )
    return MetricResult(stats.ulcer_index, full.end_time, windows)


def calc_var(context: AnalysisContext, params: ParamsVaR) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _var_metric(full.daily_returns, params.confidence_level, params.window)
    windows = _window_results(
        context.windows(),
        lambda w: _var_metric(w.daily_returns, params.confidence_level, params.window)
    )
    return MetricResult(value, full.end_time, windows, notes='historical')


def calc_cvar(context: AnalysisContext, params: ParamsVaR) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _cvar_metric(full.daily_returns, params.confidence_level, params.window)
    windows = _window_results(
        context.windows(),
        lambda w: _cvar_metric(w.daily_returns, params.confidence_level, params.window)
    )
    return MetricResult(value, full.end_time, windows, notes='historical')


def calc_skewness(context: AnalysisContext, _params: Params | None) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _skewness(full.daily_returns)
    return MetricResult(value, full.end_time, [])


def calc_kurtosis(context: AnalysisContext, _params: Params | None) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _kurtosis(full.daily_returns)
    return MetricResult(value, full.end_time, [], notes='excess')


def calc_tail_ratio(context: AnalysisContext, params: ParamsTailRatio) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _tail_ratio(full.daily_returns, params.quantile, params.window)
    windows = _window_results(
        context.windows(),
        lambda w: _tail_ratio(w.daily_returns, params.quantile, params.window)
    )
    return MetricResult(value, full.end_time, windows)


def calc_alpha(context: AnalysisContext, params: ParamsBenchmarkRelative) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    benchmark = _benchmark_for(context, params.benchmark)
    if benchmark is None:
        return SkippedResult(f'benchmark {params.benchmark} not available')
    value = _alpha(full, benchmark, params.window)
    windows = _window_results(
        context.windows(),
        lambda w: _alpha(w, benchmark, params.window)
    )
    return MetricResult(value, full.end_time, windows)


def calc_jensen_alpha(context: AnalysisContext, params: ParamsBenchmarkRelative) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    benchmark = _benchmark_for(context, params.benchmark)
    if benchmark is None:
        return SkippedResult(f'benchmark {params.benchmark} not available')
    rf_daily = risk_free_daily(params.risk_free_rate, 252)
    value = _jensen_alpha(full, benchmark, rf_daily, params.window)
    windows = _window_results(
        context.windows(),
        lambda w: _jensen_alpha(w, benchmark, rf_daily, params.window)
    )
    return MetricResult(value, full.end_time, windows)


def calc_beta(context: AnalysisContext, params: ParamsBenchmarkRelative) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    benchmark = _benchmark_for(context, params.benchmark)
    if benchmark is None:
        return SkippedResult(f'benchmark {params.benchmark} not available')
    value = _beta(full, benchmark, params.window)
    windows = _window_results(
        context.windows(),
        lambda w: _beta(w, benchmark, params.window)
    )
    return MetricResult(value, full.end_time, windows)


def calc_correlation(context: AnalysisContext, params: ParamsBenchmarkRelative) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    benchmark = _benchmark_for(context, params.benchmark)
    if benchmark is None:
        return SkippedResult(f'benchmark {params.benchmark} not available')
    value = _correlation(full, benchmark, params.window)
    windows = _window_results(
        context.windows(),
        lambda w: _correlation(w, benchmark, params.window)
    )
    return MetricResult(value, full.end_time, windows)


def calc_tracking_error(context: AnalysisContext, params: ParamsBenchmarkRelative) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    benchmark = _benchmark_for(context, params.benchmark)
    if benchmark is None:
        return SkippedResult(f'benchmark {params.benchmark} not available')
    value = _tracking_error(full, benchmark, params.window)
    windows = _window_results(
        context.windows(),
        lambda w: _tracking_error(w, benchmark, params.window)
    )
    return MetricResult(value, full.end_time, windows)


def calc_win_rate(context: AnalysisContext, _params: Params | None) -> MetricResult:
    full = context.trade_full_window()
    if full is None:
        return MetricResult(None, None, [])
    summary = context.trade_summary(full)
    windows = _trade_window_results(
        context.trade_windows(),
        lambda w: context.trade_summary(w).win_rate
    )
    return MetricResult(
        summary.win_rate,
        full.end,
        windows,
        series=_trade_series(full.points),
        notes='trade_pnl',
        extras=_trade_extras(summary)
    )


def calc_payoff_ratio(context: AnalysisContext, _params: Params | None) -> MetricResult:
    full = context.trade_full_window()
    if full is None:
        return MetricResult(None, None, [])
    summary = context.trade_summary(full)
    windows = _trade_window_results(
        context.trade_windows(),
        lambda w: context.trade_summary(w).payoff_ratio
    )
    return MetricResult(
        summary.payoff_ratio,
        full.end,
        windows,
        series=_trade_series(full.points),
        notes='trade_pnl',
        extras=_trade_extras(summary)
    )


def calc_expectancy(context: AnalysisContext, _params: Params | None) -> MetricResult:
    full = context.trade_full_window()
    if full is None:
        return MetricResult(None, None, [])
    summary = context.trade_summary(full)
    windows = _trade_window_results(
        context.trade_windows(),
        lambda w: context.trade_summary(w).expectancy
    )
    return MetricResult(
        summary.expectancy,
        full.end,
        windows,
        series=_trade_series(full.points),
        notes='trade_pnl',
        extras=_trade_extras(summary)
    )


def calc_profit_factor(context: AnalysisContext, _params: Params | None) -> MetricResult:
    full = context.trade_full_window()
    if full is None:
        return MetricResult(None, None, [])
    summary = context.trade_summary(full)
    windows = _trade_window_results(
        context.trade_windows(),
        lambda w: context.trade_summary(w).profit_factor
    )
    return MetricResult(
        summary.profit_factor,
        full.end,
        windows,
        series=_trade_series(full.points),
        notes='trade_pnl',
        extras=_trade_extras(summary)
    )


def calc_kelly_criterion(context: AnalysisContext, _params: Params | None) -> MetricResult:
    full = context.trade_full_window()
    if full is None:
        return MetricResult(None, None, [])
    summary = context.trade_summary(full)
    windows = _trade_window_results(
        context.trade_windows(),
        lambda w: context.trade_summary(w).kelly
    )
    return MetricResult(
        summary.kelly,
        full.end,
        windows,
        series=_trade_series(full.points),
        notes='trade_pnl',
        extras=_trade_extras(summary)
    )


def _geometric_mean_daily(window: WindowData) -> Optional[float]:
    if window.cumulative_return is None:
        return None
    days = (window.end_time - window.start_time).total_seconds() / 86400.0
    if days <= 0:
        return None
    base = 1.0 + window.cumulative_return
    if base <= 0:
        return None
    return base ** (1.0 / days) - 1.0


def _annualized_std(window: WindowData, trading_days: int) -> Optional[float]:
    std = weighted_std(window.daily_returns, window.daily_weights)
    if std is None:
        return None
    return std * sqrt(trading_days)


def _downside_threshold(
    minimum_acceptable_return: float,
    downside_threshold: float,
    trading_days: int
) -> float:
    threshold = downside_threshold if downside_threshold != 0.0 else minimum_acceptable_return
    return daily_threshold(threshold, trading_days)


def _downside_deviation(
    daily_returns: list[float],
    daily_weights: list[float],
    threshold: float,
    trading_days: int
) -> Optional[float]:
    daily = _downside_deviation_daily(daily_returns, daily_weights, threshold)
    if daily is None:
        return None
    return daily * sqrt(trading_days)


def _downside_deviation_daily(
    daily_returns: list[float],
    daily_weights: list[float],
    threshold: float
) -> Optional[float]:
    filtered = []
    weights = []
    for value, weight in zip(daily_returns, daily_weights):
        if value < threshold:
            filtered.append(value - threshold)
            weights.append(weight)
    if not filtered:
        return None
    variance = weighted_variance(filtered, weights)
    if variance is None:
        return None
    return sqrt(variance)


def _semi_variance(
    daily_returns: list[float],
    daily_weights: list[float],
    threshold: float,
    trading_days: int
) -> Optional[float]:
    filtered = []
    weights = []
    for value, weight in zip(daily_returns, daily_weights):
        if value < threshold:
            filtered.append(value - threshold)
            weights.append(weight)
    if not filtered:
        return None
    variance = weighted_variance(filtered, weights, ddof=0.0)
    if variance is None:
        return None
    return variance * trading_days


def _sharpe_ratio(
    daily_returns: list[float],
    daily_weights: list[float],
    rf_daily: float,
    trading_days: int
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
    trading_days: int
) -> Optional[float]:
    mean_daily = weighted_mean(daily_returns, daily_weights)
    if mean_daily is None:
        return None
    downside = _downside_deviation_daily(daily_returns, daily_weights, threshold)
    if downside is None or downside == 0:
        return None
    return (mean_daily - threshold) / downside * sqrt(trading_days)


def _treynor_ratio(
    window: WindowData,
    benchmark: BenchmarkSeries,
    rf_daily: float
) -> Optional[float]:
    beta = _beta(window, benchmark, 0)
    if beta is None or beta == 0:
        return None
    mean_daily = weighted_mean(window.daily_returns, window.daily_weights)
    if mean_daily is None:
        return None
    excess = mean_daily - rf_daily
    return excess * 252 / beta


def _information_ratio(
    window: WindowData,
    benchmark: BenchmarkSeries,
    lookback: int
) -> Optional[float]:
    port, bench, weights = _paired_daily_returns(window, benchmark)
    active = [p - b for p, b in zip(port, bench)]
    if lookback > 0 and len(active) > lookback:
        active = active[-lookback:]
        weights = weights[-lookback:]
    mean_active = weighted_mean(active, weights)
    std_active = weighted_std(active, weights)
    if mean_active is None or std_active is None or std_active == 0:
        return None
    return mean_active / std_active * sqrt(252)


def _m2(
    window: WindowData,
    benchmark: BenchmarkSeries,
    rf_daily: float,
    trading_days: int
) -> Optional[float]:
    sharpe = _sharpe_ratio(window.daily_returns, window.daily_weights, rf_daily, trading_days)
    if sharpe is None:
        return None
    bench_window = _benchmark_window(window, benchmark)
    if bench_window is None:
        return None
    bench_vol = _annualized_std(bench_window, trading_days)
    if bench_vol is None:
        return None
    return (rf_daily * trading_days) + sharpe * bench_vol


def _calmar_ratio(
    context: AnalysisContext,
    window: WindowData,
    risk_free_rate: float
) -> Optional[float]:
    stats = context.drawdown_stats(window)
    if stats.max_drawdown is None or stats.max_drawdown == 0:
        return None
    days = (window.end_time - window.start_time).total_seconds() / 86400.0
    cagr = annualize_cagr(window.cumulative_return, days, CALENDAR_DAYS)
    if cagr is None:
        return None
    return (cagr - risk_free_rate) / stats.max_drawdown


def _mar_ratio(context: AnalysisContext, window: WindowData) -> Optional[float]:
    stats = context.drawdown_stats(window)
    if stats.max_drawdown is None or stats.max_drawdown == 0:
        return None
    days = (window.end_time - window.start_time).total_seconds() / 86400.0
    cagr = annualize_cagr(window.cumulative_return, days, CALENDAR_DAYS)
    if cagr is None:
        return None
    return cagr / stats.max_drawdown


def _upi_martin_ratio(context: AnalysisContext, window: WindowData) -> Optional[float]:
    stats = context.drawdown_stats(window)
    if stats.ulcer_index is None or stats.ulcer_index == 0:
        return None
    days = (window.end_time - window.start_time).total_seconds() / 86400.0
    cagr = annualize_cagr(window.cumulative_return, days, CALENDAR_DAYS)
    if cagr is None:
        return None
    return cagr / stats.ulcer_index


def _sterling_ratio(context: AnalysisContext, window: WindowData) -> Optional[float]:
    stats = context.drawdown_stats(window)
    if stats.avg_drawdown is None or stats.avg_drawdown == 0:
        return None
    days = (window.end_time - window.start_time).total_seconds() / 86400.0
    cagr = annualize_cagr(window.cumulative_return, days, CALENDAR_DAYS)
    if cagr is None:
        return None
    return cagr / stats.avg_drawdown


def _burke_ratio(context: AnalysisContext, window: WindowData) -> Optional[float]:
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


def _pain_ratio(context: AnalysisContext, window: WindowData) -> Optional[float]:
    stats = context.drawdown_stats(window)
    if stats.pain_index is None or stats.pain_index == 0:
        return None
    days = (window.end_time - window.start_time).total_seconds() / 86400.0
    cagr = annualize_cagr(window.cumulative_return, days, CALENDAR_DAYS)
    if cagr is None:
        return None
    return cagr / stats.pain_index


def _var_metric(returns: list[float], confidence_level: float, window: int) -> Optional[float]:
    data = returns[-window:] if window > 0 else returns
    if not data:
        return None
    cutoff = quantile(data, 1.0 - confidence_level)
    if cutoff is None:
        return None
    return -cutoff


def _cvar_metric(returns: list[float], confidence_level: float, window: int) -> Optional[float]:
    data = returns[-window:] if window > 0 else returns
    if not data:
        return None
    cutoff = quantile(data, 1.0 - confidence_level)
    if cutoff is None:
        return None
    tail = [r for r in data if r <= cutoff]
    if not tail:
        return None
    return -mean(tail)


def _skewness(returns: list[float]) -> Optional[float]:
    if len(returns) < 3:
        return None
    avg = mean(returns)
    if mean is None:
        return None
    diffs = [r - avg for r in returns]
    m2 = mean([d ** 2 for d in diffs])
    m3 = mean([d ** 3 for d in diffs])
    if m2 is None or m3 is None or m2 == 0:
        return None
    return m3 / (m2 ** 1.5)


def _kurtosis(returns: list[float]) -> Optional[float]:
    if len(returns) < 4:
        return None
    avg = mean(returns)
    if avg is None:
        return None
    diffs = [r - avg for r in returns]
    m2 = mean([d ** 2 for d in diffs])
    m4 = mean([d ** 4 for d in diffs])
    if m2 is None or m4 is None or m2 == 0:
        return None
    return m4 / (m2 ** 2) - 3.0


def _tail_ratio(returns: list[float], quantile: float, window: int) -> Optional[float]:
    data = returns[-window:] if window > 0 else returns
    if not data:
        return None
    upper = quantile(data, quantile)
    lower = quantile(data, 1.0 - quantile)
    if upper is None or lower is None or lower == 0:
        return None
    return upper / abs(lower)


def _alpha(window: WindowData, benchmark: BenchmarkSeries, lookback: int) -> Optional[float]:
    port, bench, weights = _paired_daily_returns(window, benchmark)
    if lookback > 0 and len(port) > lookback:
        port = port[-lookback:]
        bench = bench[-lookback:]
        weights = weights[-lookback:]
    mean_port = weighted_mean(port, weights)
    mean_bench = weighted_mean(bench, weights)
    if mean_port is None or mean_bench is None:
        return None
    return (mean_port - mean_bench) * 252


def _jensen_alpha(
    window: WindowData,
    benchmark: BenchmarkSeries,
    rf_daily: float,
    lookback: int
) -> Optional[float]:
    port, bench, weights = _paired_daily_returns(window, benchmark)
    if lookback > 0 and len(port) > lookback:
        port = port[-lookback:]
        bench = bench[-lookback:]
        weights = weights[-lookback:]
    beta = _weighted_beta(port, bench, weights)
    if beta is None:
        return None
    mean_port = weighted_mean(port, weights)
    mean_bench = weighted_mean(bench, weights)
    if mean_port is None or mean_bench is None:
        return None
    alpha_daily = (mean_port - rf_daily) - beta * (mean_bench - rf_daily)
    return alpha_daily * 252


def _beta(window: WindowData, benchmark: BenchmarkSeries, lookback: int) -> Optional[float]:
    port, bench, weights = _paired_daily_returns(window, benchmark)
    if lookback > 0 and len(port) > lookback:
        port = port[-lookback:]
        bench = bench[-lookback:]
        weights = weights[-lookback:]
    return _weighted_beta(port, bench, weights)


def _correlation(window: WindowData, benchmark: BenchmarkSeries, lookback: int) -> Optional[float]:
    port, bench, weights = _paired_daily_returns(window, benchmark)
    if lookback > 0 and len(port) > lookback:
        port = port[-lookback:]
        bench = bench[-lookback:]
        weights = weights[-lookback:]
    return weighted_correlation(port, bench, weights)


def _tracking_error(window: WindowData, benchmark: BenchmarkSeries, lookback: int) -> Optional[float]:
    port, bench, weights = _paired_daily_returns(window, benchmark)
    active = [p - b for p, b in zip(port, bench)]
    if lookback > 0 and len(active) > lookback:
        active = active[-lookback:]
        weights = weights[-lookback:]
    std = weighted_std(active, weights)
    if std is None:
        return None
    return std * sqrt(252)


def _weighted_beta(xs: list[float], ys: list[float], weights: list[float]) -> Optional[float]:
    cov = weighted_covariance(xs, ys, weights)
    if cov is None:
        return None
    var = weighted_variance(ys, weights)
    if var is None or var == 0:
        return None
    return cov / var


def _benchmark_window(window: WindowData, benchmark: BenchmarkSeries) -> Optional[WindowData]:
    if not benchmark.return_points:
        return None
    times = window.times
    values = []
    for idx in range(window.start_index, window.end_index + 1):
        cum = benchmark.cumulative_returns[idx]
        if cum is None:
            return None
        values.append(1.0 + cum)
    return_points = benchmark.return_points[window.start_index:window.end_index]
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
        cumulative_return=cumulative_return
    )


UNSUPPORTED_METRICS: dict[AnalyzerType, str] = {
    # AnalyzerType.TRANSACTION_COST_DRAG: 'requires fee/slippage and execution data',
    # AnalyzerType.TURNOVER: 'requires trade notional to separate price moves from trades',
    # AnalyzerType.SLIPPAGE_SENSITIVITY: 'requires slippage scenarios or execution data',
}


METRIC_CALCULATORS: dict[
    AnalyzerType,
    Callable[
        [AnalysisContext, Params | None],
        MetricResult | SkippedResult
    ]
] = {
    AnalyzerType.TOTAL_RETURN: calc_total_return,
    AnalyzerType.ANNUALIZED_RETURN: calc_annualized_return,
    AnalyzerType.CAGR: calc_cagr,
    AnalyzerType.MEAN_RETURN: calc_mean_return,
    AnalyzerType.GEOMETRIC_MEAN_RETURN: calc_geometric_mean_return,
    AnalyzerType.SHARPE_RATIO: calc_sharpe_ratio,
    AnalyzerType.SORTINO_RATIO: calc_sortino_ratio,
    AnalyzerType.TREYNOR_RATIO: calc_treynor_ratio,
    AnalyzerType.IR: calc_information_ratio,
    AnalyzerType.M2: calc_m2,
    AnalyzerType.CALMAR_RATIO: calc_calmar_ratio,
    AnalyzerType.MAR: calc_mar_ratio,
    AnalyzerType.UPI_MARTIN_RATIO: calc_upi_martin_ratio,
    AnalyzerType.STERLING_RATIO: calc_sterling_ratio,
    AnalyzerType.BURKE_RATIO: calc_burke_ratio,
    AnalyzerType.PAIN_RATIO: calc_pain_ratio,
    AnalyzerType.VOLATILITY: calc_volatility,
    AnalyzerType.DOWNSIDE_DEVIATION: calc_downside_deviation,
    AnalyzerType.SEMI_VARIANCE: calc_semi_variance,
    AnalyzerType.MDD: calc_mdd,
    AnalyzerType.AVERAGE_DRAWDOWN: calc_average_drawdown,
    AnalyzerType.TUW: calc_tuw,
    AnalyzerType.UI: calc_ulcer_index,
    AnalyzerType.VAR: calc_var,
    AnalyzerType.CVAR: calc_cvar,
    AnalyzerType.SKEWNESS: calc_skewness,
    AnalyzerType.KURTOSIS: calc_kurtosis,
    AnalyzerType.TAIL_RATIO: calc_tail_ratio,
    AnalyzerType.ALPHA: calc_alpha,
    AnalyzerType.JENSEN_ALPHA: calc_jensen_alpha,
    AnalyzerType.BETA: calc_beta,
    AnalyzerType.CORRELATION: calc_correlation,
    AnalyzerType.TE: calc_tracking_error,
    AnalyzerType.WIN_RATE: calc_win_rate,
    AnalyzerType.PAYOFF_RATIO: calc_payoff_ratio,
    AnalyzerType.EXPECTANCY: calc_expectancy,
    AnalyzerType.PROFIT_FACTOR: calc_profit_factor,
    AnalyzerType.KELLY_CRITERION: calc_kelly_criterion,
}
