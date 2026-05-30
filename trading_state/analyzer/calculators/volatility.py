"""Volatility & Downside Risk calculators."""
from math import sqrt
from typing import Optional

from ..metrics_cache import AnalysisContext
from ..metrics_models import MetricResult, WindowData
from ..metrics_stats import (
    daily_threshold,
    get_trading_days,
    weighted_std,
    weighted_variance,
)
from ..types import (
    ParamsAnnualizedReturn,
    ParamsDownsideDeviation,
    ParamsSemiVariance,
)
from ._common import window_results


def calc_volatility(
    context: AnalysisContext, params: ParamsAnnualizedReturn
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    std = weighted_std(full.daily_returns, full.daily_weights)
    trading_days = get_trading_days(params)
    value = std * sqrt(trading_days) if std is not None else None
    windows = window_results(
        context.windows(), lambda w: annualized_std(w, trading_days)
    )
    return MetricResult(value, full.end_time, windows)


def calc_downside_deviation(
    context: AnalysisContext, params: ParamsDownsideDeviation
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
    value = _downside_deviation(
        full.daily_returns, full.daily_weights, threshold, trading_days
    )
    windows = window_results(
        context.windows(),
        lambda w: _downside_deviation(
            w.daily_returns, w.daily_weights, threshold, trading_days
        ),
    )
    return MetricResult(value, full.end_time, windows)


def calc_semi_variance(
    context: AnalysisContext, params: ParamsSemiVariance
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
    value = _semi_variance(
        full.daily_returns, full.daily_weights, threshold, trading_days
    )
    windows = window_results(
        context.windows(),
        lambda w: _semi_variance(
            w.daily_returns, w.daily_weights, threshold, trading_days
        ),
    )
    return MetricResult(
        value, full.end_time, windows, notes='annualized'
    )


# Re-exported so risk_adjusted's sharpe/sortino can reuse the same
# helpers without depending on volatility's public calc_* surface.

def annualized_std(window: WindowData, trading_days: int) -> Optional[float]:
    std = weighted_std(window.daily_returns, window.daily_weights)
    if std is None:
        return None
    return std * sqrt(trading_days)


def downside_threshold(
    minimum_acceptable_return: float,
    downside_threshold_param: float,
    trading_days: int,
) -> float:
    threshold = (
        downside_threshold_param
        if downside_threshold_param != 0.0
        else minimum_acceptable_return
    )
    return daily_threshold(threshold, trading_days)


def downside_deviation_daily(
    daily_returns: list[float],
    daily_weights: list[float],
    threshold: float,
) -> Optional[float]:
    filtered: list[float] = []
    weights: list[float] = []
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


def _downside_deviation(
    daily_returns: list[float],
    daily_weights: list[float],
    threshold: float,
    trading_days: int,
) -> Optional[float]:
    daily = downside_deviation_daily(daily_returns, daily_weights, threshold)
    if daily is None:
        return None
    return daily * sqrt(trading_days)


def _semi_variance(
    daily_returns: list[float],
    daily_weights: list[float],
    threshold: float,
    trading_days: int,
) -> Optional[float]:
    filtered: list[float] = []
    weights: list[float] = []
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
