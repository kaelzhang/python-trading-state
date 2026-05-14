"""Return & Growth calculators."""
from typing import Optional

from ..metrics_cache import AnalysisContext
from ..metrics_models import MetricResult, WindowData
from ..metrics_stats import (
    annualize_cagr,
    annualize_from_daily,
    get_trading_days,
    weighted_mean,
)
from ..types import Params, ParamsAnnualizedReturn
from ._common import CALENDAR_DAYS, total_return_series, window_results


def calc_total_return(
    context: AnalysisContext, _params: Params | None
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = full.cumulative_return
    windows = window_results(
        context.windows(), lambda w: w.cumulative_return
    )
    series = total_return_series(context)
    return MetricResult(value, full.end_time, windows, series)


def calc_annualized_return(
    context: AnalysisContext,
    params: ParamsAnnualizedReturn,
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    trading_days = get_trading_days(params)
    mean_daily = weighted_mean(full.daily_returns, full.daily_weights)
    value = annualize_from_daily(mean_daily, trading_days)
    windows = window_results(
        context.windows(),
        lambda w: annualize_from_daily(
            weighted_mean(w.daily_returns, w.daily_weights),
            trading_days,
        ),
    )
    return MetricResult(value, full.end_time, windows)


def calc_cagr(
    context: AnalysisContext, _params: Params | None
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    days = (full.end_time - full.start_time).total_seconds() / 86400.0
    value = annualize_cagr(full.cumulative_return, days, CALENDAR_DAYS)
    windows = window_results(
        context.windows(),
        lambda w: annualize_cagr(
            w.cumulative_return,
            (w.end_time - w.start_time).total_seconds() / 86400.0,
            CALENDAR_DAYS,
        ),
    )
    return MetricResult(value, full.end_time, windows)


def calc_mean_return(
    context: AnalysisContext, _params: Params | None
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = weighted_mean(full.daily_returns, full.daily_weights)
    windows = window_results(
        context.windows(),
        lambda w: weighted_mean(w.daily_returns, w.daily_weights),
    )
    return MetricResult(value, full.end_time, windows, notes='daily')


def calc_geometric_mean_return(
    context: AnalysisContext, _params: Params | None
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _geometric_mean_daily(full)
    windows = window_results(
        context.windows(), lambda w: _geometric_mean_daily(w)
    )
    return MetricResult(value, full.end_time, windows, notes='daily')


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
