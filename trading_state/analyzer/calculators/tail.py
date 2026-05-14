"""Tail Risk & Distribution-Shape calculators."""
from typing import Optional

from ..metrics_cache import AnalysisContext
from ..metrics_models import MetricResult
from ..metrics_stats import mean, quantile
from ..types import Params, ParamsTailRatio, ParamsVaR
from ._common import window_results


def calc_var(
    context: AnalysisContext, params: ParamsVaR
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _var_metric(
        full.daily_returns, params.confidence_level, params.window
    )
    windows = window_results(
        context.windows(),
        lambda w: _var_metric(
            w.daily_returns, params.confidence_level, params.window
        ),
    )
    return MetricResult(
        value, full.end_time, windows, notes='historical'
    )


def calc_cvar(
    context: AnalysisContext, params: ParamsVaR
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _cvar_metric(
        full.daily_returns, params.confidence_level, params.window
    )
    windows = window_results(
        context.windows(),
        lambda w: _cvar_metric(
            w.daily_returns, params.confidence_level, params.window
        ),
    )
    return MetricResult(
        value, full.end_time, windows, notes='historical'
    )


def calc_skewness(
    context: AnalysisContext, _params: Params | None
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _skewness(full.daily_returns)
    return MetricResult(value, full.end_time, [])


def calc_kurtosis(
    context: AnalysisContext, _params: Params | None
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _kurtosis(full.daily_returns)
    return MetricResult(value, full.end_time, [], notes='excess')


def calc_tail_ratio(
    context: AnalysisContext, params: ParamsTailRatio
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _tail_ratio(
        full.daily_returns, params.quantile, params.window
    )
    windows = window_results(
        context.windows(),
        lambda w: _tail_ratio(
            w.daily_returns, params.quantile, params.window
        ),
    )
    return MetricResult(value, full.end_time, windows)


def _var_metric(
    returns: list[float], confidence_level: float, window: int
) -> Optional[float]:
    data = returns[-window:] if window > 0 else returns
    if not data:
        return None
    cutoff = quantile(data, 1.0 - confidence_level)
    if cutoff is None:
        return None
    return -cutoff


def _cvar_metric(
    returns: list[float], confidence_level: float, window: int
) -> Optional[float]:
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
    if avg is None:
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


def _tail_ratio(
    returns: list[float], q: float, window: int
) -> Optional[float]:
    data = returns[-window:] if window > 0 else returns
    if not data:
        return None
    upper = quantile(data, q)
    lower = quantile(data, 1.0 - q)
    if upper is None or lower is None or lower == 0:
        return None
    return upper / abs(lower)
