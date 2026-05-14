"""Drawdown & Path-Risk calculators."""
from ..metrics_cache import AnalysisContext
from ..metrics_models import MetricResult, MetricSeriesPoint
from ..types import Params
from ._common import window_results


def calc_mdd(
    context: AnalysisContext, _params: Params | None
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    stats = context.drawdown_stats(full)
    windows = window_results(
        context.windows(),
        lambda w: context.drawdown_stats(w).max_drawdown,
    )
    return MetricResult(
        stats.max_drawdown, full.end_time, windows, series=stats.series
    )


def calc_average_drawdown(
    context: AnalysisContext, _params: Params | None
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    stats = context.drawdown_stats(full)
    windows = window_results(
        context.windows(),
        lambda w: context.drawdown_stats(w).avg_drawdown,
    )
    return MetricResult(stats.avg_drawdown, full.end_time, windows)


def calc_tuw(
    context: AnalysisContext, _params: Params | None
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    stats = context.drawdown_stats(full)
    series = [
        MetricSeriesPoint(
            time=episode.recovery_time or full.end_time,
            value=episode.duration_days,
        )
        for episode in stats.episodes
    ]
    extras: dict[str, float] = {}
    if stats.tuw_avg_days is not None:
        extras['average_days'] = stats.tuw_avg_days
    if stats.tuw_current_days is not None:
        extras['current_days'] = stats.tuw_current_days
    return MetricResult(
        stats.tuw_max_days,
        full.end_time,
        [],
        series=series,
        extras=extras or None,
    )


def calc_ulcer_index(
    context: AnalysisContext, _params: Params | None
) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    stats = context.drawdown_stats(full)
    windows = window_results(
        context.windows(),
        lambda w: context.drawdown_stats(w).ulcer_index,
    )
    return MetricResult(stats.ulcer_index, full.end_time, windows)
