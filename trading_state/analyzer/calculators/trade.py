"""Trade-level statistics calculators."""
from ..metrics_cache import AnalysisContext
from ..metrics_models import MetricResult
from ..types import Params
from ._common import trade_extras, trade_series, trade_window_results


def calc_win_rate(
    context: AnalysisContext, _params: Params | None
) -> MetricResult:
    full = context.trade_full_window()
    if full is None:
        return MetricResult(None, None, [])
    summary = context.trade_summary(full)
    windows = trade_window_results(
        context.trade_windows(),
        lambda w: context.trade_summary(w).win_rate,
    )
    return MetricResult(
        summary.win_rate,
        full.end,
        windows,
        series=trade_series(full.points),
        notes='trade_pnl',
        extras=trade_extras(summary),
    )


def calc_payoff_ratio(
    context: AnalysisContext, _params: Params | None
) -> MetricResult:
    full = context.trade_full_window()
    if full is None:
        return MetricResult(None, None, [])
    summary = context.trade_summary(full)
    windows = trade_window_results(
        context.trade_windows(),
        lambda w: context.trade_summary(w).payoff_ratio,
    )
    return MetricResult(
        summary.payoff_ratio,
        full.end,
        windows,
        series=trade_series(full.points),
        notes='trade_pnl',
        extras=trade_extras(summary),
    )


def calc_expectancy(
    context: AnalysisContext, _params: Params | None
) -> MetricResult:
    full = context.trade_full_window()
    if full is None:
        return MetricResult(None, None, [])
    summary = context.trade_summary(full)
    windows = trade_window_results(
        context.trade_windows(),
        lambda w: context.trade_summary(w).expectancy,
    )
    return MetricResult(
        summary.expectancy,
        full.end,
        windows,
        series=trade_series(full.points),
        notes='trade_pnl',
        extras=trade_extras(summary),
    )


def calc_profit_factor(
    context: AnalysisContext, _params: Params | None
) -> MetricResult:
    full = context.trade_full_window()
    if full is None:
        return MetricResult(None, None, [])
    summary = context.trade_summary(full)
    windows = trade_window_results(
        context.trade_windows(),
        lambda w: context.trade_summary(w).profit_factor,
    )
    return MetricResult(
        summary.profit_factor,
        full.end,
        windows,
        series=trade_series(full.points),
        notes='trade_pnl',
        extras=trade_extras(summary),
    )


def calc_kelly_criterion(
    context: AnalysisContext, _params: Params | None
) -> MetricResult:
    full = context.trade_full_window()
    if full is None:
        return MetricResult(None, None, [])
    summary = context.trade_summary(full)
    windows = trade_window_results(
        context.trade_windows(),
        lambda w: context.trade_summary(w).kelly,
    )
    return MetricResult(
        summary.kelly,
        full.end,
        windows,
        series=trade_series(full.points),
        notes='trade_pnl',
        extras=trade_extras(summary),
    )
