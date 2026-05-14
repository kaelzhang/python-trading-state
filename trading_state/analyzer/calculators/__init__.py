"""
Per-metric calculators split by AnalyzerCategory. The
`METRIC_CALCULATORS` dispatch table below is the only surface
`PerformanceAnalyzer.analyze` consumes.

The lower-block re-exports under ``_xxx`` names preserve the pre-split
``trading_state.analyzer.metrics_calculators`` private surface so the
existing branch-coverage tests can keep poking at the same helpers
without needing to know the new submodule layout.
"""
from typing import Callable

from ..metrics_cache import AnalysisContext
from ..metrics_models import MetricResult, SkippedResult
from ..metrics_stats import mean
from ..types import AnalyzerType, Params

from . import _common, benchmark, returns, risk_adjusted, tail, volatility
from .benchmark import (
    calc_alpha,
    calc_beta,
    calc_correlation,
    calc_jensen_alpha,
    calc_tracking_error,
)
from .drawdown import (
    calc_average_drawdown,
    calc_mdd,
    calc_tuw,
    calc_ulcer_index,
)
from .returns import (
    calc_annualized_return,
    calc_cagr,
    calc_geometric_mean_return,
    calc_mean_return,
    calc_total_return,
)
from .risk_adjusted import (
    calc_burke_ratio,
    calc_calmar_ratio,
    calc_information_ratio,
    calc_m2,
    calc_mar_ratio,
    calc_pain_ratio,
    calc_sharpe_ratio,
    calc_sortino_ratio,
    calc_sterling_ratio,
    calc_treynor_ratio,
    calc_upi_martin_ratio,
)
from .tail import (
    calc_cvar,
    calc_kurtosis,
    calc_skewness,
    calc_tail_ratio,
    calc_var,
)
from .trade import (
    calc_expectancy,
    calc_kelly_criterion,
    calc_payoff_ratio,
    calc_profit_factor,
    calc_win_rate,
)
from .volatility import (
    calc_downside_deviation,
    calc_semi_variance,
    calc_volatility,
)


# ---------------------------------------------------------------------
# Back-compat aliases for the pre-split helper surface.
#
# `trading_state.analyzer.metrics_calculators` used to expose these
# private helpers at module level. The branch-coverage test suite
# treats them as private-but-importable. Re-exporting keeps that
# contract intact without polluting the per-category modules with
# duplicated names. Monkeypatching `mean` for skewness/kurtosis must
# now target the owning submodule (`tail.mean`); patching the alias
# here would not intercept the call site.
# ---------------------------------------------------------------------
_paired_daily_returns = _common.paired_daily_returns
_benchmark_for = _common.benchmark_for
_window_results = _common.window_results
_trade_window_results = _common.trade_window_results
_total_return_series = _common.total_return_series
_trade_series = _common.trade_series
_trade_extras = _common.trade_extras

_geometric_mean_daily = returns._geometric_mean_daily

_annualized_std = volatility.annualized_std
_downside_threshold = volatility.downside_threshold
_downside_deviation = volatility._downside_deviation
_downside_deviation_daily = volatility.downside_deviation_daily
_semi_variance = volatility._semi_variance

_sharpe_ratio = risk_adjusted._sharpe_ratio
_sortino_ratio = risk_adjusted._sortino_ratio
_treynor_ratio = risk_adjusted._treynor_ratio
_information_ratio = risk_adjusted._information_ratio
_m2 = risk_adjusted._m2
_calmar_ratio = risk_adjusted._calmar_ratio
_mar_ratio = risk_adjusted._mar_ratio
_upi_martin_ratio = risk_adjusted._upi_martin_ratio
_sterling_ratio = risk_adjusted._sterling_ratio
_burke_ratio = risk_adjusted._burke_ratio
_pain_ratio = risk_adjusted._pain_ratio

_var_metric = tail._var_metric
_cvar_metric = tail._cvar_metric
_skewness = tail._skewness
_kurtosis = tail._kurtosis
_tail_ratio = tail._tail_ratio

_alpha = benchmark._alpha
_jensen_alpha = benchmark._jensen_alpha
_beta = benchmark.beta
_correlation = benchmark._correlation
_tracking_error = benchmark._tracking_error
_weighted_beta = benchmark._weighted_beta
_benchmark_window = benchmark.benchmark_window


METRIC_CALCULATORS: dict[
    AnalyzerType,
    Callable[
        [AnalysisContext, Params | None],
        MetricResult | SkippedResult,
    ],
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
