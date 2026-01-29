from datetime import datetime, timedelta
from decimal import Decimal
import math

import pytest

from trading_state.analyzer import AnalyzerType, PerformanceAnalyzer
from trading_state.analyzer import metrics_calculators, metrics_cache
from trading_state.analyzer.metrics_cache import (
    AnalysisContext,
    SeriesCache,
    compute_drawdown_stats,
)
from trading_state.analyzer.metrics_calculators import (
    calc_annualized_return,
    calc_alpha,
    calc_beta,
    calc_average_drawdown,
    calc_calmar_ratio,
    calc_cagr,
    calc_correlation,
    calc_cvar,
    calc_downside_deviation,
    calc_expectancy,
    calc_geometric_mean_return,
    calc_information_ratio,
    calc_jensen_alpha,
    calc_kelly_criterion,
    calc_kurtosis,
    calc_m2,
    calc_mean_return,
    calc_mdd,
    calc_pain_ratio,
    calc_payoff_ratio,
    calc_profit_factor,
    calc_semi_variance,
    calc_sharpe_ratio,
    calc_skewness,
    calc_sortino_ratio,
    calc_tail_ratio,
    calc_total_return,
    calc_tracking_error,
    calc_treynor_ratio,
    calc_ulcer_index,
    calc_var,
    calc_volatility,
    calc_win_rate,
)
from trading_state.analyzer.metrics_models import SkippedResult
from trading_state.analyzer.metrics_stats import (
    annualize_cagr,
    annualize_from_daily,
    compound_returns,
    daily_threshold,
    get_trading_days,
    mean,
    normalize_daily_return,
    quantile,
    risk_free_daily,
    weighted_covariance,
    weighted_mean,
    weighted_variance,
    is_order_snapshot,
)
from trading_state.analyzer.types import (
    ParamsAnnualizedReturn,
    ParamsBenchmarkRelative,
    ParamsCalmarRatio,
    ParamsDownsideDeviation,
    ParamsInformationRatio,
    ParamsM2,
    ParamsSemiVariance,
    ParamsSharpeRatio,
    ParamsSortinoRatio,
    ParamsTailRatio,
    ParamsTreynorRatio,
    ParamsVaR,
)
from trading_state.pnl import BenchmarkPerformance, PerformanceSnapshot


def _dec(value: float | Decimal) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def make_snapshot(
    time: datetime,
    value: float | Decimal,
    net_cash_flow: float | Decimal = 0,
    benchmarks: dict[str, float] | None = None,
    labels: dict | None = None,
) -> PerformanceSnapshot:
    bench_map = {}
    for asset, cum_return in (benchmarks or {}).items():
        bench_map[asset] = BenchmarkPerformance(
            asset=asset,
            price=_dec(1),
            benchmark_return=_dec(cum_return),
        )
    return PerformanceSnapshot(
        time=time,
        account_value=_dec(value),
        realized_pnl=_dec(0),
        positions={},
        benchmarks=bench_map,
        cash_flows={},
        net_cash_flow=_dec(net_cash_flow),
        labels=labels or {},
    )


def build_context(
    returns: list[float],
    bench_returns: list[float] | None = None,
    start_value: float = 100.0,
    start_time: datetime | None = None,
) -> tuple[AnalysisContext, list[datetime], list[float], list[float] | None]:
    if start_time is None:
        start_time = datetime(2024, 1, 1)
    times = [start_time + timedelta(days=i) for i in range(len(returns) + 1)]
    values: list[Decimal] = [_dec(start_value)]
    for ret in returns:
        values.append(values[-1] * (_dec(1) + _dec(ret)))
    bench_cum = None
    if bench_returns is not None:
        bench_cum = [0.0]
        growth = 1.0
        for ret in bench_returns:
            growth *= 1.0 + ret
            bench_cum.append(growth - 1.0)
    snapshots = []
    for index, time in enumerate(times):
        benchmarks = {}
        if bench_cum is not None:
            benchmarks = {'BTC': bench_cum[index]}
        snapshots.append(make_snapshot(time, values[index], benchmarks=benchmarks))
    cache = SeriesCache()
    cache.rebuild(snapshots)
    return AnalysisContext(cache), times, [float(v) for v in values], bench_cum


def sample_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_variance(values: list[float]) -> float:
    avg = sample_mean(values)
    return sum((val - avg) ** 2 for val in values) / (len(values) - 1)


def sample_std(values: list[float]) -> float:
    return math.sqrt(sample_variance(values))


def quantile_expected(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def test_metrics_stats_edge_cases():
    assert normalize_daily_return(None, 1) is None
    assert normalize_daily_return(0.1, 0) is None
    assert normalize_daily_return(-1.5, 2) == pytest.approx(-0.75)

    assert annualize_from_daily(None, 252) is None
    assert annualize_cagr(None, 1, 252) is None
    assert annualize_cagr(0.2, 0, 252) is None
    assert annualize_cagr(-1.1, 10, 252) is None

    assert mean([]) is None
    assert weighted_mean([], []) is None
    assert weighted_mean([1], [0]) is None

    assert weighted_variance([1], [1, 2]) is None
    assert weighted_variance([1, 2], [0, 0]) is None
    assert weighted_variance([1], [1]) is None

    assert weighted_covariance([], [], []) is None
    assert weighted_covariance([1], [1, 2], [1, 1]) is None
    assert weighted_covariance([1], [2], [0]) is None
    assert weighted_covariance([1], [2], [1]) is None

    assert quantile([], 0.5) is None
    assert quantile([1, 2, 3], -1) == 1
    assert quantile([1, 2, 3], 2) == 3
    assert quantile([1, 3], 0.5) == 2.0

    assert risk_free_daily(0.1, 0) == 0.0
    assert daily_threshold(0.1, 0) == 0.1

    assert get_trading_days(None, 300) == 300
    class DummyParams:
        trading_days = -5
    assert get_trading_days(DummyParams(), 252) == 252

    snapshot = make_snapshot(datetime(2024, 1, 1), 100, labels=['not-a-dict'])
    assert is_order_snapshot(snapshot) is False


def test_series_cache_and_context_edges(monkeypatch):
    empty_cache = SeriesCache()
    context = AnalysisContext(empty_cache)
    assert context.start_time is None
    assert context.end_time is None
    assert context.full_window() is None
    assert context.windows() == []
    assert context.window_by_label('1w') is None
    assert context.trade_full_window() is None
    assert context.trade_windows() == []

    start = datetime(2024, 1, 1)
    snapshots = [
        make_snapshot(start, 0, benchmarks={'BTC': 0.0}),
        make_snapshot(start + timedelta(days=1), 100, labels={'__ORDER__': True}),
        make_snapshot(start + timedelta(days=2), 110, benchmarks={'ETH': 0.1}, labels={'__ORDER__': True}),
    ]
    cache = SeriesCache()
    cache.rebuild(snapshots)

    assert cache.cumulative_returns[1] == pytest.approx(0.0)
    assert cache.benchmarks['btc'].cumulative_returns[1] is None
    assert cache.trade_points[0].pnl == pytest.approx(10.0)

    single_cache = SeriesCache()
    single_cache.rebuild([make_snapshot(start, 100)])
    single_context = AnalysisContext(single_cache)
    assert single_context.windows() == []

    monkeypatch.setattr(metrics_cache, 'DEFAULT_WINDOWS', (('zero', 0),), raising=False)
    patched_context = AnalysisContext(single_cache)
    assert patched_context.windows() == []


def test_drawdown_stats_current_episode_sets_tuw_max():
    start = datetime(2024, 1, 1)
    times = [start + timedelta(days=i) for i in range(3)]
    values = [100.0, 80.0, 70.0]
    stats = compute_drawdown_stats(times, values)
    assert stats.tuw_current_days is not None
    assert stats.tuw_max_days == stats.tuw_current_days


def test_analyzer_cache_paths(monkeypatch):
    analyzer = PerformanceAnalyzer([AnalyzerType.SHARPE_RATIO])
    assert analyzer.analyze() == {}

    start = datetime(2024, 1, 1)
    snapshots = [
        make_snapshot(start, 100),
        make_snapshot(start + timedelta(days=1), 110),
        make_snapshot(start + timedelta(days=2), 99),
    ]
    analyzer.add_snapshots(*snapshots)
    result = analyzer.analyze()
    assert AnalyzerType.SHARPE_RATIO in result
    assert analyzer._cache_mode == 'append'

    analyzer.analyze()
    assert analyzer._cache is not None
    assert len(analyzer._snapshots) == analyzer._cache.source_len

    analyzer._snapshots = analyzer._snapshots[:-1]
    analyzer.analyze()
    assert analyzer._cache_mode == 'rebuild'

    analyzer._cache_mode = 'rebuild'
    analyzer.analyze()

    analyzer._cache_mode = 'append'
    analyzer.add_snapshots(
        make_snapshot(start + timedelta(days=4), 101),
        make_snapshot(start + timedelta(days=3), 100),
    )
    analyzer.analyze()
    assert analyzer._cache_mode == 'rebuild'

    analyzer = PerformanceAnalyzer([AnalyzerType.TOTAL_RETURN])
    analyzer.add_snapshots(*snapshots)
    analyzer.analyze()
    analyzer._cache_mode = 'append'
    analyzer.add_snapshots(make_snapshot(start + timedelta(hours=1), 105))
    analyzer.analyze()
    assert analyzer._cache_mode == 'rebuild'

    monkeypatch.delitem(metrics_calculators.METRIC_CALCULATORS, AnalyzerType.TOTAL_RETURN, raising=False)
    analyzer = PerformanceAnalyzer([AnalyzerType.TOTAL_RETURN])
    analyzer.add_snapshots(*snapshots)
    result = analyzer.analyze()
    assert AnalyzerType.TOTAL_RETURN not in result


def test_metrics_calculators_core_values():
    returns = [0.1, -0.05, -0.02, 0.03]
    context, times, _, _ = build_context(returns)
    trading_days = 252

    total = calc_total_return(context, None)
    expected_total = compound_returns(returns)
    assert total.value == pytest.approx(expected_total)

    mean_daily = sample_mean(returns)
    annualized = calc_annualized_return(context, ParamsAnnualizedReturn(trading_days=trading_days))
    assert annualized.value == pytest.approx(mean_daily * trading_days)

    days = (times[-1] - times[0]).total_seconds() / 86400.0
    expected_cagr = (1.0 + expected_total) ** (365.0 / days) - 1.0
    cagr = calc_cagr(context, None)
    assert cagr.value == pytest.approx(expected_cagr)

    mean_return = calc_mean_return(context, None)
    assert mean_return.value == pytest.approx(mean_daily)

    geometric = calc_geometric_mean_return(context, None)
    expected_geo = (1.0 + expected_total) ** (1.0 / days) - 1.0
    assert geometric.value == pytest.approx(expected_geo)

    std = sample_std(returns)
    volatility = calc_volatility(context, ParamsAnnualizedReturn(trading_days=trading_days))
    assert volatility.value == pytest.approx(std * math.sqrt(trading_days))

    sharpe = calc_sharpe_ratio(context, ParamsSharpeRatio(risk_free_rate=0.0, trading_days=trading_days))
    assert sharpe.value == pytest.approx(mean_daily / std * math.sqrt(trading_days))

    negative = [r for r in returns if r < 0]
    negative_mean = sample_mean(negative)
    down_var = sum((val - negative_mean) ** 2 for val in negative) / (len(negative) - 1)
    down_daily = math.sqrt(down_var)
    downside = calc_downside_deviation(
        context,
        ParamsDownsideDeviation(minimum_acceptable_return=0.0, downside_threshold=0.0, trading_days=trading_days),
    )
    assert downside.value == pytest.approx(down_daily * math.sqrt(trading_days))

    sortino = calc_sortino_ratio(
        context,
        ParamsSortinoRatio(minimum_acceptable_return=0.0, downside_threshold=0.0, trading_days=trading_days),
    )
    assert sortino.value == pytest.approx(mean_daily / down_daily * math.sqrt(trading_days))

    semi_variance = calc_semi_variance(
        context,
        ParamsSemiVariance(minimum_acceptable_return=0.0, downside_threshold=0.0, trading_days=trading_days),
    )
    down_var_pop = sum((val - negative_mean) ** 2 for val in negative) / len(negative)
    assert semi_variance.value == pytest.approx(down_var_pop * trading_days)


def test_metrics_calculators_benchmark_relative_values():
    returns = [0.1, -0.05, -0.02, 0.03]
    bench_returns = [0.05, -0.02, -0.01, 0.02]
    context, _, _, _ = build_context(returns, bench_returns=bench_returns)
    trading_days = 252

    mean_port = sample_mean(returns)
    mean_bench = sample_mean(bench_returns)
    std_port = sample_std(returns)
    std_bench = sample_std(bench_returns)
    cov = sum((p - mean_port) * (b - mean_bench) for p, b in zip(returns, bench_returns)) / (len(returns) - 1)
    beta = cov / sample_variance(bench_returns)
    active = [p - b for p, b in zip(returns, bench_returns)]
    active_std = sample_std(active)
    active_mean = sample_mean(active)

    alpha = calc_alpha(context, ParamsBenchmarkRelative(benchmark='btc', window=0, trading_days=trading_days))
    assert alpha.value == pytest.approx((mean_port - mean_bench) * trading_days)

    jensen = calc_jensen_alpha(context, ParamsBenchmarkRelative(benchmark='btc', window=0, trading_days=trading_days))
    expected_jensen = (mean_port - beta * mean_bench) * trading_days
    assert jensen.value == pytest.approx(expected_jensen)

    beta_metric = calc_beta(context, ParamsBenchmarkRelative(benchmark='btc', window=0))
    assert beta_metric.value == pytest.approx(beta)

    corr = calc_correlation(context, ParamsBenchmarkRelative(benchmark='btc', window=0))
    expected_corr = cov / (std_port * std_bench)
    assert corr.value == pytest.approx(expected_corr)

    tracking = calc_tracking_error(context, ParamsBenchmarkRelative(benchmark='btc', window=0, trading_days=trading_days))
    assert tracking.value == pytest.approx(active_std * math.sqrt(trading_days))

    info = calc_information_ratio(
        context,
        ParamsInformationRatio(benchmark='btc', tracking_error_window=0, trading_days=trading_days),
    )
    assert info.value == pytest.approx(active_mean / active_std * math.sqrt(trading_days))

    treynor = calc_treynor_ratio(
        context,
        ParamsTreynorRatio(benchmark='btc', risk_free_rate=0.0, trading_days=trading_days),
    )
    assert treynor.value == pytest.approx(mean_port * trading_days / beta)

    sharpe = mean_port / std_port * math.sqrt(trading_days)
    m2 = calc_m2(context, ParamsM2(benchmark='btc', risk_free_rate=0.0, trading_days=trading_days))
    assert m2.value == pytest.approx(sharpe * (std_bench * math.sqrt(trading_days)))


def test_metrics_calculators_tail_and_distribution():
    returns = [0.1, -0.05, -0.02, 0.03]
    context, _, _, _ = build_context(returns)

    var = calc_var(context, ParamsVaR(confidence_level=0.95, window=0))
    cutoff = quantile_expected(returns, 0.05)
    assert var.value == pytest.approx(-cutoff)

    cvar = calc_cvar(context, ParamsVaR(confidence_level=0.95, window=0))
    tail = [r for r in returns if r <= cutoff]
    assert cvar.value == pytest.approx(-sample_mean(tail))

    skewness = calc_skewness(context, None)
    avg = sample_mean(returns)
    diffs = [r - avg for r in returns]
    m2 = sample_mean([d ** 2 for d in diffs])
    m3 = sample_mean([d ** 3 for d in diffs])
    expected_skew = m3 / (m2 ** 1.5)
    assert skewness.value == pytest.approx(expected_skew)

    kurtosis = calc_kurtosis(context, None)
    m4 = sample_mean([d ** 4 for d in diffs])
    expected_kurt = m4 / (m2 ** 2) - 3.0
    assert kurtosis.value == pytest.approx(expected_kurt)

    tail_ratio = calc_tail_ratio(context, ParamsTailRatio(quantile=0.75, window=0))
    upper = quantile_expected(returns, 0.75)
    lower = quantile_expected(returns, 0.25)
    assert tail_ratio.value == pytest.approx(upper / abs(lower))


def test_metrics_calculators_drawdown_metrics():
    returns = [0.1, -0.2, 0.05]
    context, _, _, _ = build_context(returns)

    mdd = calc_mdd(context, None)
    avg_drawdown = calc_average_drawdown(context, None)
    ulcer = calc_ulcer_index(context, None)
    pain = calc_pain_ratio(context, None)
    calmar = calc_calmar_ratio(context, ParamsCalmarRatio())

    assert mdd.value is not None
    assert avg_drawdown.value is not None
    assert ulcer.value is not None
    assert pain.value is not None
    assert calmar.value is not None


def test_trade_metrics_and_windows():
    start = datetime(2024, 1, 1)
    snapshots = [
        make_snapshot(start, 100, labels={'__ORDER__': True}),
        make_snapshot(start + timedelta(days=1), 110, labels={'__ORDER__': True}),
        make_snapshot(start + timedelta(days=2), 104, labels={'__ORDER__': True}),
        make_snapshot(start + timedelta(days=400), 104),
    ]
    cache = SeriesCache()
    cache.rebuild(snapshots)
    context = AnalysisContext(cache)

    win_rate = calc_win_rate(context, None)
    payoff = calc_payoff_ratio(context, None)
    expectancy = calc_expectancy(context, None)
    profit_factor = calc_profit_factor(context, None)
    kelly = calc_kelly_criterion(context, None)

    assert win_rate.value == pytest.approx(0.5)
    assert payoff.value == pytest.approx(10.0 / 6.0)
    assert expectancy.value == pytest.approx(2.0)
    assert profit_factor.value == pytest.approx(10.0 / 6.0)
    assert kelly.value == pytest.approx(0.2)

    assert context.trade_windows() == []


def test_metrics_calculators_skip_without_benchmark():
    context, _, _, _ = build_context([0.1, -0.1])
    result = calc_treynor_ratio(context, ParamsTreynorRatio(benchmark='btc'))
    assert isinstance(result, SkippedResult)
