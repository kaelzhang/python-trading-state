from datetime import datetime, timedelta
from decimal import Decimal

from trading_state.analyzer import AnalyzerType, PerformanceAnalyzer
from trading_state.analyzer import metrics_calculators as mc
from trading_state.analyzer.metrics_cache import AnalysisContext, SeriesCache, compute_drawdown_stats
from trading_state.analyzer.metrics_models import (
    BenchmarkSeries,
    DrawdownEpisode,
    DrawdownStats,
    MetricSeriesPoint,
    ReturnPoint,
    TradePerformancePoint,
    TradeWindowData,
    WindowData,
)
from trading_state.analyzer.metrics_stats import weighted_correlation
from trading_state.analyzer.metrics_models import SkippedResult
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


def make_window(
    return_points: list[ReturnPoint],
    daily_returns: list[float],
    daily_weights: list[float],
    cumulative_return: float | None,
    times: list[datetime] | None = None,
    values: list[float] | None = None,
    label: str = 'w',
) -> WindowData:
    if times is None:
        start = datetime(2024, 1, 1)
        times = [start + timedelta(days=i) for i in range(len(return_points) + 1)]
    if values is None:
        values = [100.0 + i for i in range(len(times))]
    period_returns = [rp.period_return for rp in return_points if rp.period_return is not None]
    return WindowData(
        label=label,
        start_index=0,
        end_index=len(times) - 1,
        start_time=times[0],
        end_time=times[-1],
        times=times,
        values=values,
        return_points=return_points,
        period_returns=period_returns,
        daily_returns=daily_returns,
        daily_weights=daily_weights,
        cumulative_return=cumulative_return,
    )


def test_analyzer_cache_extend_path():
    analyzer = PerformanceAnalyzer([AnalyzerType.TOTAL_RETURN])
    start = datetime(2024, 1, 1)
    snapshots = [
        make_snapshot(start, 100),
        make_snapshot(start + timedelta(days=1), 101),
    ]
    analyzer.add_snapshots(*snapshots)
    analyzer.analyze()

    cache_len = analyzer._cache.source_len
    analyzer.add_snapshots(make_snapshot(start + timedelta(days=2), 102))
    analyzer.analyze()

    assert analyzer._cache.source_len == cache_len + 1
    assert analyzer._cache_mode == 'append'


def test_metrics_cache_windows_and_trade_windows():
    start = datetime(2024, 1, 1)
    snapshots = [
        make_snapshot(start, 100),
        make_snapshot(start + timedelta(days=1), 101),
        make_snapshot(start + timedelta(days=9), 102, labels={'__ORDER__': True}),
        make_snapshot(start + timedelta(days=10), 103, labels={'__ORDER__': True}),
    ]
    cache = SeriesCache()
    cache.rebuild(snapshots)
    context = AnalysisContext(cache)

    assert context.window_by_label('full') is not None
    assert context.window_by_label('1w') is not None
    assert context.trade_windows()


def test_metrics_cache_empty_drawdown_stats():
    stats = compute_drawdown_stats([], [])
    assert stats.max_drawdown is None
    assert stats.tuw_max_days is None


def test_metrics_stats_weighted_correlation_none():
    assert weighted_correlation([], [], []) is None


def test_metrics_calculators_empty_context_returns_none():
    context = AnalysisContext(SeriesCache())
    cases: list[tuple[callable, object | None]] = [
        (mc.calc_total_return, None),
        (mc.calc_annualized_return, ParamsAnnualizedReturn()),
        (mc.calc_cagr, None),
        (mc.calc_mean_return, None),
        (mc.calc_geometric_mean_return, None),
        (mc.calc_volatility, ParamsAnnualizedReturn()),
        (mc.calc_downside_deviation, ParamsDownsideDeviation()),
        (mc.calc_semi_variance, ParamsSemiVariance()),
        (mc.calc_sharpe_ratio, ParamsSharpeRatio()),
        (mc.calc_sortino_ratio, ParamsSortinoRatio()),
        (mc.calc_treynor_ratio, ParamsTreynorRatio()),
        (mc.calc_information_ratio, ParamsInformationRatio()),
        (mc.calc_m2, ParamsM2()),
        (mc.calc_calmar_ratio, ParamsCalmarRatio()),
        (mc.calc_mar_ratio, None),
        (mc.calc_upi_martin_ratio, None),
        (mc.calc_sterling_ratio, None),
        (mc.calc_burke_ratio, None),
        (mc.calc_pain_ratio, None),
        (mc.calc_mdd, None),
        (mc.calc_average_drawdown, None),
        (mc.calc_tuw, None),
        (mc.calc_ulcer_index, None),
        (mc.calc_var, ParamsVaR()),
        (mc.calc_cvar, ParamsVaR()),
        (mc.calc_skewness, None),
        (mc.calc_kurtosis, None),
        (mc.calc_tail_ratio, ParamsTailRatio()),
        (mc.calc_alpha, ParamsBenchmarkRelative()),
        (mc.calc_jensen_alpha, ParamsBenchmarkRelative()),
        (mc.calc_beta, ParamsBenchmarkRelative()),
        (mc.calc_correlation, ParamsBenchmarkRelative()),
        (mc.calc_tracking_error, ParamsBenchmarkRelative()),
        (mc.calc_win_rate, None),
        (mc.calc_payoff_ratio, None),
        (mc.calc_expectancy, None),
        (mc.calc_profit_factor, None),
        (mc.calc_kelly_criterion, None),
    ]
    for func, params in cases:
        result = func(context, params)
        assert result.value is None
        assert result.as_of is None
        assert result.windows == []


def test_metrics_calculators_skipped_without_benchmark():
    start = datetime(2024, 1, 1)
    snapshots = [
        make_snapshot(start, 100),
        make_snapshot(start + timedelta(days=1), 101),
    ]
    cache = SeriesCache()
    cache.rebuild(snapshots)
    context = AnalysisContext(cache)

    assert isinstance(mc.calc_treynor_ratio(context, ParamsTreynorRatio()), SkippedResult)
    assert isinstance(mc.calc_information_ratio(context, ParamsInformationRatio()), SkippedResult)
    assert isinstance(mc.calc_m2(context, ParamsM2()), SkippedResult)
    assert isinstance(mc.calc_alpha(context, ParamsBenchmarkRelative()), SkippedResult)
    assert isinstance(mc.calc_jensen_alpha(context, ParamsBenchmarkRelative()), SkippedResult)
    assert isinstance(mc.calc_beta(context, ParamsBenchmarkRelative()), SkippedResult)
    assert isinstance(mc.calc_correlation(context, ParamsBenchmarkRelative()), SkippedResult)
    assert isinstance(mc.calc_tracking_error(context, ParamsBenchmarkRelative()), SkippedResult)


def test_metrics_calculators_helper_branches(monkeypatch):
    base = datetime(2024, 1, 1)
    rp = ReturnPoint(time=base + timedelta(days=1), period_return=0.01, days=1.0, daily_return=0.01)
    window = make_window([rp], [0.01], [1.0], 0.01)
    benchmark = BenchmarkSeries(
        asset='BTC',
        cumulative_returns=[0.0, 0.01],
        return_points=[
            ReturnPoint(time=base + timedelta(days=1), period_return=None, days=None, daily_return=None)
        ],
    )
    xs, ys, weights = mc._paired_daily_returns(window, benchmark)
    assert xs == [] and ys == [] and weights == []

    assert mc._benchmark_for(AnalysisContext(SeriesCache()), '') is None

    trade_window = TradeWindowData(
        label='t',
        start=base,
        end=base,
        points=[TradePerformancePoint(time=base, pnl=1.0, return_pct=0.1)],
        pnls=[1.0],
        returns=[0.1],
    )
    assert mc._trade_window_results([trade_window], lambda _w: None) == []

    none_window = make_window([], [], [], None, times=[base, base + timedelta(days=1)])
    assert mc._geometric_mean_daily(none_window) is None

    zero_days_window = make_window([], [], [], 0.1, times=[base, base])
    assert mc._geometric_mean_daily(zero_days_window) is None

    negative_base_window = make_window([], [], [], -1.1, times=[base, base + timedelta(days=1)])
    assert mc._geometric_mean_daily(negative_base_window) is None

    assert mc._annualized_std(make_window([], [], [], 0.1), 252) is None
    assert mc._downside_deviation_daily([-0.1], [1.0], 0.0) is None
    assert mc._semi_variance([-0.1], [0.0], 0.0, 252) is None
    assert mc._sharpe_ratio([], [], 0.0, 252) is None
    assert mc._sortino_ratio([], [], 0.0, 252) is None

    rp2 = ReturnPoint(time=base + timedelta(days=2), period_return=0.02, days=1.0, daily_return=0.02)
    window_no_mean = make_window([rp, rp2], [], [], 0.02, times=[base, base + timedelta(days=1), base + timedelta(days=2)])
    bench_points = [
        ReturnPoint(time=base + timedelta(days=1), period_return=0.01, days=1.0, daily_return=0.01),
        ReturnPoint(time=base + timedelta(days=2), period_return=0.02, days=1.0, daily_return=0.02),
    ]
    bench_series = BenchmarkSeries(
        asset='BTC',
        cumulative_returns=[0.0, 0.01, 0.03],
        return_points=bench_points,
    )
    assert mc._treynor_ratio(window_no_mean, bench_series, 0.0, 252) is None

    window_for_m2 = make_window([rp, rp2], [0.01, 0.02], [1.0, 1.0], 0.02)
    empty_benchmark = BenchmarkSeries(asset='BTC', cumulative_returns=[0.0, 0.0], return_points=[])
    assert mc._m2(window_for_m2, empty_benchmark, 0.0, 252) is None

    bench_with_empty_daily = BenchmarkSeries(
        asset='BTC',
        cumulative_returns=[0.0, 0.01, 0.03],
        return_points=[
            ReturnPoint(time=base + timedelta(days=1), period_return=None, days=None, daily_return=None),
            ReturnPoint(time=base + timedelta(days=2), period_return=None, days=None, daily_return=None),
        ],
    )
    assert mc._m2(window_for_m2, bench_with_empty_daily, 0.0, 252) is None

    stats = DrawdownStats(
        series=[MetricSeriesPoint(time=base, value=0.1)],
        episodes=[],
        max_drawdown=0.1,
        avg_drawdown=0.2,
        ulcer_index=0.3,
        pain_index=0.4,
        tuw_max_days=None,
        tuw_avg_days=None,
        tuw_current_days=None,
    )
    monkeypatch.setattr(AnalysisContext, 'drawdown_stats', lambda _self, _w: stats)
    assert mc._calmar_ratio(AnalysisContext(SeriesCache()), none_window, 0.0) is None
    assert mc._mar_ratio(AnalysisContext(SeriesCache()), none_window) is None
    assert mc._upi_martin_ratio(AnalysisContext(SeriesCache()), none_window) is None
    assert mc._sterling_ratio(AnalysisContext(SeriesCache()), none_window) is None
    assert mc._pain_ratio(AnalysisContext(SeriesCache()), none_window) is None

    stats_zero = DrawdownStats(
        series=[],
        episodes=[DrawdownEpisode(peak_time=base, trough_time=base, recovery_time=base, depth=0.0, duration_days=1.0)],
        max_drawdown=0.1,
        avg_drawdown=0.2,
        ulcer_index=0.3,
        pain_index=0.4,
        tuw_max_days=None,
        tuw_avg_days=None,
        tuw_current_days=None,
    )
    monkeypatch.setattr(AnalysisContext, 'drawdown_stats', lambda _self, _w: stats_zero)
    assert mc._burke_ratio(AnalysisContext(SeriesCache()), window_for_m2) is None

    stats_nonzero = DrawdownStats(
        series=[],
        episodes=[DrawdownEpisode(peak_time=base, trough_time=base, recovery_time=base, depth=0.1, duration_days=1.0)],
        max_drawdown=0.1,
        avg_drawdown=0.2,
        ulcer_index=0.3,
        pain_index=0.4,
        tuw_max_days=None,
        tuw_avg_days=None,
        tuw_current_days=None,
    )
    monkeypatch.setattr(AnalysisContext, 'drawdown_stats', lambda _self, _w: stats_nonzero)
    assert mc._burke_ratio(AnalysisContext(SeriesCache()), none_window) is None


def test_metrics_calculators_var_cvar_branches(monkeypatch):
    assert mc._var_metric([], 0.95, 0) is None
    monkeypatch.setattr(mc, 'quantile', lambda *_args, **_kwargs: None)
    assert mc._var_metric([0.1], 0.95, 0) is None

    assert mc._cvar_metric([], 0.95, 0) is None
    monkeypatch.setattr(mc, 'quantile', lambda *_args, **_kwargs: None)
    assert mc._cvar_metric([0.1], 0.95, 0) is None

    monkeypatch.setattr(mc, 'quantile', lambda data, _q: min(data) - 1.0)
    assert mc._cvar_metric([0.1, 0.2], 0.95, 0) is None


def test_metrics_calculators_distribution_branches(monkeypatch):
    assert mc._skewness([1.0, 2.0]) is None
    assert mc._skewness([1.0, 1.0, 1.0]) is None

    original_mean = mc.mean
    monkeypatch.setattr(mc, 'mean', lambda *_args, **_kwargs: None)
    assert mc._skewness([1.0, 2.0, 3.0]) is None

    assert mc._kurtosis([1.0, 2.0, 3.0]) is None
    monkeypatch.setattr(mc, 'mean', lambda *_args, **_kwargs: None)
    assert mc._kurtosis([1.0, 2.0, 3.0, 4.0]) is None

    monkeypatch.setattr(mc, 'mean', original_mean)
    assert mc._kurtosis([1.0, 1.0, 1.0, 1.0]) is None
    assert mc._tail_ratio([], 0.5, 0) is None
    assert mc._tail_ratio([0.0, 1.0], 1.0, 0) is None


def test_metrics_calculators_benchmark_helpers(monkeypatch):
    base = datetime(2024, 1, 1)
    rp = ReturnPoint(time=base + timedelta(days=1), period_return=0.01, days=1.0, daily_return=0.01)
    window = make_window([rp], [0.01], [1.0], 0.01)

    empty_series = BenchmarkSeries(asset='BTC', cumulative_returns=[], return_points=[])
    assert mc._benchmark_window(window, empty_series) is None

    series_with_none = BenchmarkSeries(
        asset='BTC',
        cumulative_returns=[0.0, None],
        return_points=[rp],
    )
    assert mc._benchmark_window(window, series_with_none) is None

    window_no_pairs = make_window(
        [ReturnPoint(time=base + timedelta(days=1), period_return=None, days=None, daily_return=None)],
        [],
        [],
        0.0,
    )
    assert mc._alpha(window_no_pairs, series_with_none, 0, 252) is None

    bench_points = [
        ReturnPoint(time=base + timedelta(days=1), period_return=0.01, days=1.0, daily_return=0.01),
        ReturnPoint(time=base + timedelta(days=2), period_return=0.02, days=1.0, daily_return=0.02),
    ]
    window_points = make_window(bench_points, [0.01, 0.02], [1.0, 1.0], 0.02)
    benchmark = BenchmarkSeries(asset='BTC', cumulative_returns=[0.0, 0.01, 0.03], return_points=bench_points)
    monkeypatch.setattr(mc, 'weighted_mean', lambda *_args, **_kwargs: None)
    assert mc._jensen_alpha(window_points, benchmark, 0.0, 0, 252) is None

    window_empty = make_window(
        [ReturnPoint(time=base + timedelta(days=1), period_return=None, days=None, daily_return=None)],
        [],
        [],
        0.0,
    )
    assert mc._tracking_error(window_empty, benchmark, 0, 252) is None
    assert mc._weighted_beta([], [], []) is None

    rp_none = ReturnPoint(time=base + timedelta(days=1), period_return=None, days=None, daily_return=None)
    empty_window = make_window([rp_none], [], [], 0.0)
    benchmark_none = BenchmarkSeries(asset='BTC', cumulative_returns=[0.0, 0.0], return_points=[rp_none])
    assert mc._jensen_alpha(empty_window, benchmark_none, 0.0, 0, 252) is None

    assert mc._weighted_beta([0.1, 0.2], [0.05, 0.05], [1.0, 1.0]) is None


def test_trade_window_results_appends():
    base = datetime(2024, 1, 1)
    trade_window = TradeWindowData(
        label='win',
        start=base,
        end=base + timedelta(days=1),
        points=[TradePerformancePoint(time=base, pnl=1.0, return_pct=0.1)],
        pnls=[1.0],
        returns=[0.1],
    )
    results = mc._trade_window_results([trade_window], lambda _w: 0.5)
    assert len(results) == 1
    assert results[0].label == 'win'
