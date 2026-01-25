from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from bisect import bisect_right
from math import sqrt
from typing import Callable, Iterable, Optional

from trading_state.pnl import PerformanceSnapshot

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


DEFAULT_WINDOWS: tuple[tuple[str, int], ...] = (
    ('1w', 7),
    ('1m', 30),
    ('3m', 90),
    ('6m', 180),
    ('1y', 365),
)


@dataclass(frozen=True, slots=True)
class MetricSeriesPoint:
    time: datetime
    value: float


@dataclass(frozen=True, slots=True)
class MetricWindow:
    label: str
    start: datetime
    end: datetime
    value: float


@dataclass(frozen=True, slots=True)
class MetricResult:
    value: Optional[float]
    as_of: Optional[datetime]
    windows: list[MetricWindow]
    series: Optional[list[MetricSeriesPoint]] = None
    notes: Optional[str] = None
    extras: Optional[dict[str, float]] = None


@dataclass(frozen=True, slots=True)
class SkippedResult:
    reason: str


@dataclass(frozen=True, slots=True)
class ReturnPoint:
    time: datetime
    period_return: Optional[float]
    days: Optional[float]
    daily_return: Optional[float]


@dataclass(slots=True)
class BenchmarkSeries:
    asset: str
    cumulative_returns: list[Optional[float]]
    return_points: list[ReturnPoint]


@dataclass(frozen=True, slots=True)
class WindowData:
    label: str
    start_index: int
    end_index: int
    start_time: datetime
    end_time: datetime
    times: list[datetime]
    values: list[float]
    return_points: list[ReturnPoint]
    period_returns: list[float]
    daily_returns: list[float]
    daily_weights: list[float]
    cumulative_return: Optional[float]


@dataclass(frozen=True, slots=True)
class DrawdownEpisode:
    peak_time: datetime
    trough_time: datetime
    recovery_time: Optional[datetime]
    depth: float
    duration_days: float


@dataclass(frozen=True, slots=True)
class DrawdownStats:
    series: list[MetricSeriesPoint]
    episodes: list[DrawdownEpisode]
    max_drawdown: Optional[float]
    avg_drawdown: Optional[float]
    ulcer_index: Optional[float]
    pain_index: Optional[float]
    tuw_max_days: Optional[float]
    tuw_avg_days: Optional[float]
    tuw_current_days: Optional[float]


class SeriesCache:
    def __init__(self) -> None:
        self.source_len: int = 0
        self.times: list[datetime] = []
        self.values: list[float] = []
        self.cash_flows: list[float] = []
        self.return_points: list[ReturnPoint] = []
        self.cumulative_returns: list[float] = []
        self.benchmarks: dict[str, BenchmarkSeries] = {}

    def rebuild(self, snapshots: list[PerformanceSnapshot]) -> None:
        self.source_len = 0
        self.times = []
        self.values = []
        self.cash_flows = []
        self.return_points = []
        self.cumulative_returns = []
        self.benchmarks = {}
        self.extend(snapshots)

    def extend(self, snapshots: list[PerformanceSnapshot]) -> None:
        for snapshot in snapshots:
            self._append_snapshot(snapshot)
        self.source_len += len(snapshots)

    def _append_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        time = snapshot.time
        value = _as_float(snapshot.account_value)
        cash_flow = _as_float(snapshot.net_cash_flow)

        index = len(self.times)
        self.times.append(time)
        self.values.append(value)
        self.cash_flows.append(cash_flow)

        if index == 0:
            self.cumulative_returns.append(0.0)
        else:
            prev_value = self.values[index - 1]
            prev_cash = self.cash_flows[index - 1]
            delta_cash = cash_flow - prev_cash
            period_return = None
            days = (time - self.times[index - 1]).total_seconds() / 86400.0
            if prev_value > 0:
                period_return = (value - prev_value - delta_cash) / prev_value
            daily_return = _normalize_daily_return(period_return, days)
            self.return_points.append(
                ReturnPoint(
                    time=time,
                    period_return=period_return,
                    days=days if days > 0 else None,
                    daily_return=daily_return
                )
            )
            prev_cum = self.cumulative_returns[index - 1]
            if period_return is None:
                self.cumulative_returns.append(prev_cum)
            else:
                self.cumulative_returns.append(
                    (1.0 + prev_cum) * (1.0 + period_return) - 1.0
                )

        self._append_benchmarks(snapshot, index)

    def _append_benchmarks(self, snapshot: PerformanceSnapshot, index: int) -> None:
        for asset_key, series in self.benchmarks.items():
            bench = snapshot.benchmarks.get(series.asset)
            if bench is None:
                series.cumulative_returns.append(None)
            else:
                series.cumulative_returns.append(_as_float(bench.benchmark_return))

        for asset, bench in snapshot.benchmarks.items():
            asset_key = asset.lower()
            if asset_key in self.benchmarks:
                continue
            series = BenchmarkSeries(
                asset=asset,
                cumulative_returns=[None] * index + [_as_float(bench.benchmark_return)],
                return_points=[
                    ReturnPoint(
                        time=self.times[i + 1],
                        period_return=None,
                        days=None,
                        daily_return=None
                    )
                    for i in range(index)
                ]
            )
            self.benchmarks[asset_key] = series

        if index == 0:
            return

        for series in self.benchmarks.values():
            prev_cum = series.cumulative_returns[index - 1]
            curr_cum = series.cumulative_returns[index]
            period_return = None
            if prev_cum is not None and curr_cum is not None and (1.0 + prev_cum) > 0:
                period_return = (1.0 + curr_cum) / (1.0 + prev_cum) - 1.0
            days = (self.times[index] - self.times[index - 1]).total_seconds() / 86400.0
            daily_return = _normalize_daily_return(period_return, days)
            series.return_points.append(
                ReturnPoint(
                    time=self.times[index],
                    period_return=period_return,
                    days=days if days > 0 else None,
                    daily_return=daily_return
                )
            )


class AnalysisContext:
    def __init__(self, cache: SeriesCache) -> None:
        self._cache = cache
        self._window_cache: dict[str, WindowData] = {}
        self._drawdown_cache: dict[str, DrawdownStats] = {}

    @property
    def times(self) -> list[datetime]:
        return self._cache.times

    @property
    def values(self) -> list[float]:
        return self._cache.values

    @property
    def return_points(self) -> list[ReturnPoint]:
        return self._cache.return_points

    @property
    def cumulative_returns(self) -> list[float]:
        return self._cache.cumulative_returns

    @property
    def benchmarks(self) -> dict[str, BenchmarkSeries]:
        return self._cache.benchmarks

    @property
    def end_time(self) -> Optional[datetime]:
        return self.times[-1] if self.times else None

    @property
    def start_time(self) -> Optional[datetime]:
        return self.times[0] if self.times else None

    def full_window(self) -> Optional[WindowData]:
        if not self.times:
            return None
        return self._build_window('full', 0, len(self.times) - 1)

    def windows(self) -> list[WindowData]:
        if not self.times:
            return []
        windows: list[WindowData] = []
        end_time = self.times[-1]
        for label, days in DEFAULT_WINDOWS:
            target_start = end_time - timedelta(days=days)
            start_index = bisect_right(self.times, target_start) - 1
            if start_index < 0:
                continue
            if start_index >= len(self.times) - 1:
                continue
            window = self._build_window(label, start_index, len(self.times) - 1)
            windows.append(window)
        return windows

    def window_by_label(self, label: str) -> Optional[WindowData]:
        if label == 'full':
            return self.full_window()
        for window in self.windows():
            if window.label == label:
                return window
        return None

    def drawdown_stats(self, window: WindowData) -> DrawdownStats:
        cached = self._drawdown_cache.get(window.label)
        if cached is not None:
            return cached
        stats = compute_drawdown_stats(window.times, window.values)
        self._drawdown_cache[window.label] = stats
        return stats

    def _build_window(self, label: str, start_index: int, end_index: int) -> WindowData:
        cached = self._window_cache.get(label)
        if cached is not None:
            return cached
        times = self.times[start_index:end_index + 1]
        values = self.values[start_index:end_index + 1]
        return_points = self.return_points[start_index:end_index]
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
        window = WindowData(
            label=label,
            start_index=start_index,
            end_index=end_index,
            start_time=times[0],
            end_time=times[-1],
            times=times,
            values=values,
            return_points=return_points,
            period_returns=period_returns,
            daily_returns=daily_returns,
            daily_weights=daily_weights,
            cumulative_return=cumulative_return
        )
        self._window_cache[label] = window
        return window


def compute_drawdown_stats(
    times: list[datetime],
    values: list[float]
) -> DrawdownStats:
    series: list[MetricSeriesPoint] = []
    episodes: list[DrawdownEpisode] = []
    if not times or not values:
        return DrawdownStats(series, episodes, None, None, None, None, None, None, None)

    peak_value = values[0]
    peak_time = times[0]
    in_drawdown = False
    trough_value = values[0]
    trough_time = times[0]

    for time, value in zip(times, values):
        drawdown = 0.0
        if peak_value > 0:
            drawdown = max(0.0, (peak_value - value) / peak_value)
        series.append(MetricSeriesPoint(time=time, value=drawdown))

        if value >= peak_value:
            if in_drawdown:
                duration = (time - peak_time).total_seconds() / 86400.0
                depth = max(0.0, (peak_value - trough_value) / peak_value)
                episodes.append(
                    DrawdownEpisode(
                        peak_time=peak_time,
                        trough_time=trough_time,
                        recovery_time=time,
                        depth=depth,
                        duration_days=duration
                    )
                )
                in_drawdown = False
            peak_value = value
            peak_time = time
            trough_value = value
            trough_time = time
            continue

        if not in_drawdown:
            in_drawdown = True
            trough_value = value
            trough_time = time
        elif value < trough_value:
            trough_value = value
            trough_time = time

    if in_drawdown:
        duration = (times[-1] - peak_time).total_seconds() / 86400.0
        depth = max(0.0, (peak_value - trough_value) / peak_value)
        episodes.append(
            DrawdownEpisode(
                peak_time=peak_time,
                trough_time=trough_time,
                recovery_time=None,
                depth=depth,
                duration_days=duration
            )
        )

    drawdowns = [point.value for point in series if point.value is not None]
    max_drawdown = max(drawdowns) if drawdowns else None
    avg_drawdown = _mean([ep.depth for ep in episodes]) if episodes else None
    ulcer_index = sqrt(_mean([d ** 2 for d in drawdowns])) if drawdowns else None
    pain_index = _mean(drawdowns) if drawdowns else None

    durations = [ep.duration_days for ep in episodes if ep.recovery_time is not None]
    tuw_max = max(durations) if durations else None
    tuw_avg = _mean(durations) if durations else None
    current_tuw = None
    if episodes and episodes[-1].recovery_time is None:
        current_tuw = episodes[-1].duration_days
        if tuw_max is None:
            tuw_max = current_tuw

    return DrawdownStats(
        series=series,
        episodes=episodes,
        max_drawdown=max_drawdown,
        avg_drawdown=avg_drawdown,
        ulcer_index=ulcer_index,
        pain_index=pain_index,
        tuw_max_days=tuw_max,
        tuw_avg_days=tuw_avg,
        tuw_current_days=current_tuw
    )


def _as_float(value: Decimal) -> float:
    return float(value)


def _normalize_daily_return(
    period_return: Optional[float],
    days: float
) -> Optional[float]:
    if period_return is None:
        return None
    if days <= 0:
        return None
    base = 1.0 + period_return
    if base <= 0:
        return period_return / days
    return base ** (1.0 / days) - 1.0


def compound_returns(returns: Iterable[float]) -> Optional[float]:
    total = 1.0
    has_value = False
    for value in returns:
        has_value = True
        total *= (1.0 + value)
    return total - 1.0 if has_value else None


def annualize_from_daily(mean_daily: Optional[float], trading_days: int) -> Optional[float]:
    if mean_daily is None:
        return None
    return mean_daily * trading_days


def annualize_cagr(total_return: Optional[float], days: float, trading_days: int) -> Optional[float]:
    if total_return is None or days <= 0:
        return None
    base = 1.0 + total_return
    if base <= 0:
        return None
    return base ** (trading_days / days) - 1.0


def _mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def _weighted_mean(values: list[float], weights: list[float]) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None
    total_weight = sum(weights)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def _weighted_variance(
    values: list[float],
    weights: list[float],
    ddof: float = 1.0
) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None
    mean = _weighted_mean(values, weights)
    if mean is None:
        return None
    total_weight = sum(weights)
    if total_weight <= ddof:
        return None
    variance = sum(
        weight * (value - mean) ** 2
        for value, weight in zip(values, weights)
    ) / (total_weight - ddof)
    return variance


def _weighted_std(values: list[float], weights: list[float]) -> Optional[float]:
    variance = _weighted_variance(values, weights)
    return sqrt(variance) if variance is not None else None


def _weighted_covariance(
    xs: list[float],
    ys: list[float],
    weights: list[float],
    ddof: float = 1.0
) -> Optional[float]:
    if not xs or not ys or not weights:
        return None
    if len(xs) != len(ys) or len(xs) != len(weights):
        return None
    mean_x = _weighted_mean(xs, weights)
    mean_y = _weighted_mean(ys, weights)
    if mean_x is None or mean_y is None:
        return None
    total_weight = sum(weights)
    if total_weight <= ddof:
        return None
    cov = sum(
        weight * (x - mean_x) * (y - mean_y)
        for x, y, weight in zip(xs, ys, weights)
    ) / (total_weight - ddof)
    return cov


def _weighted_correlation(
    xs: list[float],
    ys: list[float],
    weights: list[float]
) -> Optional[float]:
    cov = _weighted_covariance(xs, ys, weights)
    if cov is None:
        return None
    std_x = _weighted_std(xs, weights)
    std_y = _weighted_std(ys, weights)
    if std_x is None or std_y is None or std_x == 0 or std_y == 0:
        return None
    return cov / (std_x * std_y)


def _quantile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _risk_free_daily(risk_free_rate: float, trading_days: int) -> float:
    if trading_days <= 0:
        return 0.0
    return (1.0 + risk_free_rate) ** (1.0 / trading_days) - 1.0


def _daily_threshold(value: float, trading_days: int) -> float:
    if trading_days <= 0:
        return value
    return (1.0 + value) ** (1.0 / trading_days) - 1.0


def _get_trading_days(params: Optional[Params], default: int = 252) -> int:
    if params is None:
        return default
    value = getattr(params, 'trading_days', default)
    return value if value > 0 else default


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


def _total_return_series(context: AnalysisContext) -> list[MetricSeriesPoint]:
    points: list[MetricSeriesPoint] = []
    for time, value in zip(context.times, context.cumulative_returns):
        points.append(MetricSeriesPoint(time=time, value=value))
    return points


def calc_total_return(context: AnalysisContext, params: ParamsAnnualizedReturn) -> MetricResult:
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
    trading_days = _get_trading_days(params)
    mean_daily = _weighted_mean(full.daily_returns, full.daily_weights)
    value = annualize_from_daily(mean_daily, trading_days)
    windows = _window_results(
        context.windows(),
        lambda w: annualize_from_daily(
            _weighted_mean(w.daily_returns, w.daily_weights),
            trading_days
        )
    )
    return MetricResult(value, full.end_time, windows)


def calc_cagr(context: AnalysisContext, params: ParamsAnnualizedReturn) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    trading_days = _get_trading_days(params)
    days = (full.end_time - full.start_time).total_seconds() / 86400.0
    value = annualize_cagr(full.cumulative_return, days, trading_days)
    windows = _window_results(
        context.windows(),
        lambda w: annualize_cagr(
            w.cumulative_return,
            (w.end_time - w.start_time).total_seconds() / 86400.0,
            trading_days
        )
    )
    return MetricResult(value, full.end_time, windows)


def calc_mean_return(context: AnalysisContext, params: ParamsAnnualizedReturn) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _weighted_mean(full.daily_returns, full.daily_weights)
    windows = _window_results(
        context.windows(),
        lambda w: _weighted_mean(w.daily_returns, w.daily_weights)
    )
    return MetricResult(value, full.end_time, windows, notes='daily')


def calc_geometric_mean_return(context: AnalysisContext, params: ParamsAnnualizedReturn) -> MetricResult:
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
    std = _weighted_std(full.daily_returns, full.daily_weights)
    trading_days = _get_trading_days(params)
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
    trading_days = _get_trading_days(params)
    threshold = _downside_threshold(params.minimum_acceptable_return, params.downside_threshold, trading_days)
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
    trading_days = _get_trading_days(params)
    threshold = _downside_threshold(params.minimum_acceptable_return, params.downside_threshold, trading_days)
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
    trading_days = _get_trading_days(params)
    rf_daily = _risk_free_daily(params.risk_free_rate, trading_days)
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
    trading_days = _get_trading_days(params)
    threshold = _downside_threshold(params.minimum_acceptable_return, params.downside_threshold, trading_days)
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
    rf_daily = _risk_free_daily(params.risk_free_rate, 252)
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
    rf_daily = _risk_free_daily(params.risk_free_rate, params.trading_days)
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


def calc_mar_ratio(context: AnalysisContext, params: ParamsAnnualizedReturn) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _mar_ratio(context, full)
    windows = _window_results(
        context.windows(),
        lambda w: _mar_ratio(context, w)
    )
    return MetricResult(value, full.end_time, windows)


def calc_upi_martin_ratio(context: AnalysisContext, params: ParamsAnnualizedReturn) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _upi_martin_ratio(context, full)
    windows = _window_results(
        context.windows(),
        lambda w: _upi_martin_ratio(context, w)
    )
    return MetricResult(value, full.end_time, windows)


def calc_sterling_ratio(context: AnalysisContext, params: ParamsAnnualizedReturn) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _sterling_ratio(context, full)
    windows = _window_results(
        context.windows(),
        lambda w: _sterling_ratio(context, w)
    )
    return MetricResult(value, full.end_time, windows)


def calc_burke_ratio(context: AnalysisContext, params: ParamsAnnualizedReturn) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _burke_ratio(context, full)
    windows = _window_results(
        context.windows(),
        lambda w: _burke_ratio(context, w)
    )
    return MetricResult(value, full.end_time, windows)


def calc_pain_ratio(context: AnalysisContext, params: ParamsAnnualizedReturn) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _pain_ratio(context, full)
    windows = _window_results(
        context.windows(),
        lambda w: _pain_ratio(context, w)
    )
    return MetricResult(value, full.end_time, windows)


def calc_mdd(context: AnalysisContext, params: ParamsAnnualizedReturn) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    stats = context.drawdown_stats(full)
    windows = _window_results(
        context.windows(),
        lambda w: context.drawdown_stats(w).max_drawdown
    )
    return MetricResult(stats.max_drawdown, full.end_time, windows, series=stats.series)


def calc_average_drawdown(context: AnalysisContext, params: ParamsAnnualizedReturn) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    stats = context.drawdown_stats(full)
    windows = _window_results(
        context.windows(),
        lambda w: context.drawdown_stats(w).avg_drawdown
    )
    return MetricResult(stats.avg_drawdown, full.end_time, windows)


def calc_tuw(context: AnalysisContext, params: ParamsAnnualizedReturn) -> MetricResult:
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


def calc_ulcer_index(context: AnalysisContext, params: ParamsAnnualizedReturn) -> MetricResult:
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


def calc_skewness(context: AnalysisContext, params: ParamsAnnualizedReturn) -> MetricResult:
    full = context.full_window()
    if full is None:
        return MetricResult(None, None, [])
    value = _skewness(full.daily_returns)
    return MetricResult(value, full.end_time, [])


def calc_kurtosis(context: AnalysisContext, params: ParamsAnnualizedReturn) -> MetricResult:
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
    rf_daily = _risk_free_daily(params.risk_free_rate, 252)
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
    std = _weighted_std(window.daily_returns, window.daily_weights)
    if std is None:
        return None
    return std * sqrt(trading_days)


def _downside_threshold(
    minimum_acceptable_return: float,
    downside_threshold: float,
    trading_days: int
) -> float:
    threshold = downside_threshold if downside_threshold != 0.0 else minimum_acceptable_return
    return _daily_threshold(threshold, trading_days)


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
    variance = _weighted_variance(filtered, weights)
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
    variance = _weighted_variance(filtered, weights, ddof=0.0)
    if variance is None:
        return None
    return variance * trading_days


def _sharpe_ratio(
    daily_returns: list[float],
    daily_weights: list[float],
    rf_daily: float,
    trading_days: int
) -> Optional[float]:
    mean_daily = _weighted_mean(daily_returns, daily_weights)
    if mean_daily is None:
        return None
    std = _weighted_std(daily_returns, daily_weights)
    if std is None or std == 0:
        return None
    return (mean_daily - rf_daily) / std * sqrt(trading_days)


def _sortino_ratio(
    daily_returns: list[float],
    daily_weights: list[float],
    threshold: float,
    trading_days: int
) -> Optional[float]:
    mean_daily = _weighted_mean(daily_returns, daily_weights)
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
    mean_daily = _weighted_mean(window.daily_returns, window.daily_weights)
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
    mean_active = _weighted_mean(active, weights)
    std_active = _weighted_std(active, weights)
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
    cagr = annualize_cagr(window.cumulative_return, days, 252)
    if cagr is None:
        return None
    return (cagr - risk_free_rate) / stats.max_drawdown


def _mar_ratio(context: AnalysisContext, window: WindowData) -> Optional[float]:
    stats = context.drawdown_stats(window)
    if stats.max_drawdown is None or stats.max_drawdown == 0:
        return None
    days = (window.end_time - window.start_time).total_seconds() / 86400.0
    cagr = annualize_cagr(window.cumulative_return, days, 252)
    if cagr is None:
        return None
    return cagr / stats.max_drawdown


def _upi_martin_ratio(context: AnalysisContext, window: WindowData) -> Optional[float]:
    stats = context.drawdown_stats(window)
    if stats.ulcer_index is None or stats.ulcer_index == 0:
        return None
    days = (window.end_time - window.start_time).total_seconds() / 86400.0
    cagr = annualize_cagr(window.cumulative_return, days, 252)
    if cagr is None:
        return None
    return cagr / stats.ulcer_index


def _sterling_ratio(context: AnalysisContext, window: WindowData) -> Optional[float]:
    stats = context.drawdown_stats(window)
    if stats.avg_drawdown is None or stats.avg_drawdown == 0:
        return None
    days = (window.end_time - window.start_time).total_seconds() / 86400.0
    cagr = annualize_cagr(window.cumulative_return, days, 252)
    if cagr is None:
        return None
    return cagr / stats.avg_drawdown


def _burke_ratio(context: AnalysisContext, window: WindowData) -> Optional[float]:
    stats = context.drawdown_stats(window)
    if not stats.episodes:
        return None
    squared = [ep.depth ** 2 for ep in stats.episodes]
    denom = sqrt(_mean(squared)) if squared else None
    if denom is None or denom == 0:
        return None
    days = (window.end_time - window.start_time).total_seconds() / 86400.0
    cagr = annualize_cagr(window.cumulative_return, days, 252)
    if cagr is None:
        return None
    return cagr / denom


def _pain_ratio(context: AnalysisContext, window: WindowData) -> Optional[float]:
    stats = context.drawdown_stats(window)
    if stats.pain_index is None or stats.pain_index == 0:
        return None
    days = (window.end_time - window.start_time).total_seconds() / 86400.0
    cagr = annualize_cagr(window.cumulative_return, days, 252)
    if cagr is None:
        return None
    return cagr / stats.pain_index


def _var_metric(returns: list[float], confidence_level: float, window: int) -> Optional[float]:
    data = returns[-window:] if window > 0 else returns
    if not data:
        return None
    cutoff = _quantile(data, 1.0 - confidence_level)
    if cutoff is None:
        return None
    return -cutoff


def _cvar_metric(returns: list[float], confidence_level: float, window: int) -> Optional[float]:
    data = returns[-window:] if window > 0 else returns
    if not data:
        return None
    cutoff = _quantile(data, 1.0 - confidence_level)
    if cutoff is None:
        return None
    tail = [r for r in data if r <= cutoff]
    if not tail:
        return None
    return -_mean(tail)


def _skewness(returns: list[float]) -> Optional[float]:
    if len(returns) < 3:
        return None
    mean = _mean(returns)
    if mean is None:
        return None
    diffs = [r - mean for r in returns]
    m2 = _mean([d ** 2 for d in diffs])
    m3 = _mean([d ** 3 for d in diffs])
    if m2 is None or m3 is None or m2 == 0:
        return None
    return m3 / (m2 ** 1.5)


def _kurtosis(returns: list[float]) -> Optional[float]:
    if len(returns) < 4:
        return None
    mean = _mean(returns)
    if mean is None:
        return None
    diffs = [r - mean for r in returns]
    m2 = _mean([d ** 2 for d in diffs])
    m4 = _mean([d ** 4 for d in diffs])
    if m2 is None or m4 is None or m2 == 0:
        return None
    return m4 / (m2 ** 2) - 3.0


def _tail_ratio(returns: list[float], quantile: float, window: int) -> Optional[float]:
    data = returns[-window:] if window > 0 else returns
    if not data:
        return None
    upper = _quantile(data, quantile)
    lower = _quantile(data, 1.0 - quantile)
    if upper is None or lower is None or lower == 0:
        return None
    return upper / abs(lower)


def _alpha(window: WindowData, benchmark: BenchmarkSeries, lookback: int) -> Optional[float]:
    port, bench, weights = _paired_daily_returns(window, benchmark)
    if lookback > 0 and len(port) > lookback:
        port = port[-lookback:]
        bench = bench[-lookback:]
        weights = weights[-lookback:]
    mean_port = _weighted_mean(port, weights)
    mean_bench = _weighted_mean(bench, weights)
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
    mean_port = _weighted_mean(port, weights)
    mean_bench = _weighted_mean(bench, weights)
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
    return _weighted_correlation(port, bench, weights)


def _tracking_error(window: WindowData, benchmark: BenchmarkSeries, lookback: int) -> Optional[float]:
    port, bench, weights = _paired_daily_returns(window, benchmark)
    active = [p - b for p, b in zip(port, bench)]
    if lookback > 0 and len(active) > lookback:
        active = active[-lookback:]
        weights = weights[-lookback:]
    std = _weighted_std(active, weights)
    if std is None:
        return None
    return std * sqrt(252)


def _weighted_beta(xs: list[float], ys: list[float], weights: list[float]) -> Optional[float]:
    cov = _weighted_covariance(xs, ys, weights)
    if cov is None:
        return None
    var = _weighted_variance(ys, weights)
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
    AnalyzerType.TRANSACTION_COST_DRAG: 'requires fee/slippage and execution data',
    AnalyzerType.TURNOVER: 'requires trade notional to separate price moves from trades',
    AnalyzerType.SLIPPAGE_SENSITIVITY: 'requires slippage scenarios or execution data',
    AnalyzerType.WIN_RATE: 'requires trade-level PnL data',
    AnalyzerType.PAYOFF_RATIO: 'requires trade-level PnL data',
    AnalyzerType.EXPECTANCY: 'requires trade-level PnL data',
    AnalyzerType.PROFIT_FACTOR: 'requires trade-level PnL data',
    AnalyzerType.KELLY_CRITERION: 'requires trade-level PnL data',
}


METRIC_CALCULATORS: dict[
    AnalyzerType,
    Callable[
        [AnalysisContext, Params],
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
}
