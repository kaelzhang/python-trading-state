from datetime import datetime, timedelta
from bisect import bisect_right
from math import sqrt
from typing import Optional

from trading_state.pnl import PerformanceSnapshot

from .metrics_models import (
    DEFAULT_WINDOWS,
    MetricSeriesPoint,
    ReturnPoint,
    BenchmarkSeries,
    WindowData,
    TradePerformancePoint,
    TradeWindowData,
    TradeSummary,
    DrawdownEpisode,
    DrawdownStats
)
from .metrics_stats import (
    as_float,
    normalize_daily_return,
    compound_returns,
    mean,
    is_order_snapshot
)


class SeriesCache:
    def __init__(self) -> None:
        self.source_len: int = 0
        self.times: list[datetime] = []
        self.values: list[float] = []
        self.cash_flows: list[float] = []
        self.return_points: list[ReturnPoint] = []
        self.cumulative_returns: list[float] = []
        self.benchmarks: dict[str, BenchmarkSeries] = {}
        self.trade_points: list[TradePerformancePoint] = []
        self._last_order_index: Optional[int] = None

    def rebuild(self, snapshots: list[PerformanceSnapshot]) -> None:
        self.source_len = 0
        self.times = []
        self.values = []
        self.cash_flows = []
        self.return_points = []
        self.cumulative_returns = []
        self.benchmarks = {}
        self.trade_points = []
        self._last_order_index = None
        self.extend(snapshots)

    def extend(self, snapshots: list[PerformanceSnapshot]) -> None:
        for snapshot in snapshots:
            self._append_snapshot(snapshot)
        self.source_len += len(snapshots)

    def _append_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        time = snapshot.time
        value = as_float(snapshot.account_value)
        cash_flow = as_float(snapshot.net_cash_flow)

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
            daily_return = normalize_daily_return(period_return, days)
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
        self._append_trade_snapshot(snapshot, index)

    def _append_benchmarks(self, snapshot: PerformanceSnapshot, index: int) -> None:
        for asset_key, series in self.benchmarks.items():
            bench = snapshot.benchmarks.get(series.asset)
            if bench is None:
                series.cumulative_returns.append(None)
            else:
                series.cumulative_returns.append(as_float(bench.benchmark_return))

        for asset, bench in snapshot.benchmarks.items():
            asset_key = asset.lower()
            if asset_key in self.benchmarks:
                continue
            series = BenchmarkSeries(
                asset=asset,
                cumulative_returns=[None] * index + [as_float(bench.benchmark_return)],
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
            daily_return = normalize_daily_return(period_return, days)
            series.return_points.append(
                ReturnPoint(
                    time=self.times[index],
                    period_return=period_return,
                    days=days if days > 0 else None,
                    daily_return=daily_return
                )
            )

    def _append_trade_snapshot(self, snapshot: PerformanceSnapshot, index: int) -> None:
        is_order = is_order_snapshot(snapshot)
        if not is_order:
            return
        if self._last_order_index is None:
            self._last_order_index = index
            return
        prev_index = self._last_order_index
        prev_value = self.values[prev_index]
        prev_cash = self.cash_flows[prev_index]
        value = self.values[index]
        cash = self.cash_flows[index]
        trade_pnl = (value - prev_value) - (cash - prev_cash)
        trade_return = trade_pnl / prev_value if prev_value > 0 else None
        self.trade_points.append(
            TradePerformancePoint(
                time=self.times[index],
                pnl=trade_pnl,
                return_pct=trade_return
            )
        )
        self._last_order_index = index


class AnalysisContext:
    def __init__(self, cache: SeriesCache) -> None:
        self._cache = cache
        self._window_cache: dict[str, WindowData] = {}
        self._drawdown_cache: dict[str, DrawdownStats] = {}
        self._trade_window_cache: dict[str, TradeWindowData] = {}
        self._trade_summary_cache: dict[str, TradeSummary] = {}

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
    def trade_points(self) -> list[TradePerformancePoint]:
        return self._cache.trade_points

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

    def trade_full_window(self) -> Optional[TradeWindowData]:
        if not self.trade_points:
            return None
        start = self.trade_points[0].time
        end = self.trade_points[-1].time
        return self._build_trade_window('full', start, end)

    def trade_windows(self) -> list[TradeWindowData]:
        if not self.trade_points or self.end_time is None:
            return []
        windows: list[TradeWindowData] = []
        end_time = self.end_time
        for label, days in DEFAULT_WINDOWS:
            start_time = end_time - timedelta(days=days)
            window = self._build_trade_window(label, start_time, end_time)
            if window is None:
                continue
            windows.append(window)
        return windows

    def trade_summary(self, window: TradeWindowData) -> TradeSummary:
        cached = self._trade_summary_cache.get(window.label)
        if cached is not None:
            return cached
        summary = _trade_summary(window)
        self._trade_summary_cache[window.label] = summary
        return summary

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

    def _build_trade_window(
        self,
        label: str,
        start: datetime,
        end: datetime
    ) -> Optional[TradeWindowData]:
        cached = self._trade_window_cache.get(label)
        if cached is not None:
            return cached
        points = [
            point for point in self.trade_points
            if start <= point.time <= end
        ]
        if not points:
            return None
        pnls = [point.pnl for point in points]
        returns = [
            point.return_pct for point in points
            if point.return_pct is not None
        ]
        window = TradeWindowData(
            label=label,
            start=start,
            end=end,
            points=points,
            pnls=pnls,
            returns=returns
        )
        self._trade_window_cache[label] = window
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
    avg_drawdown = mean([ep.depth for ep in episodes]) if episodes else None
    ulcer_index = sqrt(mean([d ** 2 for d in drawdowns])) if drawdowns else None
    pain_index = mean(drawdowns) if drawdowns else None

    durations = [ep.duration_days for ep in episodes if ep.recovery_time is not None]
    tuw_max = max(durations) if durations else None
    tuw_avg = mean(durations) if durations else None
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


def _trade_summary(window: TradeWindowData) -> TradeSummary:
    pnls = window.pnls
    total = len(pnls)
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total if total > 0 else None
    avg_win = mean(wins)
    avg_loss = mean([abs(loss) for loss in losses])
    total_profit = sum(wins) if wins else None
    total_loss = abs(sum(losses)) if losses else None
    profit_factor = None
    if total_loss is not None and total_loss > 0 and total_profit is not None:
        profit_factor = total_profit / total_loss
    payoff_ratio = None
    if avg_win is not None and avg_loss is not None and avg_loss != 0:
        payoff_ratio = avg_win / avg_loss
    expectancy = mean(pnls)
    kelly = None
    if win_rate is not None and payoff_ratio is not None and payoff_ratio != 0:
        kelly = win_rate - (1.0 - win_rate) / payoff_ratio
    return TradeSummary(
        total=total,
        wins=win_count,
        losses=loss_count,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy=expectancy,
        profit_factor=profit_factor,
        payoff_ratio=payoff_ratio,
        kelly=kelly,
        total_profit=total_profit,
        total_loss=total_loss
    )
