from dataclasses import dataclass
from datetime import datetime
from typing import Optional


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


@dataclass(frozen=True, slots=True)
class TradePerformancePoint:
    time: datetime
    pnl: float
    return_pct: Optional[float]


@dataclass(frozen=True, slots=True)
class TradeWindowData:
    label: str
    start: datetime
    end: datetime
    points: list[TradePerformancePoint]
    pnls: list[float]
    returns: list[float]


@dataclass(frozen=True, slots=True)
class TradeSummary:
    total: int
    wins: int
    losses: int
    win_rate: Optional[float]
    avg_win: Optional[float]
    avg_loss: Optional[float]
    expectancy: Optional[float]
    profit_factor: Optional[float]
    payoff_ratio: Optional[float]
    kelly: Optional[float]
    total_profit: Optional[float]
    total_loss: Optional[float]


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


__all__ = [
    'DEFAULT_WINDOWS',
    'MetricSeriesPoint',
    'MetricWindow',
    'MetricResult',
    'SkippedResult',
    'ReturnPoint',
    'TradePerformancePoint',
    'TradeWindowData',
    'TradeSummary',
    'BenchmarkSeries',
    'WindowData',
    'DrawdownEpisode',
    'DrawdownStats',
]
