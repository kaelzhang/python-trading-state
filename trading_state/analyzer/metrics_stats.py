from decimal import Decimal
from math import sqrt
from typing import Iterable, Optional

from trading_state.pnl import PerformanceSnapshot

from .types import Params


def as_float(value: Decimal) -> float:
    return float(value)


def normalize_daily_return(
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


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def weighted_mean(values: list[float], weights: list[float]) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None
    total_weight = sum(weights)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def weighted_variance(
    values: list[float],
    weights: list[float],
    ddof: float = 1.0
) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None
    avg = weighted_mean(values, weights)
    if avg is None:
        return None
    total_weight = sum(weights)
    if total_weight <= ddof:
        return None
    variance = sum(
        weight * (value - avg) ** 2
        for value, weight in zip(values, weights)
    ) / (total_weight - ddof)
    return variance


def weighted_std(values: list[float], weights: list[float]) -> Optional[float]:
    variance = weighted_variance(values, weights)
    return sqrt(variance) if variance is not None else None


def weighted_covariance(
    xs: list[float],
    ys: list[float],
    weights: list[float],
    ddof: float = 1.0
) -> Optional[float]:
    if not xs or not ys or not weights:
        return None
    if len(xs) != len(ys) or len(xs) != len(weights):
        return None
    mean_x = weighted_mean(xs, weights)
    mean_y = weighted_mean(ys, weights)
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


def weighted_correlation(
    xs: list[float],
    ys: list[float],
    weights: list[float]
) -> Optional[float]:
    cov = weighted_covariance(xs, ys, weights)
    if cov is None:
        return None
    std_x = weighted_std(xs, weights)
    std_y = weighted_std(ys, weights)
    if std_x is None or std_y is None or std_x == 0 or std_y == 0:
        return None
    return cov / (std_x * std_y)


def quantile(values: list[float], q: float) -> Optional[float]:
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


def risk_free_daily(risk_free_rate: float, trading_days: int) -> float:
    if trading_days <= 0:
        return 0.0
    return (1.0 + risk_free_rate) ** (1.0 / trading_days) - 1.0


def daily_threshold(value: float, trading_days: int) -> float:
    if trading_days <= 0:
        return value
    return (1.0 + value) ** (1.0 / trading_days) - 1.0


def get_trading_days(params: Optional[Params], default: int = 252) -> int:
    if params is None:
        return default
    value = getattr(params, 'trading_days', default)
    return value if value > 0 else default


def is_order_snapshot(snapshot: PerformanceSnapshot) -> bool:
    labels = snapshot.labels or {}
    if not isinstance(labels, dict):
        return False
    return bool(labels.get('__ORDER__'))
