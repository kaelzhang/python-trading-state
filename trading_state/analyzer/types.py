from enum import Enum
from dataclasses import dataclass
from typing import (
    Type,
    Tuple,
    Self,
    Protocol,
    TypeVar
)

from trading_state.enums import StringEnum
from trading_state.common import FactoryDict


class DataclassProtocol(Protocol):
    __dataclass_fields__: dict

Params = TypeVar('Params', bound=DataclassProtocol)


class AnalyzerCategory(StringEnum):
    RETURN_AND_GROWTH = 'Return & Growth'
    RISK_ADJUSTED_PERF_RATIOS = 'Risk-Adjusted Performance Ratios'
    VOLATILITY_AND_DOWNSIDE_RISK = 'Volatility & Downside Risk'
    DRAWDOWN_AND_PATH_RISK = 'Drawdown & Path Risk'
    TAIL_RISK_AND_DISTRIBUTION = 'Tail Risk & Distribution Shape'
    BENCHMARK_RELATIVE_AND_ATTRIBUTION = 'Benchmark-Relative & Attribution'
    IMPLEMENTATION_COSTS_AND_CAPACITY = 'Implementation, Costs & Capacity'
    TRADE_LEVEL_STATISTICS = 'Trade-Level Statistics'


@dataclass(frozen=True, slots=True)
class AnalyzerTypeInfo:
    name: str
    description: str
    category: AnalyzerCategory
    params: Type[Params] | None = None


@dataclass(frozen=True, slots=True)
class ParamsAnnualizedReturn:
    trading_days: int = 252


@dataclass(frozen=True, slots=True)
class ParamsSharpeRatio:
    risk_free_rate: float = 0.0
    # Which calculates the annualization factor via sqrt(trading_days)
    trading_days: int = 252


@dataclass(frozen=True, slots=True)
class ParamsSortinoRatio:
    minimum_acceptable_return: float = 0.0
    downside_threshold: float = 0.0
    # Which calculates the annualization factor via sqrt(trading_days)
    trading_days: int = 252


@dataclass(frozen=True, slots=True)
class ParamsTreynorRatio:
    # Which is the name of an asset,
    # i.e the key name of the `PerformanceSnapshot.benchmarks` dict
    benchmark: str = 'btc'
    risk_free_rate: float = 0.0


@dataclass(frozen=True, slots=True)
class ParamsInformationRatio:
    # Same as `ParamsTreynorRatio.benchmark`
    benchmark: str = 'btc'
    tracking_error_window: int = 252


@dataclass(frozen=True, slots=True)
class ParamsM2:
    # Same as `ParamsTreynorRatio.benchmark`
    benchmark: str = 'btc'
    risk_free_rate: float = 0.0
    # Same as `ParamsAnnualizedReturn.trading_days`
    trading_days: int = 252


@dataclass(frozen=True, slots=True)
class ParamsCalmarRatio:
    risk_free_rate: float = 0.0


@dataclass(frozen=True, slots=True)
class ParamsDownsideDeviation:
    minimum_acceptable_return: float = 0.0
    downside_threshold: float = 0.0
    trading_days: int = 252


@dataclass(frozen=True, slots=True)
class ParamsSemiVariance:
    minimum_acceptable_return: float = 0.0
    downside_threshold: float = 0.0
    trading_days: int = 252


@dataclass(frozen=True, slots=True)
class ParamsVaR:
    confidence_level: float = 0.95
    window: int = 252


@dataclass(frozen=True, slots=True)
class ParamsTailRatio:
    quantile: float = 0.95
    window: int = 252


@dataclass(frozen=True, slots=True)
class ParamsBenchmarkRelative:
    # Same as `ParamsTreynorRatio.benchmark`
    benchmark: str = 'btc'
    risk_free_rate: float = 0.0
    window: int = 252


class AnalyzerType(Enum):
    TOTAL_RETURN = AnalyzerTypeInfo(
        name='Total Return',
        description='Measures the overall change in equity from start to end',
        category=AnalyzerCategory.RETURN_AND_GROWTH
    )

    ANNUALIZED_RETURN = AnalyzerTypeInfo(
        name='Annualized Return',
        description='Annualizes period returns (not necessarily compounded)',
        category=AnalyzerCategory.RETURN_AND_GROWTH,
        params=ParamsAnnualizedReturn
    )

    CAGR = AnalyzerTypeInfo(
        name='CAGR',
        description='Compounded Annual Growth Rate of the equity curve',
        category=AnalyzerCategory.RETURN_AND_GROWTH
    )

    MEAN_RETURN = AnalyzerTypeInfo(
        name='Mean Return',
        description='Arithmetic average of periodic returns as an expected-return',
        category=AnalyzerCategory.RETURN_AND_GROWTH
    )

    GEOMETRIC_MEAN_RETURN = AnalyzerTypeInfo(
        name='Geometric Mean Return',
        description='Average compounded growth per period (captures volatility drag)',
        category=AnalyzerCategory.RETURN_AND_GROWTH
    )

    SHARPE_RATIO = AnalyzerTypeInfo(
        name='Sharpe Ratio',
        description='Ratio of the expected excess return per unit of total volatility',
        category=AnalyzerCategory.RISK_ADJUSTED_PERF_RATIOS,
        params=ParamsSharpeRatio
    )

    SORTINO_RATIO = AnalyzerTypeInfo(
        name='Sortino Ratio',
        description='Excess return per unit of downside deviation (penalizes only downside risk)',
        category=AnalyzerCategory.RISK_ADJUSTED_PERF_RATIOS,
        params=ParamsSortinoRatio
    )

    TREYNOR_RATIO = AnalyzerTypeInfo(
        name='Treynor Ratio',
        description='Excess return per unit of systematic risk (beta)',
        category=AnalyzerCategory.RISK_ADJUSTED_PERF_RATIOS,
        params=ParamsTreynorRatio
    )

    IR = AnalyzerTypeInfo(
        name='Information Ratio',
        description='Active return per unit of tracking error (active risk)',
        category=AnalyzerCategory.RISK_ADJUSTED_PERF_RATIOS,
        params=ParamsInformationRatio
    )

    M2 = AnalyzerTypeInfo(
        name='Modigliani-Modigliani (M²)',
        description='Risk-adjusted return scaled to benchmark volatility',
        category=AnalyzerCategory.RISK_ADJUSTED_PERF_RATIOS,
        params=ParamsM2
    )

    CALMAR_RATIO = AnalyzerTypeInfo(
        name='Calmar Ratio',
        description='Risk-adjusted return using maximum drawdown as the risk denominator (often CAGR/MaxDD)',
        category=AnalyzerCategory.RISK_ADJUSTED_PERF_RATIOS,
        params=ParamsCalmarRatio
    )

    MAR = AnalyzerTypeInfo(
        name='MAR Ratio',
        description='CAGR divided by maximum drawdown (definition may vary by convention)',
        category=AnalyzerCategory.RISK_ADJUSTED_PERF_RATIOS
    )

    UPI_MARTIN_RATIO = AnalyzerTypeInfo(
        name='UPI Martin Ratio',
        description='Uses Ulcer Index (drawdown depth & duration) as the risk measure for risk-adjusted return',
        category=AnalyzerCategory.RISK_ADJUSTED_PERF_RATIOS
    )

    STERLING_RATIO = AnalyzerTypeInfo(
        name='Sterling Ratio',
        description='Drawdown-based ratio using (average/threshold) drawdowns as the risk measure',
        category=AnalyzerCategory.RISK_ADJUSTED_PERF_RATIOS
    )

    BURKE_RATIO = AnalyzerTypeInfo(
        name='Burke Ratio',
        description='Drawdown-based ratio that aggregates multiple drawdowns (often via squared drawdowns)',
        category=AnalyzerCategory.RISK_ADJUSTED_PERF_RATIOS
    )

    PAIN_RATIO = AnalyzerTypeInfo(
        name='Pain Ratio',
        description='Return divided by a drawdown “pain” measure (e.g., average drawdown / drawdown area)',
        category=AnalyzerCategory.RISK_ADJUSTED_PERF_RATIOS
    )

    VOLATILITY = AnalyzerTypeInfo(
        name='Volatility (Annualized Std Dev)',
        description='Annualized standard deviation of returns (total variability)',
        category=AnalyzerCategory.VOLATILITY_AND_DOWNSIDE_RISK,
        params=ParamsAnnualizedReturn
    )

    DOWNSIDE_DEVIATION = AnalyzerTypeInfo(
        name='Downside Deviation',
        description='Volatility computed only for returns below a threshold (e.g., MAR)',
        category=AnalyzerCategory.VOLATILITY_AND_DOWNSIDE_RISK,
        params=ParamsDownsideDeviation
    )

    SEMI_VARIANCE = AnalyzerTypeInfo(
        name='Semi-variance',
        description='Variance of downside deviations only',
        category=AnalyzerCategory.VOLATILITY_AND_DOWNSIDE_RISK,
        params=ParamsSemiVariance
    )

    MDD = AnalyzerTypeInfo(
        name='Maximum Drawdown',
        description='Largest peak-to-trough decline before a new peak',
        category=AnalyzerCategory.DRAWDOWN_AND_PATH_RISK
    )

    AVERAGE_DRAWDOWN = AnalyzerTypeInfo(
        name='Average Drawdown',
        description='Average depth of drawdown episodes (as defined)',
        category=AnalyzerCategory.DRAWDOWN_AND_PATH_RISK
    )

    TUW = AnalyzerTypeInfo(
        name='Time Under Water',
        description='Time from a peak to recovery above that peak',
        category=AnalyzerCategory.DRAWDOWN_AND_PATH_RISK
    )

    UI = AnalyzerTypeInfo(
        name='Ulcer Index',
        description='Downside risk metric focusing on drawdown depth and duration',
        category=AnalyzerCategory.DRAWDOWN_AND_PATH_RISK
    )

    VAR = AnalyzerTypeInfo(
        name='Value at Risk',
        description='Loss threshold not expected to be exceeded over a horizon at a given confidence level',
        category=AnalyzerCategory.TAIL_RISK_AND_DISTRIBUTION,
        params=ParamsVaR
    )

    CVAR = AnalyzerTypeInfo(
        name='Conditional Value at Risk',
        description='Expected loss in the tail beyond the VaR cutoff',
        category=AnalyzerCategory.TAIL_RISK_AND_DISTRIBUTION,
        params=ParamsVaR
    )

    SKEWNESS = AnalyzerTypeInfo(
        name='Skewness',
        description='Asymmetry of return distribution (negative skew implies crash-like losses)',
        category=AnalyzerCategory.TAIL_RISK_AND_DISTRIBUTION
    )

    KURTOSIS = AnalyzerTypeInfo(
        name='Kurtosis',
        description='Tail heaviness of returns (frequency of extreme outcomes)',
        category=AnalyzerCategory.TAIL_RISK_AND_DISTRIBUTION
    )

    TAIL_RATIO = AnalyzerTypeInfo(
        name='Tail Ratio',
        description='Compares upside-tail magnitude to downside-tail magnitude via quantiles',
        category=AnalyzerCategory.TAIL_RISK_AND_DISTRIBUTION,
        params=ParamsTailRatio
    )

    ALPHA = AnalyzerTypeInfo(
        name='Alpha',
        description='Excess performance over a benchmark or model-implied return',
        category=AnalyzerCategory.BENCHMARK_RELATIVE_AND_ATTRIBUTION,
        params=ParamsBenchmarkRelative
    )

    JENSEN_ALPHA = AnalyzerTypeInfo(
        name='Jensen\'s Alpha',
        description='Abnormal return above the CAPM/model-predicted return',
        category=AnalyzerCategory.BENCHMARK_RELATIVE_AND_ATTRIBUTION,
        params=ParamsBenchmarkRelative
    )

    BETA = AnalyzerTypeInfo(
        name='Beta',
        description='Sensitivity of portfolio returns to the benchmark (systematic risk exposure)',
        category=AnalyzerCategory.BENCHMARK_RELATIVE_AND_ATTRIBUTION,
        params=ParamsBenchmarkRelative
    )

    CORRELATION = AnalyzerTypeInfo(
        name='Correlation',
        description='Co-movement measure between two return series',
        category=AnalyzerCategory.BENCHMARK_RELATIVE_AND_ATTRIBUTION,
        params=ParamsBenchmarkRelative
    )

    TE = AnalyzerTypeInfo(
        name='Tracking Error',
        description='Std dev of active returns (portfolio minus benchmark)',
        category=AnalyzerCategory.BENCHMARK_RELATIVE_AND_ATTRIBUTION,
        params=ParamsBenchmarkRelative
    )

    TRANSACTION_COST_DRAG = AnalyzerTypeInfo(
        name='Transaction Cost Drag',
        description='Return erosion due to fees, slippage, and impact (should be included in net performance)',
        category=AnalyzerCategory.IMPLEMENTATION_COSTS_AND_CAPACITY
    )

    TURNOVER = AnalyzerTypeInfo(
        name='Turnover',
        description='Trading intensity that drives costs and capacity constraints',
        category=AnalyzerCategory.IMPLEMENTATION_COSTS_AND_CAPACITY
    )

    SLIPPAGE_SENSITIVITY = AnalyzerTypeInfo(
        name='Slippage Sensitivity',
        description='How performance changes under different slippage assumptions',
        category=AnalyzerCategory.IMPLEMENTATION_COSTS_AND_CAPACITY
    )

    WIN_RATE = AnalyzerTypeInfo(
        name='Win Rate',
        description='Fraction of trades that are profitable',
        category=AnalyzerCategory.TRADE_LEVEL_STATISTICS
    )

    PAYOFF_RATIO = AnalyzerTypeInfo(
        name='Payoff Ratio',
        description='Average win size divided by average loss size',
        category=AnalyzerCategory.TRADE_LEVEL_STATISTICS
    )

    EXPECTANCY = AnalyzerTypeInfo(
        name='Expectancy',
        description='Expected profit per trade combining hit rate and payoff',
        category=AnalyzerCategory.TRADE_LEVEL_STATISTICS
    )

    PROFIT_FACTOR = AnalyzerTypeInfo(
        name='Profit Factor',
        description='Gross profit divided by gross loss',
        category=AnalyzerCategory.TRADE_LEVEL_STATISTICS
    )

    KELLY_CRITERION = AnalyzerTypeInfo(
        name='Kelly Criterion',
        description='Theoretically optimal fraction for long-run growth given hit rate and payoff odds',
        category=AnalyzerCategory.TRADE_LEVEL_STATISTICS
    )

    @classmethod
    def all(cls) -> list['AnalyzerType']:
        return list[AnalyzerType](cls)

    @classmethod
    def availables(cls) -> str:
        availables = FactoryDict[AnalyzerCategory, list[AnalyzerType]](list)

        for analyzer in list[AnalyzerType](cls):
            availables[analyzer.value.category].append(analyzer)

        output = ''

        for category, analyzers in availables.items():
            if output:
                output += '\n'

            output += f'''{category}
----------------------------------------\n'''
            for analyzer in analyzers:
                output += f'''  * {analyzer.name} ({analyzer.value.name}):
    {analyzer.value.description}\n'''

        return output

    @property
    def description(self) -> str:
        return self.value.description

    def __str__(self) -> str:
        return self.value.name

    def params(
        self,
        *args, **kwargs
    ) -> Tuple[Self, Params]:
        if self.value.params is None:
            raise ValueError(
                f'No parameters are supported for {self.value.name}'
            )

        return self, self.value.params(*args, **kwargs)
