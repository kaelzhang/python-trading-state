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

DataclassClass = TypeVar('DataclassClass', bound=DataclassProtocol)


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
    params: Type[DataclassClass] | None = None


class AnalyzerType(Enum):
    TOTAL_RETURN = AnalyzerTypeInfo(
        name='Total Return',
        description='Measures the overall change in equity from start to end',
        category=AnalyzerCategory.RETURN_AND_GROWTH
    )

    ANNUALIZED_RETURN = AnalyzerTypeInfo(
        name='Annualized Return',
        description='Annualizes period returns (not necessarily compounded)',
        category=AnalyzerCategory.RETURN_AND_GROWTH
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
        category=AnalyzerCategory.RISK_ADJUSTED_PERF_RATIOS
    )

    SORTINO_RATIO = AnalyzerTypeInfo(
        name='Sortino Ratio',
        description='Excess return per unit of downside deviation (penalizes only downside risk)',
        category=AnalyzerCategory.RISK_ADJUSTED_PERF_RATIOS
    )

    TREYNOR_RATIO = AnalyzerTypeInfo(
        name='Treynor Ratio',
        description='Excess return per unit of systematic risk (beta)',
        category=AnalyzerCategory.RISK_ADJUSTED_PERF_RATIOS
    )

    IR = AnalyzerTypeInfo(
        name='Information Ratio',
        description='Active return per unit of tracking error (active risk)',
        category=AnalyzerCategory.RISK_ADJUSTED_PERF_RATIOS
    )

    CALMAR_RATIO = AnalyzerTypeInfo(
        name='Calmar Ratio',
        description='Risk-adjusted return using maximum drawdown as the risk denominator (often CAGR/MaxDD)',
        category=AnalyzerCategory.RISK_ADJUSTED_PERF_RATIOS
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
        category=AnalyzerCategory.VOLATILITY_AND_DOWNSIDE_RISK
    )

    DOWNSIDE_DEVIATION = AnalyzerTypeInfo(
        name='Downside Deviation',
        description='Volatility computed only for returns below a threshold (e.g., MAR)',
        category=AnalyzerCategory.VOLATILITY_AND_DOWNSIDE_RISK
    )

    SEMI_VARIANCE = AnalyzerTypeInfo(
        name='Semi-variance',
        description='Variance of downside deviations only',
        category=AnalyzerCategory.VOLATILITY_AND_DOWNSIDE_RISK
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
    ) -> Tuple[Self, DataclassClass]:
        if self.value.params is None:
            raise ValueError(
                f'No parameters are supported for {self.value.name}'
            )

        return self, self.value.params(*args, **kwargs)
