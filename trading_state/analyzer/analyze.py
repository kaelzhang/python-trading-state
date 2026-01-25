from typing import (
    Iterable,
    Tuple,
    Any,
    List
)


from trading_state.pnl import PerformanceSnapshot
from .types import (
    AnalyzerType,
    Params
)
from .metrics import (
    AnalysisContext,
    SeriesCache,
    METRIC_CALCULATORS,
    UNSUPPORTED_METRICS,
    SkippedResult
)


PerformanceSnapshots = Iterable[PerformanceSnapshot]

AnalyzeTarget = Tuple[AnalyzerType, Params | None]


class PerformanceAnalyzer:
    """
    Args:
        targets: The targets to analyze.

    Usage::

        from trading_state import TradingStateEvent

        from trading_state.analyzer import (
            PerformanceAnalyzer,
            AnalyzerType
        )

        analyzer = PerformanceAnalyzer([
            # No parameters are needed for total return
            AnalyzerType.TOTAL_RETURN,
            # Default parameters are used for Treynor Ratio
            AnalyzerType.TREYNOR_RATIO,
            # Specify parameters
            AnalyzerType.SHARPE_RATIO.params(
                trading_days=365
            )
        ])

        state.on(
            TradingStateEvent.PERFORMANCE_SNAPSHOT_RECORDED,
            analyzer.add_snapshots
        )

        analyzer.analyze()
    """

    _targets: List[AnalyzeTarget]
    _snapshots: List[PerformanceSnapshot]
    _cache: SeriesCache | None
    _cache_mode: str

    def __init__(
        self,
        targets: List[AnalyzeTarget | AnalyzerType]
    ) -> None:
        self._snapshots = []
        self._targets = [
            (target, None) if isinstance(target, AnalyzerType) else target
            for target in targets
        ]
        self._cache = None
        self._cache_mode = 'append'

    def add_snapshots(self, *snapshots: PerformanceSnapshot) -> None:
        self._snapshots.extend(snapshots)

    def analyze(
        self
    ) -> dict[AnalyzerType, Any]:
        """
        Analyze the performance snapshots and return the results according to the targets.
        """

        context = self._build_context()
        if context is None:
            return {}

        results: dict[AnalyzerType, Any] = {}
        for analyzer, params in self._targets:
            if analyzer in UNSUPPORTED_METRICS:
                results[analyzer] = SkippedResult(
                    UNSUPPORTED_METRICS[analyzer]
                )
                continue
            calculator = METRIC_CALCULATORS.get(analyzer)
            if calculator is None:
                continue
            if params is None and analyzer.value.params is not None:
                params = analyzer.value.params()
            results[analyzer] = calculator(context, params)
        return results

    def _build_context(self) -> AnalysisContext | None:
        if not self._snapshots:
            return None

        if self._cache is None:
            self._cache = SeriesCache()
            if self._is_sorted(self._snapshots):
                self._cache_mode = 'append'
                self._cache.rebuild(self._snapshots)
            else:
                self._cache_mode = 'rebuild'
                self._cache.rebuild(
                    sorted(self._snapshots, key=lambda snap: snap.time)
                )
            return AnalysisContext(self._cache)

        if self._cache_mode != 'append':
            self._cache.rebuild(
                sorted(self._snapshots, key=lambda snap: snap.time)
            )
            return AnalysisContext(self._cache)

        if len(self._snapshots) < self._cache.source_len:
            self._cache_mode = 'rebuild'
            self._cache.rebuild(
                sorted(self._snapshots, key=lambda snap: snap.time)
            )
            return AnalysisContext(self._cache)

        if len(self._snapshots) == self._cache.source_len:
            return AnalysisContext(self._cache)

        new_snapshots = self._snapshots[self._cache.source_len:]
        if not self._is_sorted(new_snapshots):
            self._cache_mode = 'rebuild'
            self._cache.rebuild(
                sorted(self._snapshots, key=lambda snap: snap.time)
            )
            return AnalysisContext(self._cache)

        if self._cache.times and new_snapshots[0].time < self._cache.times[-1]:
            self._cache_mode = 'rebuild'
            self._cache.rebuild(
                sorted(self._snapshots, key=lambda snap: snap.time)
            )
            return AnalysisContext(self._cache)

        self._cache.extend(new_snapshots)
        return AnalysisContext(self._cache)

    @staticmethod
    def _is_sorted(snapshots: List[PerformanceSnapshot]) -> bool:
        return all(
            snapshots[index - 1].time <= snapshots[index].time
            for index in range(1, len(snapshots))
        )
