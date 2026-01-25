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

    def __init__(
        self,
        targets: List[AnalyzeTarget | AnalyzerType]
    ) -> None:
        self._snapshots = []
        self._targets = [
            (target, None) if isinstance(target, AnalyzerType) else target
            for target in targets
        ]

    def add_snapshots(self, *snapshots: PerformanceSnapshot) -> None:
        self.snapshots.extend(snapshots)

    def analyze(
        self
    ) -> dict[AnalyzerType, Any]:
        """
        Analyze the performance snapshots and return the results according to the targets.
        """

        ...
