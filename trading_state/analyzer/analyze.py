from typing import (
    Iterable,
    Tuple
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
    Usage::

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

        analyzer.add_snapshots(*snapshots)

        analyzer.analyze()
    """

    _targets: list[AnalyzeTarget]
    _snapshots: list[PerformanceSnapshot]

    def __init__(
        self,
        targets: list[AnalyzeTarget | AnalyzerType]
    ) -> None:
        self._snapshots = []
        self._targets = [
            (target, None) if isinstance(target, AnalyzerType) else target
            for target in targets
        ]

    def add_snapshots(self, *snapshots: PerformanceSnapshot) -> None:
        self.snapshots.extend(snapshots)

    def analyze(self) -> dict[AnalyzerType, float]:
        ...
