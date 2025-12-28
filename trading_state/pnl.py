from typing import (
    List,
    Dict,
    Optional,
    Any,
    Iterator
)
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime

from .common import (
    DECIMAL_ZERO,
    class_repr
)

from .config import TradingConfig
from .symbol import SymbolManager
from .balance import BalanceManager
from .order import Order
from .position import (
    PositionTracker,
    PositionSnapshots
)


class CashFlow:
    """
    CashFlow is the increase or decrease in the amount of an asset initiated by deposit / withdrawal / transfer.

    Args:
        asset (str): The asset of the cash flow
        quantity (Decimal): The quantity of the cash flow
        time (datetime): The time of the cash flow
    """

    __slots__ = (
        'asset',
        'quantity',
        'time'
    )

    asset: str
    quantity: Decimal
    time: datetime

    def __init__(
        self,
        asset: str,
        quantity: Decimal,
        time: datetime
    ):
        self.asset = asset
        self.quantity = quantity
        self.time = time

    def __repr__(self) -> str:
        return class_repr(self, main='asset')


@dataclass(frozen=True, slots=True)
class BenchmarkPerformance:
    asset: str
    price: Decimal
    benchmark_return: Decimal


BenchmarkPerformances = Dict[str, BenchmarkPerformance]
Labels = Dict[str, Any]


@dataclass(frozen=True, slots=True)
class PerformanceSnapshot:
    time: datetime
    account_value: Decimal
    realized_pnl: Decimal
    positions: PositionSnapshots
    benchmarks: BenchmarkPerformances
    net_cash_flow: Decimal
    labels: Labels

    @property
    def unrealized_pnl(self) -> Decimal:
        return sum(
            position.unrealized_pnl
            for position in self.positions.values()
        )


class PerformanceAnalyzer:
    _inited: bool = False
    _net_cash_flow: Decimal = DECIMAL_ZERO
    _initial_account_value: Decimal = DECIMAL_ZERO
    _realized_pnl_total: Decimal = DECIMAL_ZERO

    _cash_flows: List[CashFlow]
    _perf_nodes: List[PerformanceSnapshot]

    _initial_benchmark_prices: Dict[str, Decimal]

    def __init__(
        self,
        config: TradingConfig,
        symbols: SymbolManager,
        balances: BalanceManager
    ):
        self._config = config
        self._symbols = symbols
        self._balances = balances

        self._cash_flows = []
        self._perf_nodes = []

        self._position_tracker = PositionTracker(symbols, balances)
        self._initial_benchmark_prices = {}

    def _get_account_value(self) -> Decimal:
        return self._balances.get_account_value(True)

    def set_cash_flow(self, cash_flow: CashFlow) -> bool:
        """See state.set_cash_flow()

        Returns:
            bool: True if the cash flow is set successfully, False otherwise
        """

        if not self._inited:
            # The performance analyzer is not initialized yet,
            # we could just ignore the cash flow
            return False

        if self._balances.get_balance(cash_flow.asset) is None:
            # The balance is not ready yet,
            # the balance of the asset will be treated as a cash flow later,
            # so we just ignore the cash flow
            return False

        if self._cash_flows and self._cash_flows[-1].time >= cash_flow.time:
            # The cash flow is not in the correct order,
            # we will ignore it
            return False

        asset = cash_flow.asset
        price = self._symbols.valuation_price(asset)

        if price.is_zero():
            # The price is not ready yet, which indicates
            # the total balance of the asset is not included in the
            # account value, that we will treat the total balance as
            # a cash flow to the account later.
            return False

        quantity = cash_flow.quantity

        self._net_cash_flow += price * quantity
        self._cash_flows.append(cash_flow)
        self._position_tracker.update_position(
            cash_flow.asset,
            quantity,
            price
        )
        self.record()

        return True

    def track_order(
        self,
        order: Order
    ) -> None:
        self._realized_pnl_total += self._position_tracker.track_order(order)

        self._record(
            # For internal labels, follow the '__XXX__' format
            labels={
                '__ORDER__': True
            },
            time=order.updated_at
        )

    def record(self, *args, **kwargs) -> PerformanceSnapshot:
        # If the performance analyzer is not initialized,
        # we will initialize it
        self._init()

        return self._record(*args, **kwargs)

    def _init(self) -> None:
        if self._inited:
            return

        self._inited = True
        self._initial_account_value = self._get_account_value()

        for asset in self._config.benchmark_assets:
            self._initial_benchmark_prices[asset] = (
                self._symbols.valuation_price(asset)
            )

        self._position_tracker.init()

    def _record(
        self,
        time: Optional[datetime] = None,
        labels: Labels = {}
    ) -> PerformanceSnapshot:
        """
        see state.record()
        """

        if time is None:
            time = datetime.now()

        account_value = self._get_account_value()
        realized_pnl = self._realized_pnl_total

        snapshots = self._position_tracker.snapshots()
        benchmarks: BenchmarkPerformances = {}

        for asset in self._config.benchmark_assets:
            price = self._symbols.valuation_price(asset)
            benchmark_return = DECIMAL_ZERO
            init_price = self._initial_benchmark_prices[asset]

            if init_price > 0:
                benchmark_return = (price - init_price) / init_price
            else:
                self._initial_benchmark_prices[asset] = price

            benchmarks[asset] = BenchmarkPerformance(
                asset,
                price,
                benchmark_return
            )

        node = PerformanceSnapshot(
            time=time,
            account_value=account_value,
            realized_pnl=realized_pnl,
            positions=snapshots,
            benchmarks=benchmarks,
            net_cash_flow=self._net_cash_flow,
            labels=labels
        )

        self._perf_nodes.append(node)

        return node

    def iterator(self, descending: bool) -> Iterator[PerformanceSnapshot]:
        return (
            reversed(self._perf_nodes)
            if descending
            else iter(self._perf_nodes)
        )

    def order_pnl(self, order: Order) -> Decimal:
        ...
