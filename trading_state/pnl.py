from typing import List
from decimal import Decimal
from datetime import datetime

from .common import (
    DECIMAL_ZERO,
    class_repr
)

from .config import TradingConfig
from .symbol import SymbolManager
from .balance import BalanceManager
from .order import (
    OrderManager,
    Order
)
from .position import (
    PositionTracker
)


class CashFlow:
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


class PerformanceNode:
    ...


class PerformanceAnalyzer:
    _cash_flows: List[CashFlow]
    _net_deposits: Decimal = DECIMAL_ZERO

    _inited: bool = False
    _initial_account_value: Decimal = DECIMAL_ZERO
    _realized_pnl_total: Decimal = DECIMAL_ZERO

    def __init__(
        self,
        config: TradingConfig,
        symbols: SymbolManager,
        balances: BalanceManager,
        orders: OrderManager
    ):
        self._config = config
        self._symbols = symbols
        self._balances = balances
        self._orders = orders

        self._cash_flows = []
        self._inited = False

        self._position_tracker = PositionTracker(symbols)

    def init(self) -> None:
        if not self._inited:
            self._inited = True

            self._initial_account_value = self._get_account_value()

    def _get_account_value(self) -> Decimal:
        return self._balances.get_account_value()

    def set_cash_flow(self, cash_flow: CashFlow) -> bool:
        """See state.set_cash_flow()

        Returns
            bool:`True` if the cash flow is set successfully
        """

        asset = cash_flow.asset
        price = self._symbols.valuation_price(asset)

        if price.is_zero():
            # The price is not ready yet, which indicates
            # the total balance of the asset is not included in the
            # account value, that we will treat the total balance as
            # a cash flow to the account later.
            return False

        self._net_deposits += price * cash_flow.quantity

        self._cash_flows.append(cash_flow)
        self.record()

        return True

    def track_order(
        self,
        order: Order
    ) -> None:
        self._realized_pnl_total += self._position_tracker.track_order(order)

        self._record()

    def record(self, *args, **kwargs) -> PerformanceNode:
        return self._record(*args, **kwargs)

    def _record(
        self,
        tags: List[str] = None,
        time: datetime = datetime.now()
    ) -> PerformanceNode:
        """
        Record current performance snapshot

        Args:
            tags (List[str] = None): List of tags to add to the snapshot
            time (datetime): Timestamp of the snapshot

        Returns:
            PerformanceNode: The created performance snapshot
        """

        account_value = self._get_account_value()
