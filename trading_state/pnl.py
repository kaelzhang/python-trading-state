from typing import List
from decimal import Decimal
from datetime import datetime

from .common import (
    DECIMAL_ZERO,
    class_repr
)

from .config import TradingConfig
from .symbol import Symbols
from .balance import BalanceManager
from .order import (
    OrderManager,
    Order
)


class CashFlow:
    __slots__ = (
        'asset',
        'amount',
        'time'
    )

    asset: str
    amount: Decimal
    time: datetime

    def __init__(
        self,
        asset: str,
        amount: Decimal,
        time: datetime
    ):
        self.asset = asset
        self.amount = amount
        self.time = time

    def __repr__(self) -> str:
        return class_repr(self, main='asset')


class PerfNode:
    ...


class PerformanceAnalyzer:
    _cash_flows: List[CashFlow]
    _net_deposits: Decimal = DECIMAL_ZERO
    _inited: bool = False
    _initial_account_value: Decimal = DECIMAL_ZERO

    def __init__(
        self,
        config: TradingConfig,
        symbols: Symbols,
        balances: BalanceManager,
        orders: OrderManager
    ):
        self._config = config
        self._symbols = symbols
        self._balances = balances
        self._orders = orders

        self._cash_flows = []
        self._inited = False

    def init(self) -> None:
        if not self._inited:
            self._inited = True

            self._initial_account_value = self._get_account_value()

    def _get_account_value(self) -> Decimal:
        return self._balances.get_account_value()

    def set_cash_flow(self, cash_flow: CashFlow) -> None:
        """See state.set_cash_flow()
        """

        asset = cash_flow.asset
        price = self._symbols.valuation_price(asset)
        self._net_deposits += price * cash_flow.amount

        self._cash_flows.append(cash_flow)
        self.record()

    def record(
        self,
        tags: List[str] = None,
        time: datetime = datetime.now()
    ) -> PerfNode:
        """
        Record current performance snapshot

        Args:
            tags (List[str] = None): List of tags to add to the snapshot
            time (datetime): Timestamp of the snapshot

        Returns:
            PerfNode: The created performance snapshot
        """

        # account_value = self._get_account_value()

    def register_order(
        self,
        order: Order
    ) -> None:
        ...

    def summary(self) -> None:
        ...
