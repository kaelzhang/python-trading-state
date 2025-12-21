# Balance Manager

from typing import (
    Iterable,
    Dict,
    Optional
)

from decimal import Decimal
from datetime import datetime

from .common import (
    DECIMAL_ZERO,
    class_repr
)
from .symbol import Symbols


class Balance:
    __slots__ = (
        'asset',
        'free',
        'locked'
    )

    asset: str
    free: Decimal
    locked: Decimal

    def __init__(
        self,
        asset: str,
        free: Decimal,
        locked: Decimal
    ):
        self.asset = asset
        self.free = free
        self.locked = locked

    @property
    def total(self) -> Decimal:
        return self.free + self.locked

    def __repr__(self) -> str:
        return class_repr(self, main='asset')


class BalanceUpdate:
    __slots__ = (
        'asset',
        'update',
        'time'
    )

    asset: str
    update: Decimal
    time: datetime

    def __init__(
        self,
        asset: str,
        update: Decimal,
        time: datetime
    ):
        self.asset = asset
        self.update = update
        self.time = time

    def __repr__(self) -> str:
        return class_repr(self, main='asset')


class BalanceManager:
    _balances: Dict[str, Balance]

    # asset -> frozen quantity
    _frozen: Dict[str, Decimal]

    def __init__(
        self,
        symbols: Symbols
    ) -> None:
        self._balances = {}
        self._frozen = {}

    def freeze(
        self,
        asset: str,
        quantity: Optional[Decimal] = None
    ) -> None:
        """
        See state.freeze()
        """

        if quantity is None:
            self._frozen.pop(asset, None)
            return

        self._frozen[asset] = quantity

    def get_balance(self, asset: str) -> Balance:
        return self._balances.get(asset)

    def set_balance(
        self,
        new: Iterable[Balance],
        delta: bool = False
    ) -> None:
        """
        See state.set_balances()
        """
        ...

    def update_balance() -> None:
        ...

    def get_account_value(self) -> Decimal:
        """
        See state.get_account_value()
        """

        return sum(
            balance.total * self._symbols.valuation_price(balance.asset)
            for balance in self._balances.values()
        )

    def get_asset_total_balance(self, asset: str) -> Decimal:
        """
        Get the total balance of an asset, which excludes
        - frozen balance

        Should be called after `asset_ready`
        """

        total = self._balances.get(asset).total

        return max(
            total - self._frozen.get(asset, DECIMAL_ZERO),
            DECIMAL_ZERO
        )
