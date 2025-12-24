# Balance Manager

from typing import (
    Dict,
    Tuple,
    Optional,
    Set,
)

from decimal import Decimal

from .common import (
    DECIMAL_ZERO,
    class_repr,
    SuccessOrException,
)
from .symbol import SymbolManager
from .exceptions import (
    BalanceNotReadyError,
    NotionalLimitNotSetError,
    AssetNotDefinedError,
    SymbolNotDefinedError,
    ValuationNotAvailableError,
    ValuationPriceNotReadyError,
    SymbolPriceNotReadyError
)
from .config import TradingConfig


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


class BalanceManager:
    _balances: Dict[str, Balance]

    # asset -> frozen quantity
    _frozen: Dict[str, Decimal]

    # asset -> notional limit
    _notional_limits: Dict[str, Decimal]

    _checked_symbol_names: Set[str]
    _checked_asset_names: Set[str]

    def __init__(
        self,
        config: TradingConfig,
        symbols: SymbolManager
    ) -> None:
        self._config = config
        self._symbols = symbols

        self._balances = {}
        self._frozen = {}
        self._notional_limits = {}

        self._checked_symbol_names = set[str]()
        self._checked_asset_names = set[str]()

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
        balance: Balance,
        delta: bool = False
    ) -> Tuple[Balance, Balance]:
        """
        See state.set_balances()
        """

        asset = balance.asset
        old_balance = self.get_balance(asset)

        if delta and old_balance is not None:
            balance.free += old_balance.free
            balance.locked += old_balance.locked

        self._balances[balance.asset] = balance

        return old_balance, balance

    def set_notional_limit(
        self,
        asset: str,
        limit: Optional[Decimal]
    ) -> None:
        """
        See state.set_notional_limit()
        """

        if limit is not None and limit < DECIMAL_ZERO:
            limit = None

        if limit is None:
            self._notional_limits.pop(asset, None)
            return

        # Just set the notional limit
        self._notional_limits[asset] = limit

    def get_notional_limit(self, asset: str) -> Optional[Decimal]:
        return self._notional_limits.get(asset)

    def get_account_value(self) -> Decimal:
        """
        See state.get_account_value()
        """

        return sum(
            balance.total * self._symbols.valuation_price(balance.asset)
            for balance in self._balances.values()
        )

    def get_asset_total_balance(self, asset: str, extra: Decimal) -> Decimal:
        """
        Get the total balance of an asset, which excludes
        - frozen balance

        Should be called after `asset_ready`
        """

        total = self._balances.get(asset).total + extra

        return max(
            total - self._frozen.get(asset, DECIMAL_ZERO),
            DECIMAL_ZERO
        )

    def check_symbol_ready(self, symbol_name: str) -> SuccessOrException:
        """
        Check whether the given symbol name is ready to trade

        Prerequisites:
        - the symbol is defined: for example: `BNBBTC`
        - the notional limit of `BNB` is set
        - the valuation price of `BNB`, i.e the price of `BNBUSDT` is ready
        """

        if symbol_name in self._checked_symbol_names:
            return

        symbol = self._symbols.get_symbol(symbol_name)

        if symbol is None:
            return SymbolNotDefinedError(symbol_name)

        if self._symbols.get_price(symbol_name) is None:
            return SymbolPriceNotReadyError(symbol_name)

        exception = self.check_asset_ready(symbol.base_asset)
        if exception is not None:
            return exception

        exception = self.check_asset_ready(symbol.quote_asset)
        if exception is not None:
            return exception

        self._checked_symbol_names.add(symbol_name)

    def check_asset_ready(self, asset: str) -> SuccessOrException:
        """
        Check whether the given asset is ready to trade
        """

        if asset in self._checked_asset_names:
            return

        if not self._symbols.has_asset(asset):
            return AssetNotDefinedError(asset)

        if not self._symbols.is_account_asset(asset):
            if asset not in self._notional_limits:
                return NotionalLimitNotSetError(asset)

            path = self._symbols.valuation_path(asset)

            if path is None:
                return ValuationNotAvailableError(asset)

            for step in path:
                if self._symbols.get_price(step.symbol.name) is None:
                    return ValuationPriceNotReadyError(asset, step.symbol)

        if asset not in self._balances:
            return BalanceNotReadyError(asset)

        self._checked_asset_names.add(asset)
