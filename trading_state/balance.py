# Balance Manager

from typing import (
    Dict,
    Tuple,
    Optional,
    Set,
    Iterator
)
from datetime import datetime
from decimal import Decimal

from .common import (
    DECIMAL_ZERO,
    class_repr,
    SuccessOrException,
    FactoryDict
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
    asset: str
    free: Decimal
    locked: Decimal
    time: Optional[datetime]

    def __init__(
        self,
        asset: str,
        free: Decimal,
        locked: Decimal,
        time: Optional[datetime] = None
    ):
        self.asset = asset
        self.free = free
        self.locked = locked
        self.time = time

    @property
    def total(self) -> Decimal:
        return self.free + self.locked

    def __repr__(self) -> str:
        return class_repr(self, main='asset', keys=[
            'free',
            'locked'
        ])


class DependencyManager:
    """
    The manager to maintain the relationship between
    - balance asset
    - its dependency symbol names for valuation purposes
    """

    _asset_symbols: Dict[str, Set[str]]
    _symbol_assets: FactoryDict[str, Set[str]]

    def __init__(self) -> None:
        self._asset_symbols = {}
        self._symbol_assets = FactoryDict[str, Set[str]](set[str])

    def add(
        self,
        asset: str,
        dependencies: Set[str]
    ) -> None:
        self._asset_symbols[asset] = dependencies

        for symbol_name in dependencies:
            self._symbol_assets[symbol_name].add(asset)

    def clear(self, asset: str) -> None:
        symbols = self._asset_symbols.pop(asset)

        if symbols is None:
            return

        for symbol_name in symbols:
            assets = self._symbol_assets[symbol_name]

            assets.discard(asset)
            if not assets:
                del self._symbol_assets[symbol_name]

    def dependents(self, symbol_name: str) -> Optional[Set[str]]:
        return self._symbol_assets.get(symbol_name)


"""
Low-hanging fruit conclusions:
1. CF before get_account_value:
  => just abandon, unnecessary to track

Case 1:
1. get_account_value for BTC + USDT (but BTC not ready)
2. cash_flow BTC
    => just abandon, because Balance(BTC).total will be treated as a CF
3. set_price BTCUSDT => balance BTC -> cash flow
4. buy BTC, balance BTC increase => no cash flow
5. get_account_value for BTC + USDT (BTC ready now)

Case 2: (impossible)
1. get_account_value for BTC + USDT (but BTC not ready)
2. buy BTC, balance BTC increase => impossible, because BTC is not ready yet
"""


class BalanceManager:
    _balances: Dict[str, Balance]

    # asset -> frozen quantity
    _frozen: Dict[str, Decimal]

    # asset -> notional limit
    _notional_limits: Dict[str, Decimal]

    _checked_symbol_names: Set[str]
    _checked_asset_names: Set[str]
    not_ready_assets: DependencyManager

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

        self.not_ready_assets = DependencyManager()

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

    def get_balance(self, asset: str) -> Optional[Balance]:
        return self._balances.get(asset)

    def get_balances(self) -> Iterator[Balance]:
        return self._balances.values()

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

        # Just set the notional limit
        self._notional_limits[asset] = limit

    def get_notional_limit(self, asset: str) -> Optional[Decimal]:
        return self._notional_limits.get(asset)

    def get_account_value(
        self,
        check_ready: bool
    ) -> Decimal:
        """
        See state.get_account_value()

        It is ok that the valuation price is not ready for some assets,
        if a certain asset becomes ready later, we will treat it as
        a cash flow to the account.
        """

        summary = DECIMAL_ZERO

        for balance in self._balances.values():
            total = balance.total

            if total.is_zero():
                # Dirty data
                continue

            price, deps = self._symbols.valuation_price_info(
                balance.asset
            )

            if price is None:
                if check_ready:
                    self.not_ready_assets.add(balance.asset, deps)
                continue

            summary += total * price

        return summary

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
                # To avoid human mistake,
                # it is a must to set the notional limit explicitly
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
