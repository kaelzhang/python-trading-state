from __future__ import annotations
from typing import (
    List,
    Dict,
    Set,
    Tuple,
    Optional,
    overload,
    TYPE_CHECKING
)
from enum import Enum
from decimal import Decimal
from dataclasses import dataclass
from collections import deque

from .filters import BaseFilter, FilterResult
from .enums import (
    FeatureType
)
from .exceptions import (
    BalanceNotReadyError,
    NotionalLimitNotSetError,
    AssetNotDefinedError,
    SymbolNotDefinedError,
    ValuationPriceNotReadyError,
    SymbolPriceNotReadyError
)
from .config import TradingConfig
from .common import (
    DECIMAL_ONE,
    SuccessOrException,
    DictSet,
)

if TYPE_CHECKING:
    from .order_ticket import OrderTicket


class Symbol:
    """
    Symbol is a class that contains the information of a symbol
    """

    name: str
    base_asset: str
    quote_asset: str

    _filters: List[BaseFilter]
    _allowed_features: Dict[FeatureType, bool | List[Enum]]

    def __repr__(self) -> str:
        return f'<Symbol {self.base_asset} / {self.quote_asset}>'

    def __init__(
        self,
        name: str,
        base_asset: str,
        quote_asset: str
    ):
        self.name = name
        self.base_asset = base_asset
        self.quote_asset = quote_asset
        self._filters = []
        self._allowed_features = {}

    @overload
    def allow(
        self,
        feature: FeatureType,
        allow: bool = True
    ) -> None:
        ...

    @overload
    def allow(
        self,
        feature: FeatureType,
        allow: List[Enum]
    ) -> None:
        ...

    def allow(
        self,
        feature: FeatureType,
        allow: bool | List[Enum]
    ) -> None:
        self._allowed_features[feature] = allow

    def support(
        self,
        feature: FeatureType,
        value: Optional[Enum] = None
    ) -> bool:
        """
        Check if the symbol supports a certain feature.

        Args:
            feature: (FeatureType) the feature to check
            value: (Optional[Enum]=None) the value to check

        Returns:
            bool: whether the symbol supports the feature
        """

        allowed = self._allowed_features.get(feature, None)

        if isinstance(allowed, list):
            if value is None:
                raise ValueError(f'symbol.support {feature} requires a value for symbol {self}, but got None')

            return value in allowed

        if value is not None:
            raise ValueError(f'symbol.support {feature} does not allow to test a value for symbol {self}, but got {value}')

        if allowed is None:
            # The feature is not specified, we treat it as not supported
            return False

        return allowed

    def add_filter (self, filter: BaseFilter) -> None:
        self._filters.append(filter)

    def apply_filters(
        self,
        ticket: OrderTicket,
        validate_only: bool,
        **kwargs
    ) -> FilterResult:
        """
        Apply the filter to the order ticket, and try to fix the ticket if possible if `validate_only` is `False`.

        Args:
            ticket: (OrderTicket) the order ticket to apply the filter to
            validate_only: (Optional[bool]=False) whether only to validate the ticket. If `True`, the filter will NOT try to fix the ticket and return an exception even for a tiny mismatch against the filter.

        Returns a tuple of
        - Optional[Exception]: the exception if the filter is not successfully applied
        - bool: whether the ticket has been modified
        """

        modified = False

        for filter in self._filters:
            if not filter.when(ticket):
                continue

            exception, new_modified = filter.apply(
                ticket, validate_only, **kwargs
            )

            if new_modified:
                modified = True

            if exception:
                return exception, modified

        return None, modified


@dataclass(frozen=True, slots=True)
class ValuationPathStep:
    symbol: Symbol
    forward: bool

ValuationPath = List[ValuationPathStep]

SearchState = Tuple[str, Optional[bool]]


class Symbols:
    # symbol name -> symbol
    _symbols: Dict[str, Symbol]

    _checked_symbol_names: Set[str]
    _checked_asset_names: Set[str]

    _assets: Set[str]
    _underlying_assets: Set[str]

    # base asset -> symbol
    _base_asset_symbols: DictSet[str, Symbol]

    # quote asset -> symbol
    _quote_asset_symbols: DictSet[str, Symbol]

    # symbol name -> price
    _symbol_prices: Dict[str, Decimal]
    _valuation_paths: Dict[str, ValuationPath]

    def __init__(
        self,
        config: TradingConfig
    ) -> None:
        self._symbols = {}
        self._checked_symbol_names = set[str]()
        self._checked_asset_names = set[str]()

        self._assets = set[str]()
        self._underlying_assets = set[str]()

        self._base_asset_symbols = DictSet[str, Symbol]()
        self._quote_asset_symbols = DictSet[str, Symbol]()

        self._symbol_prices = {}
        self._valuation_paths = {}
        self._account_assets = set(config.account_currencies)

    def set_price(
        self,
        symbol_name: str,
        price: Decimal
    ) -> bool:
        """
        see state.set_price()
        """

        old_price = self._symbol_prices.get(symbol_name)

        if price == old_price:
            # If the price does not change, should not reset diff
            return False

        self._symbol_prices[symbol_name] = price

        return True

    def get_price(
        self,
        symbol_name: str
    ) -> Decimal | None:
        return self._symbol_prices.get(symbol_name)

    def set_symbol(
        self,
        symbol: Symbol
    ) -> bool:
        """
        see state.set_symbol()
        """

        if symbol.name in self._symbols:
            return False

        self._symbols[symbol.name] = symbol

        asset = symbol.base_asset
        quote_asset = symbol.quote_asset

        self._assets.add(asset)
        self._assets.add(quote_asset)
        self._base_asset_symbols[asset].add(symbol)
        self._quote_asset_symbols[quote_asset].add(symbol)

        if not quote_asset:
            # If the symbol has no quote asset,
            # it is the underlying asset of the account currency,
            # such as a stock asset, AAPL, etc.
            self._underlying_assets.add(asset)

        return True

    def get_symbol(self, symbol_name: str) -> Optional[Symbol]:
        return self._symbols.get(symbol_name)

    def has(self, symbol_name: str) -> bool:
        return symbol_name in self._symbols

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

        symbol = self._symbols.get(symbol_name)

        if symbol is None:
            return SymbolNotDefinedError(symbol_name)

        if symbol_name not in self._symbol_prices:
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

        if asset not in self._assets:
            return AssetNotDefinedError(asset)

        if not self._is_account_asset(asset):
            if asset not in self._notional_limits:
                return NotionalLimitNotSetError(asset)

            for step in self._valuation_path(asset):
                if step.symbol.name not in self._symbol_prices:
                    return ValuationPriceNotReadyError(asset, step.symbol)

        if asset not in self._balances:
            return BalanceNotReadyError(asset)

        self._checked_asset_names.add(asset)

    def valuation_price(self, asset: str) -> Decimal:
        """
        Get the valuation price of an asset

        Should be called after `symbol_ready`
        """

        price = DECIMAL_ONE

        for step in self._valuation_path(asset):
            symbol_price = self.get_price(step.symbol.name)

            if step.forward:
                price *= symbol_price
            else:
                price /= symbol_price

        return price

    def _is_account_asset(self, asset: str) -> bool:
        return asset in self._account_assets

    def _valuation_path(
        self, asset: str
    ) -> Optional[ValuationPath]:
        if asset in self._underlying_assets:
            return [(self.get_symbol(asset), True)]

        if self._is_account_asset(asset):
            return []

        if asset in self._valuation_paths:
            return self._valuation_paths[asset]

        path = self._shortest_valuation_path(asset)
        self._valuation_paths[asset] = path

        return path

    def _shortest_valuation_path(
        self, asset: str
    ) -> Optional[ValuationPath]:
        """
        Returns the shortest List[Step] that starts from `asset` and ends at any asset in
        `account_quote_assets` with the final Step.forward == True. Returns None if no path exists.

        Graph interpretation (standard for tradable pairs):
        - If Step.forward is True : input = symbol.base_asset,  output = symbol.quote_asset
        - If Step.forward is False: input = symbol.quote_asset, output = symbol.base_asset
        """

        # State includes "how we arrived" so we don't discard an asset
        #   reached with different last-step direction.
        # last_forward is None only for the start state (no step taken yet).
        start: SearchState = (asset, None)

        queue = deque[SearchState]([start])
        visited: Set[SearchState] = {start}

        # parent[state] = (prev_state, step_used_to_reach_state)
        prev_linker: Dict[
            SearchState,
            Tuple[SearchState, ValuationPathStep]
        ] = {}

        def search(
            current_state: SearchState,
            forward: bool
        ):
            current_asset, _ = current_state
            symbols = (
                self._base_asset_symbols
                if forward
                else self._quote_asset_symbols
            )[current_asset]

            for symbol in symbols:
                next_state = (
                    symbol.quote_asset if forward else symbol.base_asset,
                    forward
                )
                if next_state not in visited:
                    visited.add(next_state)
                    prev_linker[next_state] = (
                        current_state,
                        ValuationPathStep(symbol, forward)
                    )
                    queue.append(next_state)

        while queue:
            current_state = queue.popleft()
            current_asset, last_forward = current_state

            # Valid terminal condition: last step must be forward,
            #   and we must land in a target quote asset.
            if last_forward is True and current_asset in self._account_assets:
                path: List[ValuationPathStep] = []
                while current_state != start:
                    prev, step = prev_linker[current_state]
                    path.append(step)
                    current_state = prev
                path.reverse()

                return path

            # Expand neighbors.
            # Enforce "S0.symbol.base_asset is A" by restricting the
            #   first expansion to forward steps
            # from symbols whose base_asset is the starting asset.
            if last_forward is None:
                search(current_state, True)
                continue

            search(current_state, True)
            search(current_state, False)

        return None
