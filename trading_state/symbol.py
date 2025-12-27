from __future__ import annotations
from typing import (
    List,
    Dict,
    Set,
    Tuple,
    Optional,
    overload,
    Union,
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
from .config import TradingConfig
from .common import (
    DECIMAL_ONE,
    DECIMAL_ZERO,
    FactoryDict,
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
        return f'Symbol({self.base_asset}/{self.quote_asset})'

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
        allow: Union[bool, List[Enum]] = True
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
    """
    Args:
        symbol (Symbol)
        forward (bool): True means the quote_asset of the current symbol is the account asset or the base asset of the next symbol
    """

    symbol: Symbol
    forward: bool


ValuationPath = List[ValuationPathStep]

SearchState = Tuple[str, Optional[bool]]


class SymbolManager:
    # symbol name -> symbol
    _symbols: Dict[str, Symbol]

    _assets: Set[str]
    _underlying_assets: Set[str]

    # base asset -> symbol
    _base_asset_symbols: FactoryDict[str, Set[Symbol]]

    # quote asset -> symbol
    _quote_asset_symbols: FactoryDict[str, Set[Symbol]]

    # symbol name -> price
    _symbol_prices: Dict[str, Decimal]
    _valuation_paths: Dict[str, ValuationPath]

    def __init__(
        self,
        config: TradingConfig
    ) -> None:
        self._config = config
        self._symbols = {}

        self._assets = set[str]()
        self._underlying_assets = set[str]()

        self._base_asset_symbols = FactoryDict[str, Set[Symbol]](set[Symbol])
        self._quote_asset_symbols = FactoryDict[str, Set[Symbol]](set[Symbol])

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

        Returns:
            bool: whether the symbol has been added successfully
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

    def has_symbol(self, symbol_name: str) -> bool:
        return symbol_name in self._symbols

    def has_asset(self, asset: str) -> bool:
        return asset in self._assets

    def valuation_price_info(
        self,
        asset: str
    ) -> Union[Tuple[None, Set[str]], Tuple[Decimal, None]]:
        """
        Get the valuation price of an asset

        Returns: (or)
            - Tuple[None, Set[str]]: if the price is not ready yet, return the dependencies
            - Tuple[Decimal, None]: if the price is ready, return the price and None

        Could be called before `symbol_ready`
        """

        price = DECIMAL_ONE
        dependencies = set[str]()

        path = self.valuation_path(asset)

        if path is None:
            # No valuation path available, which is not supported
            return None, dependencies

        for step in path:
            symbol_price = self.get_price(step.symbol.name)

            if symbol_price is None:
                # Collect all dependencies
                dependencies.add(step.symbol.name)
                continue

            if dependencies:
                continue

            if step.forward:
                price *= symbol_price
            else:
                price /= symbol_price

        if dependencies:
            return None, dependencies

        return price, None

    def valuation_price(self, asset: str) -> Decimal:
        price, _ = self.valuation_price_info(asset)
        return price or DECIMAL_ZERO

    def is_account_asset(self, asset: str) -> bool:
        return asset in self._account_assets

    def valuation_path(
        self, asset: str
    ) -> Optional[ValuationPath]:
        if asset in self._underlying_assets:
            return [ValuationPathStep(self.get_symbol(asset), True)]

        if self.is_account_asset(asset):
            return []

        if asset in self._valuation_paths:
            return self._valuation_paths[asset]

        path = self._shortest_valuation_path(asset)
        if path is not None:
            self._use_primary_account_currency(path)

        # If path is None, we still cache the result,
        # because Symbols need to be initialized
        # at the beginning of the trading session,
        self._valuation_paths[asset] = path

        return path

    def _use_primary_account_currency(self, path: ValuationPath) -> None:
        """
        Try to use primary account currency as much as possible
        """

        last = path[-1]
        symbol = last.symbol

        if symbol.quote_asset not in self._config.alt_account_currencies:
            return

        primary_symbol_name = self._config.get_symbol_name(
            symbol.base_asset, self._config.account_currency
        )

        primary_symbol = self.get_symbol(primary_symbol_name)

        if primary_symbol is None:
            return

        step = ValuationPathStep(primary_symbol, last.forward)
        path[-1] = step

    def _shortest_valuation_path(
        self, asset: str
    ) -> Optional[ValuationPath]:
        """Find the shortest path to get the valuation value of a certain asset.
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
