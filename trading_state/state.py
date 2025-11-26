from typing import (
    List,
    Dict,
    Tuple,
    Set,
    Optional,
    Callable,
    Literal,
    Union
)
from dataclasses import dataclass

from decimal import Decimal

from .exceptions import (
    ExpectWithoutQuotaError,
    SymbolNotDefinedError,
    NumerairePriceNotReadyError,
    SymbolPriceNotReadyError,
    ExpectWithoutPriceError
)
from .enums import (
    OrderStatus,
    OrderSide,
    TimeInForce,
    MarketQuantityType,
)
from .symbol import Symbol
from .balance import Balance
from .order import (
    Order,
    OrderHistory,
)
from .order_ticket import (
    LimitOrderTicket,
    MarketOrderTicket
)
from .types import (
    AssetPosition,
    PositionMetaData
)
from .common import (
    DECIMAL_ZERO,
)


FLOAT_ONE = 1.0
FLOAT_ZERO = 0.0

CreateOrderReturn = Tuple[
    Set[Order], # orders to create
    Set[Order]  # orders to cancel
]

SuccessOrException = Union[
    Tuple[Literal[False], Exception],
    Tuple[Literal[True], None]
]


"""
Logical Principles:
- terminology aligned with professional trading
- be pure
- be passive, no triggers
- no strategies, strategies should be driven by external invocations
- suppose the initialization is done before using the state, including
  - setting up the symbols
  - setting up the balances
  - setting up the symbol prices
- do not handle position control, just proceed with position expectations
  - the purpose of a position change could be marked in position.data

Implementation Principles:
- be sync
- no checking for unexpected param types, but should check unexpected values
- no default parameters for all methods to avoid unexpected behavior
"""


def DEFAULT_GET_SYMBOL_NAME(base_asset: str, quote_asset: str) -> str:
    return f"{base_asset}{quote_asset}"


@dataclass
class TradingConfig:
    """
    Args:
        numeraire (str): the valuation currency (ref: https://en.wikipedia.org/wiki/Num%C3%A9raire) to use to:
        - calculate value of positions
        - calculate value of quotas

        max_order_history_size (int): the maximum size of the order history

        get_symbol_name (Callable[[str, str], str]): a function to get the name of a symbol from its base and quote assets
    """
    numeraire: str
    max_order_history_size: int = 10000
    get_symbol_name: Callable[[str, str], str] = DEFAULT_GET_SYMBOL_NAME


class TradingState:
    """State Phase II

    - support base asset position between 0 and 1
    - support multiple base assets
    - support multiple quote assets

    Convention:
    - For a certain base asset,
      its related tickets should has the same direction

    Design principle:
    - The expectation settings are the final state that the system try to achieve.
    """

    _config: TradingConfig

    # asset -> balance
    _balances: Dict[str, Balance]

    # symbol name -> symbol
    _symbols: Dict[str, Symbol]

    _checked_symbol_names: Set[str]

    # base asset -> symbol
    _base_asset_symbols: Dict[str, Symbol]

    # quote asset -> symbol
    _quote_asset_symbols: Dict[str, Symbol]

    # symbol name -> price
    _symbol_prices: Dict[str, Decimal]

    # asset -> quota
    _quotas: Dict[str, Decimal]

    # asset -> position expectation
    _expected: Dict[str, AssetPosition]
    _old_expected: Dict[str, AssetPosition]

    _orders_to_cancel: Set[Order]

    # Only allow one order for a single symbol
    _symbol_orders: Dict[Symbol, Order]
    _base_asset_orders: Dict[str, Set[Order]]
    _quote_asset_orders: Dict[str, Set[Order]]

    # Order.id -> Order
    _id_orders: Dict[str, Order]
    _open_orders: Set[Order]

    _history: OrderHistory

    def __init__(
        self,
        config: TradingConfig
    ) -> None:
        self._config = config

        self._symbols = {}
        self._checked_symbol_names = set[str]()

        self._base_asset_symbols = {}
        self._quote_asset_symbols = {}

        self._symbol_prices = {}

        self._quotas = {}

        self._balances = {}
        self._expected = {}

        # In the beginning, they are the same
        self._old_expected = self._expected

        self._orders_to_cancel = set[Order]()
        self._symbol_orders = {}
        self._base_asset_orders = {}
        self._quote_asset_orders = {}

        self._id_orders = {}
        self._orders = set[Order]()

        self._history = OrderHistory(max_size=config.max_order_history_size)

    # Public methods
    # ------------------------------------------------------------------------

    def set_price(
        self,
        symbol_name: str,
        price: Decimal
    ) -> bool:
        """
        Set the price of a symbol

        Returns `True` if the price changes
        """

        old_price = self._symbol_prices.get(symbol_name)

        if price == old_price:
            # If the price does not change, should not reset diff
            return False

        self._symbol_prices[symbol_name] = price
        self._reset_diff()

        return True

    def set_symbol(
        self,
        symbol: Symbol
    ) -> None:
        """
        Set (add or update) the symbol info for a symbol

        Args:
            symbol (Symbol): the symbol to set
        """

        self._symbols[symbol.name] = symbol
        self._base_asset_symbols[symbol.base_asset] = symbol
        self._quote_asset_symbols[symbol.quote_asset] = symbol

    def set_quota(
        self,
        asset: str,
        quota: Optional[Decimal]
    ) -> None:
        """
        Set the quota for a certain asset.

        The quota of an asset limits:
        - the maximum quantity of the **numeraire** asset the trader could **BUY** the asset,
        - no SELL.

        Args:
            asset (str): the asset to set the quota for
            quota (Decimal | None): the maximum quantity of the numeraire asset the trader could BUY the asset. `None` means no quota.

        For example, if::

            state.set_quota('BTC', Decimal('35000'))

        - current BTC price: $7000
        - base asset balance (USDT): $70000

        Then, the trader could only buy 5 BTC,
        although the balance is enough to buy 10 BTC
        """

        if quota is not None and quota < DECIMAL_ZERO:
            # Decimal('-1') means no limit
            quota = None

        old_quota = self._quotas.get(asset)

        if quota is None:
            # Just remove the quota if no limit
            del self._quotas[asset]
        else:
            self._quotas[asset] = quota

        if old_quota != quota:
            self._reset_diff()

    def update_balances(
        self,
        new: List[Balance]
    ) -> None:
        """
        Update user balances, including normal assets and quote assets

        Usage::

            state.update_balances([
                Balance('BTC', Decimal('8'), Decimal('0'))
            ])
        """

        balances = []

        for balance in new:
            balances.append(
                (balance.asset, balance)
            )

        self._balances.update(balances)

        # This will trigger diffing again
        self._reset_diff()

    def get_price(
        self,
        symbol_name: str
    ) -> Decimal | None:
        """
        Get the price of a symbol
        """

        return self._symbol_prices.get(symbol_name)

    def support_symbol(self, symbol_name: str) -> bool:
        """
        Check whether the symbol is supported
        """

        return symbol_name in self._symbols

    # def summarize(self):
    #     ...

    def position(
        self,
        asset: str
    ) -> Optional[float]:
        """
        Get the position of an asset

        Args:
            asset (str): the asset name to get the position for

        Returns:
            Optional[float]: the position of the symbol. If the symbol is not supported, returns `None`.
        """

        position = self._expected.get(asset)

        if position is not None:
            return position.value

        return self._get_position_by_asset_value(asset)

    def cancel_order(self, order: Order) -> None:
        """
        Cancel an order from the trading state, and trigger the cancellation the next tick

        The method should have no side effects if called multiple times
        """

        # This method might be called
        # - from outside of the state
        # - from the status updated callback triggered by user actions
        # so we should check the status
        if order.status.lt(OrderStatus.ABOUT_TO_CANCEL):
            order.status = OrderStatus.ABOUT_TO_CANCEL
            self._orders_to_cancel.add(order)

        self._open_orders.discard(order)

        symbol = order.ticket.symbol
        del self._symbol_orders[symbol]

        asset = symbol.base_asset

        current_position = self._expected.get(asset)

        if current_position is None:
            return

        if current_position is order.position:
            # Clean the related expectation,
            # so that current position will be recalculated
            del self._expected[asset]

    # Prerequisites:
    # - current price: $10
    # - balance: 1
    # - quota: $20
    # - open orders:
    #   - buy 1 @ $9
    #   - sell 1 @ $11
    # - quota: $11
    #
    # Case 1:
    # expect: + 0.1 @ current (+ $2)
    # Result:
    # we will cancel the buy order
    # since we plan to buy in a higher price

    def expect(
        self,
        symbol_name: str,
        position: float,
        price: Decimal | None,
        asap: bool,
        data: PositionMetaData = {}
    ) -> SuccessOrException:
        """
        Update expectation, returns whether it is successfully updated

        Args:
            symbol_name (str): the name of the symbol to trade with
            position (float): the position to expect, should be between `0.0` and `1.0`. The position refers to the current holding of the base asset as a proportion of its maximum allowed quota
            asap (bool = False): whether to execute the expectation immediately, that it will generate a market order
            price (Decimal | None = None): the price to expect. If not provided, the price will be determined by the current price.
            data (Dict[str, Any] = {}): the data attached to the expectation, which will also attached to the created order, order history for future reference.

        Returns:
            tuple:
            - bool: whether the expectation is successfully updated
            - Optional[Exception]: the reason exception if the expectation is not successfully updated

        Usage::

            # to all-in BTC within the quota
            state.expect('BTCUSDT', 1., ...)
        """

        passed, exception = self._check_symbol_name(symbol_name)

        if not passed:
            return False, exception

        symbol = self._get_symbol(symbol_name)
        asset = symbol.base_asset

        # Normalize the position to be between 0 and 1
        position = max(FLOAT_ZERO, min(position, FLOAT_ONE))

        current_position = self._get_position(asset)

        if current_position == position:
            # If the position is the same, no need to update
            # We treat it as a success
            return True, None

        # TODO: whether we should use a new dict here?
        # Create a new dict so that will be considered as changed
        self._expected = {
            **self._expected
        }

        if asap:
            price = None
        else:
            if price is None:
                return False, ExpectWithoutPriceError(asset)

        self._expected[asset] = AssetPosition(
            symbol=symbol,
            value=position,
            asap=asap,
            price=price,
            data=data
        )

        return True, None

    def get_orders(self) -> Tuple[
        Set[Order],
        Set[Order]
    ]:
        """
        Diff the orders, and get all unsubmitted orders, by calling this method
        - Orders of OrderStatus.INIT -> OrderStatus.SUBMITTING
        - Orders of OrderStatus.ABOUT_TO_CANCEL -> OrderStatus.CANCELLING

        Returns
            tuple:
            - a set of available orders
            - a set of orders to cancel
        """

        self._diff()

        orders_to_cancel = self._orders_to_cancel
        self._orders_to_cancel = set[Order]()

        for order in orders_to_cancel:
            order.status = OrderStatus.CANCELLING

        orders_to_create = set[Order]()

        for order in self._open_orders:
            if order.status is OrderStatus.INIT:
                orders_to_create.add(order)
                order.status = OrderStatus.SUBMITTING
                self._history.append(order)

        return orders_to_create, orders_to_cancel

    # End of public methods ---------------------------------------------

    def _check_symbol_name(self, symbol_name: str) -> SuccessOrException:
        """
        Check whether the given symbol name is ready to trade

        Prerequisites:
        - the symbol is defined: for example: `BNBBTC`
        - the quota of `BNB` is set
        - the numeraire price of `BNB`, i.e the price of `BNBUSDT` is ready
        """

        if symbol_name in self._checked_symbol_names:
            return True, None

        symbol = self._symbols.get(symbol_name)

        if symbol is None:
            return False, SymbolNotDefinedError(symbol_name)

        if symbol_name not in self._symbol_prices:
            return False, SymbolPriceNotReadyError(symbol_name)

        asset = symbol.base_asset

        if asset not in self._quotas:
            return False, ExpectWithoutQuotaError(asset)

        numeraire_symbol_name = self._get_numeraire_symbol_name(asset)

        if numeraire_symbol_name not in self._symbol_prices:
            return False, NumerairePriceNotReadyError(asset)

        self._checked_symbol_names.add(symbol_name)

        return True, None

    def _get_numeraire_symbol_name(self, asset: str) -> str:
        return self._config.get_symbol_name(
            asset,
            self._config.numeraire
        )

    def _get_asset_numeraire_price(self, asset: str) -> Optional[Decimal]:
        numeraire_symbol = self._get_numeraire_symbol_name(asset)
        return self.get_price(numeraire_symbol)

    def _get_symbol(self, symbol_name: str) -> Optional[Symbol]:
        return self._symbols.get(symbol_name)

    def _has_diff(self) -> bool:
        """
        Check whether the trading state has diff
        """

        return self._expected is self._old_expected

    def _reset_diff(self):
        self._old_expected = {}

    def _diff(self) -> None:
        """
        Diff the position expectations
        """

        if self._has_diff():
            # If there is no new expectation,
            # so skip diffing
            return

        self._old_expected = self._expected

        for position in self._expected.values():
            self._create_order_from_position(position)

    def _create_order_from_position(
        self,
        position: AssetPosition
    ) -> None:
        """
        Create a order from an asset position
        """

        if position.reached:
            return

        symbol = position.symbol

        existing_order = self._symbol_orders.get(symbol)

        # We only keep one order for a single symbol
        if existing_order is not None:
            if existing_order.position == position:
                # already a valid order for the position,
                # do not need to create a new order
                return

            self.cancel_order(existing_order)

        # Precheck price
        # --------------------------------------------------------
        price = position.price
        symbol_name = symbol.name

        if not position.asap and price is None:
            price = self.get_price(symbol_name)

            if price is None:
                # The price is not ready,
                # then skip to avoid unexpected result,
                # it will have another in the next tick
                return

        # Precheck current position
        # --------------------------------------------------------
        asset = symbol.base_asset
        numeraire_price = self._get_asset_numeraire_price(asset)

        if numeraire_price is None:
            # The asset is not supported for valuation
            return None

        balance = self._balances.get(asset)

        free = balance.free if balance is not None else DECIMAL_ZERO

        value = numeraire_price * free

        # quota must not be None, because of the position
        quota = self._quotas.get(asset)
        value_delta = position.value * quota - value
        side = OrderSide.BUY if value_delta > DECIMAL_ZERO else OrderSide.SELL
        quantity = value_delta / numeraire_price

        ticket = (
            MarketOrderTicket(
                symbol=symbol,
                side=side,
                quantity=quantity,
                quantity_type=MarketQuantityType.BASE
            )
            if position.asap
            else LimitOrderTicket(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=position.price,
                time_in_force=TimeInForce.GTC
            )
        )

        exception, _ = symbol.apply_filters(ticket, validate_only=False)

        if exception is not None:
            return

        order = Order(
            ticket=ticket,
            position=position
        )

        order.when_status_updated(
            self._on_order_status_updated
        )

        self._open_orders.add(order)

    def _on_order_status_updated(
        self,
        order: Order,
        status: OrderStatus
    ) -> None:
        match status:
            case OrderStatus.FILLED:
                position = order.position

                if position is not None:
                    position.reached = True

            case OrderStatus.CANCELLED:
                self.cancel_order(order)
