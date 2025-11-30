from typing import (
    Iterable,
    Dict,
    Tuple,
    Set,
    Optional,
    Callable,
    Union,
    Any,
    List
)
from dataclasses import dataclass, field

from decimal import Decimal

from .exceptions import (
    BalanceNotReadyError,
    NotionalLimitNotSetError,
    AssetNotDefinedError,
    SymbolNotDefinedError,
    ValuationPriceNotReadyError,
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
    OrderUpdatedType,
    OrderManager
)
from .order_ticket import (
    LimitOrderTicket,
    MarketOrderTicket
)
from .types import (
    PositionTarget,
    PositionMetaData
)
from .common import (
    DECIMAL_ZERO,
    DictSet
)


FLOAT_ONE = 1.0
FLOAT_ZERO = 0.0

CreateOrderReturn = Tuple[
    Set[Order], # orders to create
    Set[Order]  # orders to cancel
]

SuccessOrException = Optional[Exception]
type ValueOrException[T] = Union[
    Tuple[None, T],
    Tuple[Exception, None]
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
        account_currency (str): the account currency (ref: https://en.wikipedia.org/wiki/Num%C3%A9raire) to use to:
        - calculate value of limit exposures
        - calculate value of notional limits

        max_order_history_size (int): the maximum size of the order history

        get_symbol_name (Callable[[str, str], str]): a function to get the name of a symbol from its base and quote assets
    """
    account_currency: str
    context: Dict[str, Any] = field(default_factory=dict)
    max_order_history_size: int = 10000
    get_symbol_name: Callable[[str, str], str] = DEFAULT_GET_SYMBOL_NAME


class TradingState:
    """State Phase II

    - support base asset limit exposure between 0 and 1
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
    _checked_asset_names: Set[str]

    _assets: Set[str]
    # base asset -> symbol
    _base_asset_symbols: DictSet[str, Symbol]

    # quote asset -> symbol
    _quote_asset_symbols: DictSet[str, Symbol]

    # symbol name -> price
    _symbol_prices: Dict[str, Decimal]

    # asset -> notional limit
    _notional_limits: Dict[str, Decimal]

    # asset -> position expectation
    _expected: Dict[str, PositionTarget]

    _orders: OrderManager

    def __init__(
        self,
        config: TradingConfig
    ) -> None:
        self._config = config

        self._symbols = {}
        self._checked_symbol_names = set[str]()
        self._checked_asset_names = set[str]()

        self._assets = set[str]()
        self._base_asset_symbols = DictSet[str, Symbol]()
        self._quote_asset_symbols = DictSet[str, Symbol]()

        self._symbol_prices = {}

        self._notional_limits = {}

        self._balances = {}
        self._expected = {}

        self._orders = OrderManager(config.max_order_history_size)

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

        asset = symbol.base_asset
        quote_asset = symbol.quote_asset

        self._assets.add(asset)
        self._assets.add(quote_asset)
        self._base_asset_symbols[asset].add(symbol)
        self._quote_asset_symbols[quote_asset].add(symbol)

    def set_notional_limit(
        self,
        asset: str,
        limit: Optional[Decimal]
    ) -> None:
        """
        Set the notional limit for a certain asset. Pay attention that, by design, it is mandatory to set the notional limit for an asset before trading with the trading state.

        The notional limit of an asset limits:
        - the maximum quantity of the **account_currency** asset the trader could **BUY** the asset,
        - no SELL.
        - the maximum quantity of the asset the trader could **SELL** the asset,
        - no BUY.

        Args:
            asset (str): the asset to set the notional limit for
            limit (Decimal | None): the maximum quantity of the account currency the trader could BUY the asset. `None` means no notional limit.

        For example, if::

            state.set_notional_limit('BTC', Decimal('35000'))

        - current BTC price: $7000
        - base asset balance (USDT): $70000

        Then, the trader could only buy 5 BTC,
        although the balance is enough to buy 10 BTC
        """

        if limit is not None and limit < DECIMAL_ZERO:
            limit = None

        if limit is None:
            self._notional_limits.pop(asset, None)
            return

        # Just set the notional limit
        self._notional_limits[asset] = limit

    def set_balances(
        self,
        new: Iterable[Balance]
    ) -> None:
        """
        Update user balances, including normal assets and quote assets

        Usage::

            state.set_balances([
                Balance('BTC', Decimal('8'), Decimal('0'))
            ])
        """

        for balance in new:
            self._set_balance(balance)

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

    def exposure(
        self,
        asset: str
    ) -> ValueOrException[float]:
        """
        Get the current expected limit exposure or the calculated limit exposure of an asset

        Args:
            asset (str): the asset name to get the limit exposure for

        Returns:
            - Tuple[Exception, None]: the exception if the asset is not ready
            - Tuple[None, float]: the limit exposure of the asset
        """

        exception = self._check_asset_ready(asset)
        if exception is not None:
            return exception, None

        target = self._expected.get(asset)

        if target is not None:
            return None, target.exposure

        return None, self._get_asset_exposure(asset)

    def cancel_order(self, order: Order) -> None:
        """
        Cancel an order from the trading state, and trigger the cancellation the next tick

        The method should have no side effects if called multiple times
        """

        self._orders.cancel(order)

        asset = order.ticket.symbol.base_asset

        current_target = self._expected.get(asset)

        if current_target is None:
            return

        if current_target is order.target:
            # Clean the related expectation,
            # so that current target will be recalculated

            # If we do not remove the expection,
            # the trading state will try to create a new order
            # for the target, which might cause unexpected behavior

            # It is allowed to cancel an order multiple times,
            # use pop to avoid unexpected raise
            self._expected.pop(asset, None)

    def query_orders(
        self,
        descending: bool = True,
        limit: Optional[int] = None,
        **criteria
    ) -> List[Order]:
        """
        Query the history orders by the given criteria

        Args:
            descending (bool): Whether to query the history in descending order, ie. the most recent orders first
            limit (Optional[int]): Maximum number of orders to return. `None` means no limit.
            **criteria: Criteria to match the orders

        Usage::

            results = state.query_orders(
                status=OrderStatus.FILLED,
                created_at=lambda x: x.timestamp() > 1717171717,
            )

            print(results)
        """

        return self._orders.history.query(
            descending=descending,
            limit=limit,
            **criteria
        )

    def get_order_by_id(self, order_id: str) -> Optional[Order]:
        return self._orders.get_order_by_id(order_id)

    def expect(
        self,
        symbol_name: str,
        exposure: float,
        price: Decimal | None,
        use_market_order: bool,
        data: PositionMetaData = {}
    ) -> ValueOrException[bool]:
        """
        Update expectation, returns whether it is successfully updated

        Args:
            symbol_name (str): the name of the symbol to trade with
            exposure (float): the limit exposure to expect, should be between `0.0` and `1.0`. The exposure refers to the current holding of the base asset as a proportion of its maximum allowed notional limit
            use_market_order (bool = False): whether to execute the expectation use_market_orderly, that it will generate a market order
            price (Decimal | None = None): the price to expect. If not provided, the price will be determined by the current price.
            data (Dict[str, Any] = {}): the data attached to the expectation, which will also attached to the created order, order history for future reference.

        Returns:
            Tuple[Optional[Exception], bool]:
            - the reason exception if the expectation is not successfully updated
            - whether the expectation is successfully updated

        Usage::

            # to all-in BTC within the notional limit
            state.expect('BTCUSDT', 1., ...)
        """

        exception = self._check_symbol_ready(symbol_name)

        if exception is not None:
            return exception, None

        symbol = self._get_symbol(symbol_name)
        asset = symbol.base_asset

        if use_market_order:
            price = None
        else:
            if price is None:
                return ExpectWithoutPriceError(asset), None

        # Normalize the exposure to be between 0 and 1
        exposure = max(FLOAT_ZERO, min(exposure, FLOAT_ONE))

        open_target = self._expected.get(asset)

        if open_target is not None:
            if (
                open_target.exposure == exposure
                and open_target.price == price
                and open_target.use_market_order == use_market_order
            ):
                # If the target is the same, no need to update
                # We treat it as a success
                return None, False

        calculated_exposure = self._get_asset_exposure(asset)

        if calculated_exposure == exposure:
            # If the target is the same, no need to update
            # We treat it as a success
            return None, False

        self._expected[asset] = PositionTarget(
            symbol=symbol,
            exposure=exposure,
            use_market_order=use_market_order,
            price=price,
            data=data
        )

        return None, True

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

        return self._orders.get_orders()

    # End of public methods ---------------------------------------------

    def _check_symbol_ready(self, symbol_name: str) -> SuccessOrException:
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

        exception = self._check_asset_ready(symbol.base_asset)

        if exception is not None:
            return exception

        self._checked_symbol_names.add(symbol_name)

    def _check_asset_ready(self, asset: str) -> SuccessOrException:
        """
        Check whether the given asset is ready to trade
        """

        if asset in self._checked_asset_names:
            return

        if asset not in self._assets:
            return AssetNotDefinedError(asset)

        if asset not in self._notional_limits:
            return NotionalLimitNotSetError(asset)

        valuation_symbol_name = self._get_valuation_symbol_name(asset)

        if valuation_symbol_name not in self._symbol_prices:
            return ValuationPriceNotReadyError(asset)

        if asset not in self._balances:
            return BalanceNotReadyError(asset)

        self._checked_asset_names.add(asset)

    def _get_valuation_symbol_name(self, asset: str) -> str:
        return self._config.get_symbol_name(
            asset,
            self._config.account_currency
        )

    def _get_asset_valuation_price(self, asset: str) -> Decimal:
        """
        Get the price of an asset in the account currency

        Should only be called after `asset_ready`
        """

        valuation_symbol = self._get_valuation_symbol_name(asset)
        return self.get_price(valuation_symbol)

    def _set_balance(self, balance: Balance) -> None:
        """
        Set the balance of an asset
        """

        asset = balance.asset
        old_balance = self._balances.get(asset)

        self._balances[balance.asset] = balance

        target = self._expected.get(asset)

        if old_balance is None:
            return

        if target is None or not target.fulfilled:
            # There is no expectation or
            # the expectation is still being fulfilled,
            # we do not need to recalculate the target
            return

        if old_balance.free == balance.free:
            return

        calculated_exposure = self._get_asset_exposure(asset)
        if calculated_exposure == target.exposure:
            return

        # We need to remove the expectation,
        # so that self.exposure() will return the recalculated exposure
        del self._expected[asset]

    def _get_asset_balance(self, asset: str) -> Decimal:
        """
        Get the total balance of an asset, which includes
        - free balance
        - balance locked by open (SELL) orders

        Exclude:
        - balance locked by the Earn wallet or others

        Should be called after `asset_ready`
        """

        balance = self._balances.get(asset)
        free = balance.free

        orders = self._orders.get_orders_by_base_asset(asset)
        for order in orders:
            if order.ticket.side is OrderSide.SELL:
                free += order.ticket.quantity - order.filled_quantity

        return free

    def _get_asset_exposure(self, asset: str) -> float:
        """
        Get the calculated limit exposure of an asset

        Should only be called after `asset_ready`

        Returns:
            float: the calculated limit exposure of the asset
        """

        balance = self._get_asset_balance(asset)
        price = self._get_asset_valuation_price(asset)
        limit = self._notional_limits.get(asset)

        return float(balance * price / limit)

    def _get_symbol(self, symbol_name: str) -> Optional[Symbol]:
        return self._symbols.get(symbol_name)

    def _diff(self) -> None:
        """
        Diff the position expectations
        """

        for target in self._expected.values():
            self._create_order_from_position_target(target)

    def _create_order_from_position_target(
        self,
        target: PositionTarget
    ) -> None:
        """
        Create a order from an asset position target.

        Actually it is always called after `symbol_ready`
        because of `self.expect(...)`
        """

        if target.fulfilled:
            return

        symbol = target.symbol

        existing_order = self._orders.get_order_by_symbol(symbol)

        # We only keep one order for a single symbol
        if existing_order is not None:
            if existing_order.target == target:
                # already a valid order for the position,
                # do not need to create a new order
                return

            self.cancel_order(existing_order)

        # Calculate the valuation value of the asset
        # --------------------------------------------------------
        asset = symbol.base_asset
        balance = self._get_asset_balance(asset)
        valuation_price = self._get_asset_valuation_price(asset)
        value = balance * valuation_price

        limit = self._notional_limits.get(asset)
        value_delta = Decimal(str(target.exposure)) * limit - value
        side = OrderSide.BUY if value_delta > DECIMAL_ZERO else OrderSide.SELL
        quantity = value_delta / valuation_price

        ticket = (
            MarketOrderTicket(
                symbol=symbol,
                side=side,
                quantity=quantity,
                quantity_type=MarketQuantityType.BASE
            )
            if target.use_market_order
            else LimitOrderTicket(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=target.price,
                time_in_force=TimeInForce.GTC
            )
        )

        exception, _ = symbol.apply_filters(
            ticket,
            validate_only=False,
            **self._config.context
        )

        if exception is not None:
            # TODO: logging
            return

        order = Order(
            ticket=ticket,
            target=target
        )

        order.on(
            OrderUpdatedType.STATUS_UPDATED,
            self._on_order_status_updated
        )

        self._orders.add(order)

    def _on_order_status_updated(
        self,
        order: Order,
        status: OrderStatus
    ) -> None:
        if status is OrderStatus.CANCELLED:
            self.cancel_order(order)
