from typing import (
    Callable, Optional,
    List,
    Any,
    Set,
    Dict,
    Tuple,
    Iterator
)
from decimal import Decimal
from dataclasses import dataclass
from itertools import islice
from datetime import datetime
from enum import Enum

from .symbol import (
    Symbol,
    SymbolManager,
)
from .enums import OrderStatus
from .order_ticket import OrderTicket
from .target import PositionTarget
from .common import (
    class_repr,
    DECIMAL_ZERO,
    FactoryDict,
    EventEmitter
)


class OrderUpdatedType(Enum):
    STATUS_UPDATED = 1
    FILLED_QUANTITY_UPDATED = 2


@dataclass(frozen=True, slots=True)
class Trade:
    """The trade for the order

    Args:
        base_quantity (Decimal): the base asset quantity of the trade
        base_price (Decimal): the average price of the base asset based on the account currency
        quote_quantity (Decimal): the quote asset quantity of the trade
        quote_price (Decimal): the price of the quote asset
        commission_cost (Decimal): the cost of commission asset based on the account currency
    """

    base_quantity: Decimal
    base_price: Decimal
    quote_quantity: Decimal
    quote_price: Decimal
    commission_cost: Decimal


class Order(EventEmitter[OrderUpdatedType]):
    """Order

    Args:
        ticket (OrderTicket): the ticket of the order
        target (PositionTarget): the target which the order is trying to achieve
    """

    ticket: OrderTicket
    target: PositionTarget

    _status: OrderStatus
    _id: Optional[str] = None

    # Mutable properties
    # Cumulative filled base quantity
    filled_quantity: Decimal = DECIMAL_ZERO

    # Cumulative quote asset transacted quantity
    quote_quantity: Decimal = DECIMAL_ZERO

    commission_asset: Optional[str] = None
    commission_quantity: Decimal = DECIMAL_ZERO

    created_at: Optional[datetime]
    trades: List[Trade]

    # Whether the order has been added to the order history
    _added: bool = False

    def __repr__(self) -> str:
        return class_repr(self, keys=[
            'id',
            'ticket',
            'status',
            'target',
            'filled_quantity',
        ])

    def __init__(
        self,
        ticket: OrderTicket,
        target: PositionTarget
    ) -> None:
        super().__init__()

        self.ticket = ticket
        self.target = target

        self._status = OrderStatus.INIT

        self.created_at = None
        self.trades = []

    def update(
        self,
        symbols: SymbolManager,
        status: Optional[OrderStatus] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        filled_quantity: Optional[Decimal] = None,
        quote_quantity: Optional[Decimal] = None,
        commission_asset: Optional[str] = None,
        commission_quantity: Optional[Decimal] = None,
        id: str = None
    ) -> None:
        """see state.update_order()
        """

        if self.status.completed():
            # The order is already completed,
            # it should not be updated any more
            return

        old_filled_quantity = self.filled_quantity
        old_quote_quantity = self.quote_quantity
        old_commission_quantity = self.commission_quantity

        if commission_asset is not None:
            self.commission_asset = commission_asset

        if quote_quantity is not None:
            self.quote_quantity = quote_quantity

        if commission_quantity is not None:
            self.commission_quantity = commission_quantity

        if (
            filled_quantity is not None
            and old_filled_quantity != filled_quantity
        ):
            self.filled_quantity = filled_quantity

            # Only emit the event if the status is not changed
            self.emit(
                OrderUpdatedType.FILLED_QUANTITY_UPDATED,
                self,
                filled_quantity
            )

        self._update_trades(
            symbols,
            old_filled_quantity,
            old_quote_quantity,
            old_commission_quantity
        )

        if id is not None and self._id is None:
            # Do not allow to change order id after set
            self._id = id

        if self._status.lt(OrderStatus.CREATED):
            # Not setting created_at is not fatal
            self.created_at = created_at
            self.updated_at = created_at
        else:
            if updated_at is not None:
                self.updated_at = updated_at

        if status is not None and self._status != status:
            self._status = status
            self.emit(OrderUpdatedType.STATUS_UPDATED, self, status)

    # Ref:
    # https://developers.binance.com/docs/binance-spot-api-docs/user-data-stream#order-update
    def _update_trades(
        self,
        symbols: SymbolManager,
        old_filled_quantity: Decimal,
        old_quote_quantity: Decimal,
        old_commission_quantity: Decimal
    ) -> None:
        """Update the trades of the order, also calculate the valuation value of asset cost according to the current price
        """

        base_quantity_delta = self.filled_quantity - old_filled_quantity

        if base_quantity_delta <= 0:
            # No new fills or the data is stale
            return

        quote_quantity_delta = self.quote_quantity - old_quote_quantity
        commission_quantity_delta = (
            self.commission_quantity - old_commission_quantity
        )

        ticket = self.ticket
        quote_asset = ticket.symbol.quote_asset
        quote_price = symbols.valuation_price(quote_asset)

        base_price = (
            # We should always calculate the cost of a trade by using
            # quote asset transacted quantity * its valuation price.

            # Because when we place a limit order at a certain price,
            # the actual average price might be lower than the price
            quote_quantity_delta * quote_price
            / base_quantity_delta
        )

        commission_cost = DECIMAL_ZERO

        if self.commission_asset is not None:
            commission_cost = (
                commission_quantity_delta * symbols.valuation_price(
                    self.commission_asset
                )
            )

        self.trades.append(
            Trade(
                base_quantity=base_quantity_delta,
                base_price=base_price,
                quote_quantity=quote_quantity_delta,
                quote_price=quote_price,
                commission_cost=commission_cost
            )
        )

    @property
    def status(self) -> OrderStatus:
        return self._status

    @property
    def id(self) -> Optional[str]:
        return self._id


ORDER_COMPARISON_KEYS = [
    'ticket',
    'target'
]


def _compare(
    order: Order,
    key: Any,
    expected: Any
) -> bool:
    if not hasattr(order, key):
        return False

    value = getattr(order, key)

    if isinstance(expected, Callable):
        # Supports a `key` argument so that
        # the matcher function could test multiple attributes
        return expected(value, key)

    elif key in ORDER_COMPARISON_KEYS and isinstance(expected, dict):
        return all(
            _compare(value, k, v)
            for k, v in expected.items()
        )

    return value == expected


class OrderHistory:
    """
    OrderHistory is only for orders once created by the exchange.
    """

    _history: List[Order]

    def __init__(
        self,
        max_size: int
    ) -> None:
        self._max_size = max_size
        self._history = []

    def _check_size(self) -> None:
        if len(self._history) > self._max_size:
            self._history.pop(0)

    def append(
        self,
        order: Order
    ) -> None:
        if not order._added:
            # Mark the order as added to the history
            order._added = True

            self._history.append(order)
            self._check_size()

    def query(
        self,
        descending: bool,
        limit: Optional[int],
        **criteria
    ) -> Iterator[Order]:
        """
        See state.query_orders()
        """

        iterator = (
            reversed(self._history)
            if descending
            else iter(self._history)
        )

        if len(criteria) == 0:
            if limit is None:
                return iterator

            return islice(iterator, limit)

        if limit is None:
            limit = len(self._history)

        return islice((
            order
            for order in iterator
            if all(
                _compare(order, key, expected)
                for key, expected in criteria.items()
            )
        ), limit)


class OrderManager:
    _open_orders: Set[Order]
    _id_orders: Dict[str, Order]

    _orders_to_cancel: Set[Order]

    # Only allow one order for a single symbol
    _symbol_orders: Dict[Symbol, Order]
    _base_asset_orders: FactoryDict[str, Set[Order]]
    # _quote_asset_orders: DictSet[str, Order]

    # Just set it as a public property for convenience
    history: OrderHistory

    def __init__(
        self,
        max_order_history_size: int,
        symbols: SymbolManager
    ) -> None:
        self._symbols = SymbolManager

        self._open_orders = set[Order]()
        self._id_orders = {}
        self._orders_to_cancel = set[Order]()

        self._symbol_orders = {}
        self._base_asset_orders = FactoryDict[str, Set[Order]](set[Order])
        # self._quote_asset_orders = DictSet[str,Order]()
        self.history = OrderHistory(max_order_history_size)

    def _on_order_status_updated(
        self,
        order: Order,
        status: OrderStatus
    ) -> None:
        match status:
            case OrderStatus.CREATED:
                # When an order has an id,
                # it means it has been created by the exchange,
                # so we should add it to the order history
                self.history.append(order)
                if order.id is not None:
                    self._id_orders[order.id] = order

            case OrderStatus.FILLED:
                # The order might be filled directly
                self.history.append(order)
                self._purge_order(order)
                order.off()

            case OrderStatus.ABOUT_TO_CANCEL:
                # If the cancelation request to the server is failed,
                # order.status should be set to ABOUT_TO_CANCEL again,
                # so we should add the order to the cancelation list
                self._orders_to_cancel.add(order)

            case OrderStatus.CANCELLED:
                # Redudant cancellation
                self._purge_order(order)
                order.off()

            case OrderStatus.REJECTED:
                self._purge_order(order)
                order.off()

    def get_order_by_id(self, order_id: str) -> Optional[Order]:
        return self._id_orders.get(order_id)

    def get_order_by_symbol(self, symbol: Symbol) -> Optional[Order]:
        return self._symbol_orders.get(symbol)

    def get_orders_by_base_asset(self, asset: str) -> Set[Order]:
        return self._base_asset_orders[asset]

    # def get_orders_by_quote_asset(self, asset: str) -> Set[Order]:
    #     return self._quote_asset_orders[asset]

    def cancel(self, order: Order) -> None:
        # This method might be called
        # - from outside of the state
        # - when user cancels the order on the exchange manually and
        #   the order status is changed by the callback of the
        #   `executionReport` of the exchange event
        # so we should check the status
        if order.status.lt(OrderStatus.ABOUT_TO_CANCEL):
            order.update(
                self._symbols,
                status = OrderStatus.ABOUT_TO_CANCEL
            )

        self._purge_order(order)

    def _register_order(self, order: Order) -> None:
        self._open_orders.add(order)

        symbol = order.ticket.symbol
        asset = symbol.base_asset

        self._symbol_orders[symbol] = order
        self._base_asset_orders[asset].add(order)
        # self._quote_asset_orders[symbol.quote_asset].add(order)

    def _purge_order(self, order: Order) -> None:
        self._open_orders.discard(order)

        symbol = order.ticket.symbol
        asset = symbol.base_asset

        self._symbol_orders.pop(symbol, None)
        self._base_asset_orders[asset].discard(order)
        # self._quote_asset_orders[symbol.quote_asset].discard(order)

        if order.id is not None:
            self._id_orders.pop(order.id, None)

    def add(
        self,
        order: Order
    ) -> None:
        self._register_order(order)

        order.on(
            OrderUpdatedType.STATUS_UPDATED,
            self._on_order_status_updated
        )

    def get_orders(self) -> Tuple[
        Set[Order],
        Set[Order]
    ]:
        orders_to_cancel = self._orders_to_cancel
        self._orders_to_cancel = set[Order]()

        orders_to_create = set[Order]()

        for order in self._open_orders:
            if order.status is OrderStatus.INIT:
                orders_to_create.add(order)

        return orders_to_create, orders_to_cancel
