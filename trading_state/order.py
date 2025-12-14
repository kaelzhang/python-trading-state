from decimal import Decimal
from typing import (
    Callable, Optional,
    List, Iterator,
    Any,
    Set,
    Dict,
    Tuple,
)
from datetime import datetime
from enum import Enum

from .symbol import (
    Symbol,
)
from .enums import (
    OrderStatus
)
from .order_ticket import (
    OrderTicket
)
from .target import (
    PositionTarget,
)
from .common import (
    class_repr,
    DECIMAL_ZERO,
    DictSet,
    EventEmitter
)


class OrderUpdatedType(Enum):
    STATUS_UPDATED = 1
    FILLED_QUANTITY_UPDATED = 2


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
    filled_quantity: Decimal

    # Cumulative quote asset transacted quantity
    quote_quantity: Decimal
    created_at: Optional[datetime]

    def __repr__(self) -> str:
        return class_repr(self, keys=[
            'id',
            'ticket',
            'status',
            'target',
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
        self.filled_quantity = DECIMAL_ZERO
        self.quote_quantity = DECIMAL_ZERO
        self.created_at = None

    def update(
        self,
        status: OrderStatus = None,
        created_at: datetime = None,
        updated_at: datetime = None,
        filled_quantity: Decimal = None,
        quote_quantity: Decimal = None,
        order_id: str = None
    ) -> None:
        """Update the order

        Args:
            status (OrderStatus = None): The new status of the order
            created_at (datetime = None): The creation time of the order
            filled_quantity (Decimal = None): The new filled quantity of the order
            quote_quantity (Decimal = None): The new quote quantity of the order
            order_id (str = None): The client order id

        Usage::

            order.update(
                filled_quantity = Decimal('0.5'),
                quote_quantity = Decimal('1000')
            )
        """

        if quote_quantity is not None:
            self.quote_quantity = quote_quantity

        if filled_quantity is not None:
            self.filled_quantity = filled_quantity

            if status is None:
                # Only emit the event if the status is not changed
                self.emit(
                    OrderUpdatedType.FILLED_QUANTITY_UPDATED,
                    self,
                    filled_quantity
                )

        if status is None:
            return

        if self._status == status:
            # Status not changed
            return

        if status is OrderStatus.CREATED:
            if order_id is None:
                raise ValueError(
                    'order_id is required when status is CREATED'
                )

            self._id = order_id

            # Not setting created_at is not fatal
            self.created_at = created_at
            self.updated_at = created_at
        else:
            if updated_at is not None:
                self.updated_at = updated_at

        self._status = status
        self.emit(OrderUpdatedType.STATUS_UPDATED, self, status)

    @property
    def status(self) -> OrderStatus:
        return self._status

    @property
    def id(self) -> Optional[str]:
        return self._id


def _compare_order(
    order: Order,
    key: Any,
    expected: Any
) -> bool:
    if not hasattr(order, key):
        return False

    value = getattr(order, key)

    if isinstance(expected, Callable):
        return expected(value)

    return value == expected


class OrderHistory:
    _history: List[Order]

    def __init__(
        self,
        max_size: int
    ) -> None:
        self._max_size = max_size
        self._history = []

    def __len__(self) -> int:
        return len(self._history)

    def iterator(self, descending: bool) -> Iterator[Order]:
        """
        Returns an iterator over the history orders.

        Args:
            descending (bool): Whether to iterate the history in descending order, ie. the most recent orders first

        Usage::

            for order in history.iterator(descending=True):
                print(order)
        """
        return (
            reversed(self._orders)
            if descending
            else iter(self._orders)
        )

    def _check_size(self) -> None:
        if len(self._history) > self._max_size:
            self._history.pop(0)

    def append(
        self,
        order: Order
    ) -> None:
        self._history.append(order)
        self._check_size()

    def query(
        self,
        descending: bool,
        limit: Optional[int],
        **criteria
    ) -> List[Order]:
        """
        See state.query_orders()
        """

        matched = []
        count = 0

        for order in self._history.iterator(descending):
            if all(
                _compare_order(order, key, expected)
                for key, expected in criteria.items()
            ):
                matched.append(order)
                count += 1

                if limit is not None and count >= limit:
                    break

        return matched


class OrderManager:
    _open_orders: Set[Order]
    _id_orders: Dict[str, Order]

    _orders_to_cancel: Set[Order]

    # Only allow one order for a single symbol
    _symbol_orders: Dict[Symbol, Order]
    _base_asset_orders: DictSet[str, Order]
    _quote_asset_orders: DictSet[str, Order]

    # Just set it as a public property for convenience
    history: OrderHistory

    def __init__(
        self,
        max_order_history_size: int
    ) -> None:
        self._open_orders = set[Order]()
        self._id_orders = {}
        self._orders_to_cancel = set[Order]()

        self._symbol_orders = {}
        self._base_asset_orders = DictSet[str, Order]()
        self._quote_asset_orders = DictSet[str,Order]()
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
                self._id_orders[order.id] = order

            case OrderStatus.FILLED:
                target = order.target

                if target is not None:
                    target.achieved = True

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

    def get_order_by_id(self, order_id: str) -> Optional[Order]:
        return self._id_orders.get(order_id)

    def get_order_by_symbol(self, symbol: Symbol) -> Optional[Order]:
        return self._symbol_orders.get(symbol)

    def get_orders_by_base_asset(self, asset: str) -> Set[Order]:
        return self._base_asset_orders[asset]

    def get_orders_by_quote_asset(self, asset: str) -> Set[Order]:
        return self._quote_asset_orders[asset]

    def cancel(self, order: Order) -> None:
        # This method might be called
        # - from outside of the state
        # - when user cancels the order on the exchange manually and
        #   the order status is changed by the callback of the
        #   `executionReport` of the exchange event
        # so we should check the status
        if order.status.lt(OrderStatus.ABOUT_TO_CANCEL):
            order.update(
                status = OrderStatus.ABOUT_TO_CANCEL
            )

        self._purge_order(order)

    def _register_order(self, order: Order) -> None:
        self._open_orders.add(order)

        symbol = order.ticket.symbol
        asset = symbol.base_asset
        quote_asset = symbol.quote_asset

        self._symbol_orders[symbol] = order
        self._base_asset_orders[asset].add(order)
        self._quote_asset_orders[quote_asset].add(order)

    def _purge_order(self, order: Order) -> None:
        self._open_orders.discard(order)

        symbol = order.ticket.symbol
        asset = symbol.base_asset
        quote_asset = symbol.quote_asset

        self._symbol_orders.pop(symbol, None)
        self._base_asset_orders[asset].discard(order)
        self._quote_asset_orders[quote_asset].discard(order)

        if order.id is not None:
            self._id_orders.pop(order.id, None)

    def add(
        self,
        order: Order
    ) -> None:
        if order.status is not OrderStatus.INIT:
            # The order is not in the initial state,
            # so it should not be added to the open orders
            return

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

        for order in orders_to_cancel:
            order.update(
                status = OrderStatus.CANCELLING
            )

        orders_to_create = set[Order]()

        for order in self._open_orders:
            if order.status is OrderStatus.INIT:
                orders_to_create.add(order)
                order.update(
                    status = OrderStatus.SUBMITTING
                )

        return orders_to_create, orders_to_cancel
