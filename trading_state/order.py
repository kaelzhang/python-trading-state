from typing import (
    Callable, Optional,
    List,
    Any,
    Set,
    Dict,
    Iterator
)
from decimal import Decimal
from dataclasses import dataclass
from itertools import islice
from datetime import datetime
from enum import Enum

from .symbol import (
    SymbolManager,
)
from .enums import OrderStatus
from .order_ticket import OrderTicket
from .common import (
    class_repr,
    DECIMAL_ZERO,
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
        data (Optional[Dict[str, Any]]): caller-supplied metadata stored
            on the order; opaque to trading_state. Defaults to an empty
            dict when None. A defensive copy is taken on construction so
            mutating the dict the caller passed in (or sharing the same
            default across orders) cannot leak across orders.
    """

    ticket: OrderTicket
    data: Dict[str, Any]

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
    updated_at: Optional[datetime]
    trades: List[Trade]

    # Whether the order has been added to the order history
    _added: bool = False

    def __repr__(self) -> str:
        return class_repr(self, keys=[
            'id',
            'ticket',
            'status',
            'data',
            'filled_quantity',
        ])

    def __init__(
        self,
        ticket: OrderTicket,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.ticket = ticket
        # Defensive copy: callers may pass a shared dict (or fall back
        # to the default {}); we never want order.data leakage across
        # instances.
        self.data = dict(data) if data else {}

        self._status = OrderStatus.INIT

        self.created_at = None
        self.updated_at = None
        self.trades = []

    def update(
        self,
        symbols: SymbolManager, /,
        *,
        status: Optional[OrderStatus],
        updated_at: Optional[datetime],
        id: Optional[str],
        filled_quantity: Optional[Decimal],
        quote_quantity: Optional[Decimal],
        commission_asset: Optional[str],
        commission_quantity: Optional[Decimal],
    ) -> None:
        """
        Apply field-level updates to this order.

        All keyword arguments are required (pass `None` for a field
        that is not being updated). Stale-data detection (status /
        filled_quantity / updated_at regression) is owned by
        TradingState.update_order; this method assumes the caller has
        already gated stale inputs and the values it receives are
        monotonic for any field that is not None.

        Notes:
        - When the order transitions out of pre-CREATED state, the
          first non-None `updated_at` is also recorded as `created_at`.
        - A change in filled_quantity emits
          OrderUpdatedType.FILLED_QUANTITY_UPDATED.
        - A change in status emits OrderUpdatedType.STATUS_UPDATED.
        - Completed orders (status.completed()) reject further updates
          silently.
        """

        if self.status.completed():
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

            self.emit(
                OrderUpdatedType.FILLED_QUANTITY_UPDATED,
                self,
                filled_quantity,
            )

        self._update_trades(
            symbols,
            old_filled_quantity,
            old_quote_quantity,
            old_commission_quantity,
        )

        if id is not None and self._id is None:
            # Do not allow to change order id after set
            self._id = id

        if self._status.lt(OrderStatus.CREATED):
            # First crossing into CREATED-or-beyond: treat the incoming
            # `updated_at` as the order's created_at too.
            if updated_at is not None:
                self.created_at = updated_at
                self.updated_at = updated_at
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
    'data',
]


def _compare(
    order: Any,
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

    if key in ORDER_COMPARISON_KEYS and isinstance(expected, dict):
        if isinstance(value, dict):
            # Subset match: every expected (k, v) must be present in
            # value with the same value. Lets callers do
            # `query_orders(data={'strategy': 'momentum'})` without
            # having to match the full dict exactly.
            return all(value.get(k) == v for k, v in expected.items())
        # Object subset match: recurse on each expected attribute.
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

    # Just set it as a public property for convenience
    history: OrderHistory

    def __init__(
        self,
        max_order_history_size: int,
        symbols: SymbolManager
    ) -> None:
        self._symbols = symbols

        self._open_orders = set[Order]()
        self._id_orders = {}

        self.history = OrderHistory(max_order_history_size)

    def _on_order_status_updated(
        self,
        order: Order,
        status: OrderStatus
    ) -> None:
        match status:
            case OrderStatus.CREATED:
                # When an order has been created by the exchange,
                # add it to the order history once we see a non-None id.
                self.history.append(order)
                if order.id is not None:
                    self._id_orders[order.id] = order

            case OrderStatus.FILLED:
                # The order might be filled directly
                self.history.append(order)
                self._purge_order(order)
                order.off()

            case OrderStatus.CANCELLED:
                # Redudant cancellation
                self._purge_order(order)
                order.off()

            case OrderStatus.REJECTED:
                self._purge_order(order)
                order.off()

    def get_order_by_id(self, order_id: str) -> Optional[Order]:
        return self._id_orders.get(order_id)

    @property
    def open_orders(self) -> Set[Order]:
        return self._open_orders

    def cancel(self, order: Order) -> None:
        """
        Mark `order` as being cancelled. Caller is responsible for the
        actual cancel request to the exchange and for relaying the
        exchange's CANCELLED confirmation back via state.update_order.

        Idempotent: no-op once status >= CANCELLING.
        """
        if order.status.lt(OrderStatus.CANCELLING):
            order.update(
                self._symbols,
                status=OrderStatus.CANCELLING,
                updated_at=None,
                id=None,
                filled_quantity=None,
                quote_quantity=None,
                commission_asset=None,
                commission_quantity=None,
            )

    def _register_order(self, order: Order) -> None:
        self._open_orders.add(order)

    def _purge_order(self, order: Order) -> None:
        self._open_orders.discard(order)

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
