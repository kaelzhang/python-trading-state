from decimal import Decimal
from typing import (
    Callable, Optional,
    List, Iterator,
    Any,
    Set,
    Dict,
)
from datetime import datetime

from .symbol import (
    Symbol,
)
from .enums import (
    OrderStatus
)
from .order_ticket import (
    OrderTicket
)
from .types import (
    AssetPosition,
)
from .common import (
    class_repr,
    DECIMAL_ZERO,
)


StatusUpdatedCallback = Callable[[OrderStatus], None]


class Order:
    ticket: OrderTicket
    position: Optional[AssetPosition]
    # locked_asset: str
    # locked_quantity: Decimal

    # Mutable properties
    _status: OrderStatus
    id: Optional[str] = None

    # Cumulative filled quantity
    filled_quantity: Decimal
    created_at: Optional[datetime]

    _status_updated_callback: Optional[StatusUpdatedCallback]

    def __repr__(self) -> str:
        return class_repr(self, keys=[
            'id',
            'ticket',
            'status',
            'position',
        ])

    def __init__(
        self,
        ticket: OrderTicket,
        position: Optional[AssetPosition],

        # The quantity of which asset has been locked,
        # could be either base asset or quote asset
        # locked_asset: str,
        # locked quantity
        # locked_quantity: Decimal
    ) -> None:
        self._status_updated_callback = None

        self.ticket = ticket
        self.position = position
        # self.locked_asset = locked_asset
        # self.locked_quantity = locked_quantity

        self._status = OrderStatus.INIT
        self.filled_quantity = DECIMAL_ZERO
        self.created_at = None

    def when_status_updated(
        self,
        callback: StatusUpdatedCallback
    ) -> None:
        self._status_updated_callback = callback

    def _trigger_status_updated(self, status: OrderStatus) -> None:
        if self._status_updated_callback is not None:
            self._status_updated_callback(self, status)

    @property
    def status(self) -> OrderStatus:
        return self._status

    @status.setter
    def status(self, status: OrderStatus) -> None:
        self._status = status
        self._trigger_status_updated(status)

        if (
            status is OrderStatus.CANCELLED
            or status is OrderStatus.FILLED
        ):
            # Clean the callback to avoid memory leak
            self._status_updated_callback = None


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

    def __iter__(self) -> Iterator[Order]:
        return iter(self._history)

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
        **criteria
    ) -> List[Order]:
        """
        Query the history orders by the given criteria

        Usage::

            results = history.query(
                status=OrderStatus.FILLED,
                created_at=lambda x: x.timestamp() > 1717171717,
            )

            print(results)
        """

        matched = []

        for order in self._history:
            if all(
                _compare_order(order, key, expected)
                for key, expected in criteria.items()
            ):
                matched.append(order)

        return matched


class OrderManager:
    _orders: Set[Order]
    _symbol_orders: Dict[Symbol, Set[Order]]
