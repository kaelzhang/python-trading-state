from decimal import Decimal
from typing import (
    Callable, Optional,
    List, Iterator,
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
    _UID: int = 0

    # Immutable properties
    id: int
    ticket: OrderTicket
    position: Optional[AssetPosition]
    # locked_asset: str
    # locked_quantity: Decimal

    # Mutable properties
    _status: OrderStatus

    # Cumulative filled quantity
    filled_quantity: Decimal

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
        self.id = OrderTicket._UID
        OrderTicket._UID += 1

        self._status_updated_callback = None

        self.ticket = ticket
        self.position = position
        # self.locked_asset = locked_asset
        # self.locked_quantity = locked_quantity

        self._status = OrderStatus.INIT
        self.filled_quantity = DECIMAL_ZERO

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
        """

        matched = []

        for order in self._history:
            if all(
                getattr(order, key) == value
                for key, value in criteria.items()
            ):
                matched.append(order)

        return matched
