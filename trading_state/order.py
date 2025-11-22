from decimal import Decimal
from typing import Set, Callable, Awaitable

from .types import OrderStatus, OrderSide, OrderTicket, SymbolPosition
from .util import class_repr


class Order:
    __slots__ = (
        'ticket',
        'status'
    )

    _UID: int = 0

    ticket: OrderTicket
    status: OrderStatus
    locked_asset: str
    locked_quantity: Decimal

    def __init__(
        self,
        ticket: OrderTicket,

        # The quantity of which asset has been locked,
        # could be either base asset or quote asset
        locked_asset: str,
        # locked quantity
        locked_quantity: Decimal,

        position: SymbolPosition
    ) -> None:
        self.id = OrderTicket._UID

        OrderTicket._UID += 1

        self.ticket = ticket
        self.status = OrderStatus.INIT
        self.locked_asset = locked_asset
        self.locked_quantity = locked_quantity

    __repr__ = class_repr


class OrderHistory:
    ...


class OrderGroup:
    """
    Group sell and group buy are mutually exclusive
    """

    __slots__ = (
        'buy',
        'sell'
    )

    # The tickets that buy the asset
    buy: Set[Order]
    # The tickets that sell the asset
    sell: Set[Order]

    def __init__(self):
        self.buy = set()
        self.sell = set()

    def get(
        self,
        direction: OrderSide
    ) -> Set[Order]:
        """
        Get a copied group of the given direction
        """

        return (
            self.buy if direction is OrderSide.BUY else self.sell
        ).copy()

    def close(self, order: Order) -> None:
        """
        Close a ticket
        """

        (self.buy or self.sell).discard(order)

    def clear(self) -> None:
        """
        Clear all tickets
        """

        (self.buy or self.sell).clear()


FuncCancelOrder = Callable[[Order], Awaitable[None]]
