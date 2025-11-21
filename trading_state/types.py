from typing import (
    Set,
    Callable,
    Awaitable,
    Optional
)

from decimal import Decimal

from .symbol import Symbol
from .enums import OrderSide, OrderStatus
from .order_ticket import OrderTicket

from .util import (
    class_repr,
    # float_to_str,
    # datetime_now_str
)

# FLOAT_ZERO = 0.


class SymbolPosition:
    """The position ratio relative to the whole balance value

    Args:
        value (float): should between 0 and 1, for now only support 0 or 1
    """

    __slots__ = (
        'symbol',
        'value',
        'asap',
        'price'
    )

    symbol: Symbol
    value: float
    asap: bool
    price: Optional[Decimal]

    def __init__(
        self,
        symbol: Symbol,
        value: float,
        asap: bool,
        price: Optional[Decimal] = None
    ) -> None:
        self.symbol = symbol
        self.value = value

        # If price is fixed, then we could not trade asap,
        # because we could not set the price for market limit order
        self.asap = False if price is not None else asap
        self.price = price

    def __repr__(self) -> str:
        return class_repr(self, main='symbol')

    def equals_to(
        self,
        position: 'SymbolPosition'
    ) -> bool:
        """To detect whether the given `SymbolPosition` has the same goal of the current one
        """

        return (
            self.asap == position.asap
            and self.value == position.value
            and self.price == position.price
        )


class Order:
    __slots__ = (
        'ticket',
        'status',
        # 'id',
        # 'side',
        # 'symbol',
        # 'price',
        # 'quantity',
        # 'locked_asset',
        # 'locked_quantity',
        # 'position',
        # 'info',
        # 'status',
        # 'filled_quantity',
        # 'time'
    )

    _UID: int = 0

    ticket: OrderTicket
    status: OrderStatus
    locked_asset: str
    locked_quantity: Decimal

    # id: int
    # side: OrderSide
    # symbol: Symbol
    # price: float
    # quantity: str
    # locked_asset: str
    # locked_quantity: float
    # position: SymbolPosition
    # info: Symbol
    # status: OrderStatus
    # filled_quantity: str
    # time: str

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
        # self.position = position

        # self.side = side

        # self.symbol = symbol
        # self.price = price
        # self.quantity = quantity

        # self.locked_asset = locked_asset
        # self.locked_quantity = locked_quantity

        # self.position = position
        # self.info = info

        # self.status = OrderStatus.INIT

        # self.filled_quantity = '0'

        # self.time = datetime_now_str()

    __repr__ = class_repr


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
