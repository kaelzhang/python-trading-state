from typing import (
    Set,
    Callable,
    Awaitable,
    Optional
)

from decimal import Decimal

from .symbol import Symbol
from .enums import TicketOrderSide, TicketOrderStatus
from .order_ticket import OrderTicket

from .util import (
    class_repr,
    # float_to_str,
    # datetime_now_str
)

FLOAT_ZERO = 0.


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
    price: Optional[float]

    def __init__(
        self,
        symbol: Symbol,
        value: float,
        asap: bool,
        price: Optional[float]
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
        'id',
        'order_side',
        'symbol',
        'price',
        'quantity',
        'locked_asset',
        'locked_quantity',
        'position',
        'info',
        'status',
        'filled_quantity',
        'time'
    )

    _UID: int = 0

    id: int
    order_side: TicketOrderSide
    symbol: Symbol
    price: float
    quantity: str
    locked_asset: str
    locked_quantity: float
    position: SymbolPosition
    info: Symbol
    status: TicketOrderStatus
    filled_quantity: str
    time: str

    def __init__(
        self,

        # Order type, which will be used by trader
        order_side: TicketOrderSide,
        # The related symbol
        symbol: Symbol,
        # price of symbl
        price: float,
        # We should use str as quantity
        quantity: str,

        # The quantity of which asset has been locked,
        # could be either target asset and cash asset
        locked_asset: str,
        # locked quantity
        locked_quantity: float,

        position: SymbolPosition,
        info: SymbolInfo
    ) -> None:
        self.id = OrderTicket._UID

        OrderTicket._UID += 1

        self.order_side = order_side

        self.symbol = symbol
        self.price = price
        self.quantity = quantity

        self.locked_asset = locked_asset
        self.locked_quantity = locked_quantity

        self.position = position
        self.info = info

        self.status = TicketOrderStatus.INIT

        self.filled_quantity = '0'

        # self.time = datetime_now_str()

    __repr__ = class_repr


class Balance:
    __slots__ = (
        'asset',
        'free',
        'locked'
    )

    asset: str
    free: Decimal
    locked: Decimal

    def __init__(
        self,
        asset: str,
        free: Decimal,
        locked: Decimal
    ):
        self.asset = asset
        self.free = free
        self.locked = locked

    def __repr__(self) -> str:
        return class_repr(self, main='asset')

    def exists(self) -> bool:
        """
        If the total balance is 0, then the asset does not exist
        """

        return not (self.free + self.locked).is_zero()


class TicketGroup:
    """
    Group sell and group buy are mutually exclusive
    """

    __slots__ = (
        'buy',
        'sell'
    )

    # The tickets that buy the asset
    buy: Set[OrderTicket]
    # The tickets that sell the asset
    sell: Set[OrderTicket]

    def __init__(self):
        self.buy = set()
        self.sell = set()

    def get(
        self,
        direction: TicketOrderSide
    ) -> Set[OrderTicket]:
        """
        Get a copied group of the given direction
        """

        return (
            self.buy if direction is TicketOrderSide.BUY else self.sell
        ).copy()

    def close(self, ticket: OrderTicket) -> None:
        """
        Close a ticket
        """

        (self.buy or self.sell).discard(ticket)

    def clear(self) -> None:
        """
        Clear all tickets
        """

        (self.buy or self.sell).clear()


FuncCancelOrder = Callable[[OrderTicket], Awaitable[None]]
