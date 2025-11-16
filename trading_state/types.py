from typing import (
    Set,
    Callable,
    Awaitable,
    Optional
)

from enum import Enum

from .util import (
    class_repr,
    float_to_str,
    # datetime_now_str
)

FLOAT_ZERO = 0.


class Symbol:
    __slots__ = (
        'name',
        'base_asset',
        'quote_asset'
    )

    name: str
    base_asset: str
    quote_asset: str

    def __init__(
        self,
        name: str,
        base_asset: str,
        quote_asset: str
    ) -> None:
        self.name = name
        self.base_asset = base_asset
        self.quote_asset = quote_asset

    def __repr__(self) -> str:
        return self.name


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


class SymbolInfo:
    __slots__ = (
        'symbol',

        'min_price',
        'max_price',

        # Minimum price step of a symbol
        'min_price_step',
        '_min_price_step_precision',

        'min_quantity',
        'max_quantity',

        # Minimum quantity step of an asset based on quote asset
        'min_quantity_step',
        'min_quantity_step_precision',

        # Minimum value which is the product of price and quantity
        'min_notional'
    )

    symbol: Symbol

    min_price: float
    max_price: float
    min_price_step: float

    min_quantity: float
    max_quantity: float
    min_quantity_step: float
    min_quantity_step_precision: int

    min_notional: float

    def __init__(
        self,
        symbol: str,
        base_asset: str,
        quote_asset: str,

        min_price: str,
        max_price: str,
        min_price_step: str,

        min_quantity: str,
        max_quantity: str,
        min_quantity_step: str,

        # An order's notional value is the `price` * `quantity`
        min_notional: str
    ):
        self.symbol = Symbol(
            symbol,
            base_asset,
            quote_asset
        )

        self.min_price = float(min_price)
        self.max_price = float(max_price)

        self.min_price_step = float(min_price_step)
        self._min_price_step_precision = len(
            float_to_str(self.min_price_step)
        ) - 2

        self.min_quantity = float(min_quantity)
        self.max_quantity = float(max_quantity)

        self.min_quantity_step = float(min_quantity_step)
        self.min_quantity_step_precision = len(
            float_to_str(self.min_quantity_step)
        ) - 2

        self.min_notional = float(min_notional)

    def __repr__(self) -> str:
        price = format(
            self.min_price_step,
            f'.{self._min_price_step_precision}f'
        )

        quantity = format(
            self.min_quantity_step,
            f'.{self.min_quantity_step_precision}f'
        )

        return f'<SymbolInfo {self.symbol}: price_step >= {price}, quantity_step >= {quantity}>'


class TicketOrderSide(Enum):
    def __str__(self):
        return self.value

    # USDT is locked
    BUY = 'BUY'
    # BTC is locked
    SELL = 'SELL'


class TicketOrderStatus(Enum):
    def __str__(self):
        return self.value[0]

    # The ticket is initialized but has not been submitted to the exchange,
    # or the ticket is failed to create order so back to the initial state
    INIT = 'INIT'

    # The ticket order is creating,
    #   the request is about to send to the exchange,
    #   but not yet get response
    SUBMITTING = 'SUBMITTING'

    # The ticket order is created via the exchange API
    CREATED = 'CREATED'

    def lt(self, status: 'TicketOrderStatus') -> bool:
        """
        Returns `True` if the current status is less than the given status
        """

        return self.value[1] <= status.value[1]


class OrderTicket:
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
    info: SymbolInfo
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
    free: float
    locked: float

    def __init__(
        self,
        asset: str,
        free: float,
        locked: float
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

        return self.free + self.locked != 0.


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
