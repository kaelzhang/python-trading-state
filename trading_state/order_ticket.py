# Ref
# https://developers.binance.com/docs/binance-spot-api-docs/rest-api/trading-endpoints

from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal
from enum import Enum
from typing import (
    ClassVar,
    Optional,
    TYPE_CHECKING,
    Union,
)

from .enums import (
    MarketQuantityType,
    OrderSide,
    OrderType,
    STPMode,
    TimeInForce,
)

if TYPE_CHECKING:
    from .symbol import Symbol


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class BaseOrderTicket:
    """
    Common base for all order ticket value objects.

    Tickets are frozen — once constructed they are immutable. Filters
    produce normalized variants via `dataclasses.replace(ticket, ...)`,
    not by mutating the input.

    `type` is declared as ClassVar so it is bound at the subclass level
    and is NOT a dataclass field.
    """

    type: ClassVar[OrderType]

    symbol: 'Symbol'
    side: OrderSide
    quantity: Decimal
    stp: Optional[STPMode] = None

    def __repr__(self) -> str:
        # Uses `str()` for each field value, not `repr()`, so that
        # StringEnum subclasses (OrderSide, TimeInForce, …) render as
        # 'BUY' / 'GTC' / … and Decimals render without the
        # `Decimal('…')` wrapper. Keeps the surface stable for tests
        # and operator-facing logs.
        cls = type(self)
        parts = [f'type={self.type}']
        for f in fields(self):
            parts.append(f'{f.name}={getattr(self, f.name)}')
        return f'{cls.__name__}({", ".join(parts)})'

    def has(self, param: str) -> bool:
        """
        True when the ticket has `param` as an attribute AND its value is
        not None.
        """
        return getattr(self, param, None) is not None

    def is_a(
        self,
        order_type: OrderType,
        **kwargs,
    ) -> bool:
        """
        True iff self.type matches `order_type` and every kwarg matches
        the corresponding attribute on self.
        """
        if self.type is not order_type:
            return False
        for key, value in kwargs.items():
            if getattr(self, key, None) != value:
                return False
        return True

@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class LimitOrderTicket(BaseOrderTicket):
    type: ClassVar[OrderType] = OrderType.LIMIT

    price: Decimal
    # Required for a plain LIMIT order; must be omitted (None) for
    # post_only=True (the Binance Spot LIMIT_MAKER variant has no
    # timeInForce).
    time_in_force: Optional[TimeInForce] = None
    post_only: bool = False
    iceberg_quantity: Optional[Decimal] = None

    def __post_init__(self) -> None:
        if self.post_only:
            if self.time_in_force is not None:
                raise ValueError(
                    'post_only is not allowed with time_in_force'
                )
        elif self.time_in_force is None:
            raise ValueError(
                'time_in_force is required for a LIMIT order when '
                'post_only is False'
            )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class MarketOrderTicket(BaseOrderTicket):
    type: ClassVar[OrderType] = OrderType.MARKET

    quantity_type: MarketQuantityType

    # Estimated average fill price; used by MARKET_LOT_SIZE math and by
    # NotionalFilter's local approximation of the exchange's lastPrice
    # average (the exchange's authoritative check still happens server-side).
    estimated_price: Decimal


def _require_stop_price_or_trailing_delta(
    stop_price: Optional[Decimal],
    trailing_delta: Optional[Decimal],
) -> None:
    if stop_price is None and trailing_delta is None:
        raise ValueError(
            'Either stop_price or trailing_delta must be set'
        )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StopLossOrderTicket(BaseOrderTicket):
    type: ClassVar[OrderType] = OrderType.STOP_LOSS

    stop_price: Optional[Decimal] = None
    trailing_delta: Optional[Decimal] = None

    def __post_init__(self) -> None:
        _require_stop_price_or_trailing_delta(
            self.stop_price, self.trailing_delta
        )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StopLossLimitOrderTicket(StopLossOrderTicket):
    type: ClassVar[OrderType] = OrderType.STOP_LOSS_LIMIT

    price: Decimal
    time_in_force: TimeInForce
    iceberg_quantity: Optional[Decimal] = None


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class TakeProfitOrderTicket(StopLossOrderTicket):
    type: ClassVar[OrderType] = OrderType.TAKE_PROFIT


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class TakeProfitLimitOrderTicket(StopLossLimitOrderTicket):
    type: ClassVar[OrderType] = OrderType.TAKE_PROFIT_LIMIT


OrderTicket = Union[
    LimitOrderTicket,
    MarketOrderTicket,
    StopLossOrderTicket,
    StopLossLimitOrderTicket,
    TakeProfitOrderTicket,
    TakeProfitLimitOrderTicket,
]


class OrderTicketEnum(Enum):
    LIMIT = LimitOrderTicket
    MARKET = MarketOrderTicket
    STOP_LOSS = StopLossOrderTicket
    STOP_LOSS_LIMIT = StopLossLimitOrderTicket
    TAKE_PROFIT = TakeProfitOrderTicket
    TAKE_PROFIT_LIMIT = TakeProfitLimitOrderTicket
