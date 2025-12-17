# Ref
# https://developers.binance.com/docs/binance-spot-api-docs/rest-api/trading-endpoints

from __future__ import annotations
from typing import (
    List, Optional, Union,
    TYPE_CHECKING
)
from enum import Enum

from decimal import Decimal

from .enums import (
    OrderSide, OrderType, TimeInForce, MarketQuantityType, STPMode
)

from .common import (
    class_repr
)

if TYPE_CHECKING:
    from .symbol import Symbol


class BaseOrderTicket:
    """
    An order ticket contains the necessary information to create an order,
    but order ticket is not an order
    """

    type: OrderType

    BASE_MANDOTORY_PARAMS: List[str] = ['symbol', 'side', 'quantity']
    ADDITIONAL_MANDOTORY_PARAMS: List[str] = []

    symbol: Symbol
    side: OrderSide
    quantity: Decimal

    BASE_OPTIONAL_PARAMS: List[str] = ['stp']
    ADDITIONAL_OPTIONAL_PARAMS: List[str] = []

    stp: STPMode | None = None

    @property
    def REQUIRED_PARAMS(self) -> List[str]:
        return (
            self.BASE_MANDOTORY_PARAMS + self.ADDITIONAL_MANDOTORY_PARAMS
        )

    @property
    def OPTIONAL_PARAMS(self) -> List[str]:
        return (
            self.BASE_OPTIONAL_PARAMS + self.ADDITIONAL_OPTIONAL_PARAMS
        )

    @property
    def PARAMS(self) -> List[str]:
        return (
            self.REQUIRED_PARAMS + self.OPTIONAL_PARAMS
        )

    others: dict[str, any]

    def __repr__(self) -> str:
        return class_repr(self, keys=['type', *self.PARAMS])

    def __init__(
        self,
        **kwargs
    ) -> None:
        for param in self.REQUIRED_PARAMS:
            if param not in kwargs:
                raise ValueError(f'"{param}" is a required parameter for Order')

            setattr(self, param, kwargs[param])
            del kwargs[param]

        for param in self.OPTIONAL_PARAMS:
            if param in kwargs:
                setattr(self, param, kwargs[param])
                del kwargs[param]

        self.others = kwargs
        self._validate_params()

    def has(self, param: str) -> bool:
        return (
            param in self.PARAMS
            and getattr(self, param) is not None
        )

    def is_a(
        self,
        order_type: OrderType,
        **kwargs
    ) -> bool:
        if self.type != order_type:
            return False

        if not kwargs:
            return True

        for key, value in kwargs.items():
            if value is None:
                continue

            if getattr(self, key) != value:
                return False

        return True

    def _validate_params(self) -> None:
        # do nothing by default
        # no extra validation is needed
        ...


class LimitOrderTicket(BaseOrderTicket):
    type = OrderType.LIMIT

    ADDITIONAL_MANDOTORY_PARAMS = ['price', 'time_in_force']

    price: Decimal
    time_in_force: TimeInForce

    ADDITIONAL_OPTIONAL_PARAMS = ['post_only', 'iceberg_quantity']

    post_only: bool = False
    iceberg_quantity: Optional[Decimal] = None

    def _validate_params(self) -> None:
        if (
            self.time_in_force is not None
            and self.post_only
        ):
            raise ValueError('post_only is not allowed with time_in_force')


class MarketOrderTicket(BaseOrderTicket):
    type = OrderType.MARKET

    ADDITIONAL_MANDOTORY_PARAMS = ['quantity_type', 'estimated_price']

    quantity_type: MarketQuantityType

    # We introduced a special parameter for market order
    # to estimate the quantity for MARKET_LOT_SIZE filter
    estimated_price: Decimal


def validate_stop_price_and_trailing_delta(self) -> None:
    if self.stop_price is None and self.trailing_delta is None:
        raise ValueError('Either stop_price or trailing_delta must be set')

    # stop_price and trailing_delta could be combined together


PARAMS_STOP_PRICE_AND_TRAILING_DELTA = ['stop_price', 'trailing_delta']
PARAMS_ST_AND_ICEBERG_QUANTITY = [
    *PARAMS_STOP_PRICE_AND_TRAILING_DELTA,
    'iceberg_quantity'
]

class StopLossOrderTicket(BaseOrderTicket):
    type = OrderType.STOP_LOSS

    ADDITIONAL_OPTIONAL_PARAMS = PARAMS_STOP_PRICE_AND_TRAILING_DELTA

    stop_price: Optional[Decimal] = None
    trailing_delta: Optional[Decimal] = None

    _validate_params = validate_stop_price_and_trailing_delta


class StopLossLimitOrderTicket(StopLossOrderTicket):
    type = OrderType.STOP_LOSS_LIMIT

    ADDITIONAL_MANDOTORY_PARAMS = ['price', 'time_in_force']

    price: Decimal
    time_in_force: TimeInForce

    ADDITIONAL_OPTIONAL_PARAMS = PARAMS_ST_AND_ICEBERG_QUANTITY

    iceberg_quantity: Optional[Decimal] = None


# At the rest API level, they are structurally identical
class TakeProfitOrderTicket(StopLossOrderTicket):
    type = OrderType.TAKE_PROFIT


class TakeProfitLimitOrderTicket(StopLossLimitOrderTicket):
    type = OrderType.TAKE_PROFIT_LIMIT


OrderTicket = Union[
    LimitOrderTicket,
    MarketOrderTicket,
    StopLossOrderTicket,
    StopLossLimitOrderTicket,
    TakeProfitOrderTicket,
    TakeProfitLimitOrderTicket
]


class OrderTicketEnum(Enum):
    LIMIT = LimitOrderTicket
    MARKET = MarketOrderTicket
    STOP_LOSS = StopLossOrderTicket
    STOP_LOSS_LIMIT = StopLossLimitOrderTicket
    TAKE_PROFIT = TakeProfitOrderTicket
    TAKE_PROFIT_LIMIT = TakeProfitLimitOrderTicket
