# Ref
# https://developers.binance.com/docs/binance-spot-api-docs/rest-api/trading-endpoints

# from dataclasses import dataclass
from typing import (
    List, Union, Optional
)

from decimal import Decimal

from .symbol import Symbol
from .enums import (
    OrderSide, OrderType, TimeInForce, MarketQuantityType, STPMode
)
from .util import dynamic_dataclass


@dynamic_dataclass
class BaseOrderTicket:
    """
    An order ticket contains the necessary information to create an order
    """

    BASE_MANDOTORY_PARAMS: List[str] = ['symbol', 'side']
    ADDITIONAL_MANDOTORY_PARAMS: List[str] = []

    BASE_OPTIONAL_PARAMS: List[str] = [
        'stp'
    ]
    ADDITIONAL_OPTIONAL_PARAMS: List[str] = []

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

    symbol: Symbol
    side: OrderSide
    type: OrderType
    stp: Optional[STPMode] = None
    others: dict[str, any]

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

    def _validate_params(self) -> None:
        ...


class LimitOrderTicket(BaseOrderTicket):
    type = OrderType.LIMIT

    ADDITIONAL_MANDOTORY_PARAMS = ['price', 'quantity', 'time_in_force']

    price: Decimal
    quantity: Decimal
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

    ADDITIONAL_MANDOTORY_PARAMS = ['quantity', 'quantity_type']

    quantity: Decimal
    quantity_type: MarketQuantityType


def validate_stop_price_and_trailing_delta(self) -> None:
    if self.stop_price is None and self.trailing_delta is None:
        raise ValueError('Either stop_price or trailing_delta must be set')

    if self.stop_price is not None and self.trailing_delta is not None:
        raise ValueError('Only one of stop_price or trailing_delta can be set')


PARAMS_STOP_PRICE_AND_TRAILING_DELTA = ['stop_price', 'trailing_delta']
PARAMS_ST_AND_ICEBERG_QUANTITY = [
    *PARAMS_STOP_PRICE_AND_TRAILING_DELTA,
    'iceberg_quantity'
]

class StopLossOrderTicket(BaseOrderTicket):
    type = OrderType.STOP_LOSS

    ADDITIONAL_MANDOTORY_PARAMS = ['quantity']

    quantity: Decimal

    ADDITIONAL_OPTIONAL_PARAMS = PARAMS_STOP_PRICE_AND_TRAILING_DELTA

    stop_price: Optional[Decimal] = None
    trailing_delta: Optional[Decimal] = None
    iceberg_quantity: Optional[Decimal] = None

    _validate_params = validate_stop_price_and_trailing_delta



class StopLossLimitOrderTicket(StopLossOrderTicket):
    type = OrderType.STOP_LOSS_LIMIT

    ADDITIONAL_MANDOTORY_PARAMS = ['price', 'quantity', 'time_in_force']

    price: Decimal
    time_in_force: TimeInForce

    ADDITIONAL_OPTIONAL_PARAMS = PARAMS_ST_AND_ICEBERG_QUANTITY

    iceberg_quantity: Optional[Decimal] = None


class TakeProfitOrderTicket(BaseOrderTicket):
    type = OrderType.TAKE_PROFIT

    ADDITIONAL_MANDOTORY_PARAMS = ['quantity']

    quantity: Decimal

    ADDITIONAL_OPTIONAL_PARAMS = ['stop_price', 'trailing_delta']

    stop_price: Optional[Decimal] = None
    trailing_delta: Optional[Decimal] = None

    _validate_params = validate_stop_price_and_trailing_delta


class TakeProfitLimitOrderTicket(TakeProfitOrderTicket):
    type = OrderType.TAKE_PROFIT_LIMIT

    ADDITIONAL_MANDOTORY_PARAMS = ['price', 'quantity', 'time_in_force']

    price: Decimal
    time_in_force: TimeInForce

    ADDITIONAL_OPTIONAL_PARAMS = PARAMS_ST_AND_ICEBERG_QUANTITY
    iceberg_quantity: Optional[Decimal] = None


OrderTicket = Union[
    LimitOrderTicket,
    MarketOrderTicket,
    StopLossOrderTicket,
    StopLossLimitOrderTicket,
    TakeProfitOrderTicket,
    TakeProfitLimitOrderTicket
]
