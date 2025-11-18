# Ref
# https://developers.binance.com/docs/binance-spot-api-docs/rest-api/trading-endpoints

# from dataclasses import dataclass
from typing import (
    List, Union
)

from decimal import Decimal

from .symbol import Symbol
from .enums import (
    OrderSide, OrderType, TimeInForce, MarketQuantityType
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

    symbol: Symbol
    side: OrderSide
    type: OrderType
    stp: str
    others: dict[str, any]

    def __init__(
        self,
        **kwargs
    ) -> None:
        required_params = (
            self.BASE_MANDOTORY_PARAMS + self.ADDITIONAL_MANDOTORY_PARAMS
        )
        optional_params = (
            self.BASE_OPTIONAL_PARAMS + self.ADDITIONAL_OPTIONAL_PARAMS
        )

        for param in required_params:
            if param not in kwargs:
                raise ValueError(f'"{param}" is a required parameter for Order')

            setattr(self, param, kwargs[param])
            del kwargs[param]

        for param in optional_params:
            if param in kwargs:
                setattr(self, param, kwargs[param])
                del kwargs[param]
            else:
                # Make sure the parameter is set
                setattr(self, param, None)

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

    ADDITIONAL_OPTIONAL_PARAMS = ['post_only']

    post_only: bool


class MarketOrderTicket(BaseOrderTicket):
    type = OrderType.MARKET

    ADDITIONAL_MANDOTORY_PARAMS = ['quantity', 'quantity_type']

    quantity: Decimal
    quantity_type: MarketQuantityType


def stop_price_or_trailing_delta(self) -> None:
    if self.stop_price is None and self.trailing_delta is None:
        raise ValueError('Either stop_price or trailing_delta must be set')

    if self.stop_price is not None and self.trailing_delta is not None:
        raise ValueError('Only one of stop_price or trailing_delta can be set')


class StopLossOrderTicket(BaseOrderTicket):
    type = OrderType.STOP_LOSS

    ADDITIONAL_MANDOTORY_PARAMS = ['quantity']

    quantity: Decimal

    ADDITIONAL_OPTIONAL_PARAMS = ['stop_price','trailing_delta']

    stop_price: Decimal
    trailing_delta: Decimal

    _validate_params = stop_price_or_trailing_delta



class StopLossLimitOrderTicket(StopLossOrderTicket):
    type = OrderType.STOP_LOSS_LIMIT

    ADDITIONAL_MANDOTORY_PARAMS = ['price', 'quantity', 'time_in_force']

    price: Decimal
    time_in_force: TimeInForce


class TakeProfitOrderTicket(BaseOrderTicket):
    type = OrderType.TAKE_PROFIT

    ADDITIONAL_MANDOTORY_PARAMS = ['quantity']

    quantity: Decimal

    ADDITIONAL_OPTIONAL_PARAMS = ['stop_price', 'trailing_delta']

    stop_price: Decimal
    trailing_delta: Decimal

    _validate_params = stop_price_or_trailing_delta


class TakeProfitLimitOrderTicket(TakeProfitOrderTicket):
    type = OrderType.TAKE_PROFIT_LIMIT

    ADDITIONAL_MANDOTORY_PARAMS = ['price', 'quantity', 'time_in_force']

    price: Decimal
    time_in_force: TimeInForce


Order = Union[
    LimitOrderTicket,
    MarketOrderTicket,
    StopLossOrderTicket,
    StopLossLimitOrderTicket,
    TakeProfitOrderTicket,
    TakeProfitLimitOrderTicket
]
