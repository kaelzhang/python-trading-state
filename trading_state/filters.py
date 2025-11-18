from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import (
    Optional,
    Tuple
)

from .order import OrderTicket
# from .enums import OrderType
from .util import (
    apply_precision, apply_tick_size
)


FilterResult = Tuple[Optional[Exception], bool]


class BaseFilter(ABC):
    @abstractmethod
    def when(
        self,
        ticket: OrderTicket
    ) -> bool:
        ...

    @abstractmethod
    def apply(
        self,
        ticket: OrderTicket,
        validate_only: bool
    ) -> FilterResult:
        ...


PARAM_PRICE = 'price'
PARAM_STOP_PRICE = 'stop_price'


@dataclass
class PriceFilter(BaseFilter):
    min_price: Decimal
    max_price: Decimal
    tick_size: Decimal

    def when(
        self,
        ticket: OrderTicket
    ) -> bool:
        # Just return True,
        # it will be tested in the apply method
        return True

    def _apply_price(
        self,
        price: Decimal,
        param_name: str,
        validate_only: bool
    ) -> Tuple[None, Decimal] | Tuple[Exception, None]:
        if not self.min_price.is_zero() and price < self.min_price:
            return (
                ValueError(f'ticket.{param_name} {price} is less than the minimum price {self.min_price}'),
                None
            )

        if not self.max_price.is_zero() and price > self.max_price:
            return (
                ValueError(f'ticket.{param_name} {price} is greater than the maximum price {self.max_price}'),
                None
            )

        if self.tick_size.is_zero():
            return None, price

        new_price = apply_tick_size(price, self.tick_size)

        if validate_only and new_price != price:
            return (
                ValueError(f'ticket.{param_name} {price} does not follow the tick size {self.tick_size}'),
                None
            )

        return None, new_price

    def apply(
        self,
        ticket: OrderTicket,
        validate_only: bool
    ) -> FilterResult:
        modified = False

        if ticket.has(PARAM_PRICE):
            exception, new_price = self._apply_price(ticket.price, PARAM_PRICE)
            if exception:
                return exception, False

            if new_price != ticket.price:
                modified = True
                ticket.price = new_price

        if ticket.has(PARAM_STOP_PRICE):
            exception, new_stop_price = self._apply_price(ticket.stop_price, PARAM_STOP_PRICE)
            if exception:
                return exception, False

            if new_stop_price != ticket.stop_price:
                modified = True
                ticket.stop_price = new_stop_price

        return None, modified


@dataclass
class QuantityFilter(BaseFilter):
    min_quantity: Decimal
    max_quantity: Decimal
    step_size: Decimal

    def when(
        self,
        ticket: OrderTicket
    ) -> bool:
        return True
