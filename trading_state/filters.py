# Ref:
# https://developers.binance.com/docs/binance-spot-api-docs/filters

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import (
    Optional,
    Tuple,
    # Callable
)

from .order_ticket import OrderTicket
from .enums import (
    OrderType, OrderSide, MarketQuantityType, FeatureType
)
from .common import (
    apply_precision,
    apply_tick_size
)
from .exceptions import FeatureNotAllowedError


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
        validate_only: bool,
        **kwargs
    ) -> FilterResult:
        """
        Returns:
            Tuple[Optional[Exception], bool]:
            - the reason exception if the filter is not successfully applied
            - whether the order ticket is modified
        """

        ...


PARAM_PRICE = 'price'
PARAM_STOP_PRICE = 'stop_price'

ApplyRangeResult = Tuple[None, Decimal] | Tuple[Exception, None]

def apply_range(
    target: Decimal,
    min_value: Decimal,
    max_value: Decimal,
    tick_size: Decimal,
    validate_only: bool,
    param_name: str,
    name: str
) -> ApplyRangeResult:
    if not min_value.is_zero() and target < min_value:
        return (
            ValueError(f'ticket.{param_name} {target} is less than the minimum {name} {min_value}'),
            None
        )

    if not max_value.is_zero() and target > max_value:
        return (
            ValueError(f'ticket.{param_name} {target} is greater than the maximum {name} {max_value}'),
            None
        )

    # No tick size restriction
    if tick_size.is_zero():
        return None, target

    new_target = apply_tick_size(target, tick_size)

    if validate_only and new_target != target:
        return (
            ValueError(f'ticket.{param_name} {target} does not follow the tick size {tick_size}'),
            None
        )

    return None, new_target


@dataclass(frozen=True, slots=True)
class PrecisionFilter(BaseFilter):
    base_asset_precision: int
    quote_asset_precision: int

    def when(
        self,
        ticket: OrderTicket
    ) -> bool:
        return True

    def apply(
        self,
        ticket: OrderTicket,
        validate_only: bool,
        **kwargs
    ) -> FilterResult:
        is_market_order = ticket.is_a(OrderType.MARKET)

        precision = (
            self.base_asset_precision
            if (
                (
                    not is_market_order
                    and ticket.side == OrderSide.SELL
                )
                or (
                    is_market_order
                    and ticket.quantity_type == MarketQuantityType.QUOTE
                )
            )
            else self.quote_asset_precision
        )

        quantity = apply_precision(ticket.quantity, precision)
        modified = quantity != ticket.quantity

        if modified and validate_only:
            return (
                ValueError(f'ticket.quantity {ticket.quantity} does not follow the precision {precision}'),
                False
            )

        ticket.quantity = quantity

        return None, modified


# TODO:
# @dataclass
# class FeatureGateFilter(BaseFilter):
#     iceberg: bool
#     oco: bool
#     oto: bool
#     trailing_stop: bool


@dataclass(frozen=True, slots=True)
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
    ) -> ApplyRangeResult:
        return apply_range(
            target=price,
            min_value=self.min_price,
            max_value=self.max_price,
            tick_size=self.tick_size,
            validate_only=validate_only,
            param_name=param_name,
            name='price'
        )

    def apply(
        self,
        ticket: OrderTicket,
        validate_only: bool,
        **kwargs
    ) -> FilterResult:
        modified = False

        if ticket.has(PARAM_PRICE):
            exception, new_price = self._apply_price(
                ticket.price, PARAM_PRICE, validate_only
            )
            if exception:
                return exception, False

            if new_price != ticket.price:
                modified = True
                ticket.price = new_price

        if ticket.has(PARAM_STOP_PRICE):
            exception, new_stop_price = self._apply_price(
                ticket.stop_price, PARAM_STOP_PRICE
            )
            if exception:
                return exception, False

            if new_stop_price != ticket.stop_price:
                modified = True
                ticket.stop_price = new_stop_price

        return None, modified


@dataclass(frozen=True, slots=True)
class QuantityFilter(BaseFilter):
    min_quantity: Decimal
    max_quantity: Decimal
    step_size: Decimal

    def when(
        self,
        ticket: OrderTicket
    ) -> bool:
        return ticket.type != OrderType.MARKET

    def apply(
        self,
        ticket: OrderTicket,
        validate_only: bool,
        **kwargs
    ) -> FilterResult:
        exception, new_quantity = apply_range(
            target=self._get_quantity(ticket),
            min_value=self.min_quantity,
            max_value=self.max_quantity,
            tick_size=self.step_size,
            validate_only=validate_only,
            param_name='quantity',
            name='quantity'
        )

        if exception:
            return exception, False

        return None, new_quantity != ticket.quantity

    def _get_quantity(self, ticket: OrderTicket) -> Decimal:
        return ticket.quantity


class MarketQuantityFilter(QuantityFilter):
    def when(
        self,
        ticket: OrderTicket
    ) -> bool:
        return ticket.type == OrderType.MARKET

    def _get_quantity(self, ticket: OrderTicket) -> Decimal:
        return (
            ticket.quantity / ticket.estimated_price
            if ticket.quantity_type == MarketQuantityType.QUOTE
            else ticket.quantity
        )

    def apply(
        self,
        ticket: OrderTicket,
        *args, **kwargs
    ) -> FilterResult:
        if (
            ticket.quantity_type == MarketQuantityType.QUOTE
            and not (symbol := ticket.symbol).support(
                (feature := FeatureType.QUOTE_ORDER_QUANTITY)
            )
        ):
            return (
                FeatureNotAllowedError(
                    symbol,
                    feature,
                    f'quote order quantity for "{symbol.name}" is not allowed'
                ),
                False
            )

        return super().apply(ticket, *args, **kwargs)


PARAM_ICEBERG_QUANTITY = 'iceberg_quantity'

@dataclass(frozen=True, slots=True)
class IcebergQuantityFilter(BaseFilter):
    limit: int

    def when(
        self,
        ticket: OrderTicket
    ) -> bool:
        return ticket.has(PARAM_ICEBERG_QUANTITY)

    def apply(
        self,
        ticket: OrderTicket,
        validate_only: bool,
        **kwargs
    ) -> FilterResult:
        if ticket.iceberg_quantity > self.limit:
            if validate_only:
                return (
                    ValueError(f'ticket.iceberg_quantity {ticket.iceberg_quantity} is greater than the limit {self.limit}'),
                    False
                )

            ticket.iceberg_quantity = self.limit
            return None, True

        return None, False


PARAM_TRAILING_DELTA = 'trailing_delta'

@dataclass(frozen=True, slots=True)
class TrailingDeltaFilter(BaseFilter):
    min_trailing_above_delta: int
    max_trailing_above_delta: int
    min_trailing_below_delta: int
    max_trailing_below_delta: int

    def when(
        self,
        ticket: OrderTicket
    ) -> bool:
        return ticket.has(PARAM_TRAILING_DELTA)

    def apply(
        self,
        ticket: OrderTicket,
        validate_only: bool,
        **kwargs
    ) -> FilterResult:
        modified = False

        if (
            ticket.is_a(OrderType.STOP_LOSS, side=OrderSide.BUY)
            or ticket.is_a(OrderType.STOP_LOSS_LIMIT, side=OrderSide.BUY)
            or ticket.is_a(OrderType.TAKE_PROFIT, side=OrderSide.SELL)
            or ticket.is_a(OrderType.TAKE_PROFIT_LIMIT, side=OrderSide.SELL)
        ):
            min_delta = self.min_trailing_above_delta
            max_delta = self.max_trailing_above_delta
        else:
            min_delta = self.min_trailing_below_delta
            max_delta = self.max_trailing_below_delta

        if ticket.trailing_delta < min_delta:
            if validate_only:
                return (
                    ValueError(f'ticket.trailing_delta {ticket.trailing_delta} is less than the minimum {min_delta}'),
                    False
                )

            modified = True
            ticket.trailing_delta = min_delta

        if ticket.trailing_delta > max_delta:
            if validate_only:
                return (
                    ValueError(f'ticket.trailing_delta {ticket.trailing_delta} is greater than the maximum {max_delta}'),
                    False
                )

            modified = True
            ticket.trailing_delta = max_delta

        return None, modified


@dataclass(frozen=True, slots=True)
class NotionalFilter(BaseFilter):
    min_notional: Decimal
    apply_min_to_market: bool
    max_notional: Decimal
    apply_max_to_market: bool
    avg_price_mins: int

    def when(
        self,
        ticket: OrderTicket
    ) -> bool:
        return True

    def apply(
        self,
        ticket: OrderTicket,
        validate_only: bool,
        **kwargs
    ) -> FilterResult:
        # TODO: other market order types
        market_order = ticket.is_a(OrderType.MARKET)

        if (
            market_order
            and not self.apply_min_to_market
            and not self.apply_max_to_market
        ):
            return None, False

        min_notional = self.min_notional
        max_notional = self.max_notional

        if market_order:
            price: Decimal = kwargs['get_avg_price'](
                ticket.symbol.name,
                self.avg_price_mins
            )

            if not self.apply_min_to_market:
                min_notional = Decimal('0')
            if not self.apply_max_to_market:
                max_notional = Decimal('Infinity')
        else:
            price = ticket.price

        notional = price * ticket.quantity

        if notional < min_notional:
            # In this situation, we should not fix the order ticket,
            # or there might be severe side effects.
            return (
                ValueError(f'ticket notional {notional} is less than the minimum {min_notional}'),
                False
            )

        if notional > max_notional:
            # Similar to the above
            return (
                ValueError(f'ticket notional {notional} is greater than the maximum {max_notional}'),
                False
            )

        return None, False
