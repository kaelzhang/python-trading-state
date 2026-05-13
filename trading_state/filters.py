# Ref:
# https://developers.binance.com/docs/binance-spot-api-docs/filters

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_CEILING
from typing import (
    Optional,
    Tuple,
)

from .order_ticket import OrderTicket
from .enums import (
    FeatureType,
    MarketQuantityType,
    OrderSide,
    OrderType,
)
from .common import (
    apply_precision,
    apply_tick_size,
)
from .exceptions import FeatureNotAllowedError


FilterResult = Tuple[Optional[Exception], Optional[OrderTicket]]


class BaseFilter(ABC):
    """
    A symbol-bound constraint check.

    `apply` is intentionally side-effect-free relative to the input ticket
    (tickets are frozen). When normalization is required, return a fresh
    instance via `dataclasses.replace(ticket, field=new_value)`.
    """

    @abstractmethod
    def when(
        self,
        ticket: OrderTicket,
    ) -> bool:
        ...

    @abstractmethod
    def apply(
        self,
        ticket: OrderTicket,
        validate_only: bool,
    ) -> FilterResult:
        """
        Returns:
            (None, None)        — pass; no change needed
            (None, new_ticket)  — pass; normalized via `dataclasses.replace`
            (exc, None)         — reject
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
    name: str,
) -> ApplyRangeResult:
    if not min_value.is_zero() and target < min_value:
        return (
            ValueError(
                f'ticket.{param_name} {target} is less than the minimum '
                f'{name} {min_value}'
            ),
            None,
        )

    if not max_value.is_zero() and target > max_value:
        return (
            ValueError(
                f'ticket.{param_name} {target} is greater than the maximum '
                f'{name} {max_value}'
            ),
            None,
        )

    # No tick size restriction
    if tick_size.is_zero():
        return None, target

    new_target = apply_tick_size(target, tick_size)

    if validate_only and new_target != target:
        return (
            ValueError(
                f'ticket.{param_name} {target} does not follow the '
                f'tick size {tick_size}'
            ),
            None,
        )

    return None, new_target


@dataclass(frozen=True, slots=True)
class PrecisionFilter(BaseFilter):
    base_asset_precision: int
    quote_asset_precision: int

    def when(
        self,
        ticket: OrderTicket,
    ) -> bool:
        return True

    def apply(
        self,
        ticket: OrderTicket,
        validate_only: bool,
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

        new_quantity = apply_precision(ticket.quantity, precision)

        # Compare by full Decimal representation (digits + exponent), so
        # `Decimal('1.0') -> Decimal('1.00000000')` is still treated as a
        # normalization that must be propagated. Pure `==` would miss it
        # (Decimal equality ignores trailing-zero precision).
        if new_quantity.as_tuple() == ticket.quantity.as_tuple():
            return None, None

        # validate_only rejects only when the values themselves differ;
        # a representation-only difference (e.g. trailing zeros) is not a
        # user-visible violation, but we still normalize when allowed.
        if new_quantity != ticket.quantity and validate_only:
            return (
                ValueError(
                    f'ticket.quantity {ticket.quantity} does not follow '
                    f'the precision {precision}'
                ),
                None,
            )

        return None, replace(ticket, quantity=new_quantity)


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
        ticket: OrderTicket,
    ) -> bool:
        # Just return True,
        # it will be tested in the apply method
        return True

    def _apply_price(
        self,
        price: Decimal,
        param_name: str,
        validate_only: bool,
    ) -> ApplyRangeResult:
        return apply_range(
            target=price,
            min_value=self.min_price,
            max_value=self.max_price,
            tick_size=self.tick_size,
            validate_only=validate_only,
            param_name=param_name,
            name='price',
        )

    def apply(
        self,
        ticket: OrderTicket,
        validate_only: bool,
    ) -> FilterResult:
        updates: dict = {}

        if ticket.has(PARAM_PRICE):
            exception, new_price = self._apply_price(
                ticket.price, PARAM_PRICE, validate_only
            )
            if exception:
                return exception, None

            if new_price != ticket.price:
                updates[PARAM_PRICE] = new_price

        if ticket.has(PARAM_STOP_PRICE):
            exception, new_stop_price = self._apply_price(
                ticket.stop_price, PARAM_STOP_PRICE, validate_only
            )
            if exception:
                return exception, None

            if new_stop_price != ticket.stop_price:
                updates[PARAM_STOP_PRICE] = new_stop_price

        if not updates:
            return None, None

        return None, replace(ticket, **updates)


@dataclass(frozen=True, slots=True)
class QuantityFilter(BaseFilter):
    min_quantity: Decimal
    max_quantity: Decimal
    step_size: Decimal

    def when(
        self,
        ticket: OrderTicket,
    ) -> bool:
        return ticket.type != OrderType.MARKET

    def _apply_quantity(
        self,
        ticket: OrderTicket,
        validate_only: bool,
    ) -> ApplyRangeResult:
        return apply_range(
            target=self._get_quantity(ticket),
            min_value=self.min_quantity,
            max_value=self.max_quantity,
            tick_size=self.step_size,
            validate_only=validate_only,
            param_name='quantity',
            name='quantity',
        )

    def apply(
        self,
        ticket: OrderTicket,
        validate_only: bool,
    ) -> FilterResult:
        exception, new_quantity = self._apply_quantity(ticket, validate_only)

        if exception:
            return exception, None

        if new_quantity == ticket.quantity:
            return None, None

        return None, replace(ticket, quantity=new_quantity)

    def _get_quantity(self, ticket: OrderTicket) -> Decimal:
        return ticket.quantity


class MarketQuantityFilter(QuantityFilter):
    def when(
        self,
        ticket: OrderTicket,
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
        validate_only: bool,
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
                    f'quote order quantity for "{symbol.name}" is not allowed',
                ),
                None,
            )

        exception, new_quantity = self._apply_quantity(ticket, validate_only)

        if exception:
            return exception, None

        if ticket.quantity_type == MarketQuantityType.QUOTE:
            new_quote_quantity = new_quantity * ticket.estimated_price
            if new_quote_quantity == ticket.quantity:
                return None, None
            return None, replace(ticket, quantity=new_quote_quantity)

        if new_quantity == ticket.quantity:
            return None, None
        return None, replace(ticket, quantity=new_quantity)


PARAM_ICEBERG_QUANTITY = 'iceberg_quantity'


@dataclass(frozen=True, slots=True)
class IcebergQuantityFilter(BaseFilter):
    limit: int

    def when(
        self,
        ticket: OrderTicket,
    ) -> bool:
        return ticket.has(PARAM_ICEBERG_QUANTITY)

    def apply(
        self,
        ticket: OrderTicket,
        validate_only: bool,
    ) -> FilterResult:
        if ticket.iceberg_quantity <= 0:
            return (
                ValueError('ticket.iceberg_quantity must be greater than 0'),
                None,
            )

        parts = (
            (ticket.quantity / ticket.iceberg_quantity)
            .to_integral_value(rounding=ROUND_CEILING)
        )

        if parts > self.limit:
            if validate_only:
                return (
                    ValueError(
                        f'iceberg parts {parts} is greater than the limit '
                        f'{self.limit}'
                    ),
                    None,
                )

            min_iceberg = ticket.quantity / Decimal(self.limit)
            return None, replace(ticket, iceberg_quantity=min_iceberg)

        return None, None


PARAM_TRAILING_DELTA = 'trailing_delta'


@dataclass(frozen=True, slots=True)
class TrailingDeltaFilter(BaseFilter):
    min_trailing_above_delta: int
    max_trailing_above_delta: int
    min_trailing_below_delta: int
    max_trailing_below_delta: int

    def when(
        self,
        ticket: OrderTicket,
    ) -> bool:
        return ticket.has(PARAM_TRAILING_DELTA)

    def apply(
        self,
        ticket: OrderTicket,
        validate_only: bool,
    ) -> FilterResult:
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

        new_trailing = ticket.trailing_delta

        if new_trailing < min_delta:
            if validate_only:
                return (
                    ValueError(
                        f'ticket.trailing_delta {ticket.trailing_delta} '
                        f'is less than the minimum {min_delta}'
                    ),
                    None,
                )
            new_trailing = min_delta

        if new_trailing > max_delta:
            if validate_only:
                return (
                    ValueError(
                        f'ticket.trailing_delta {ticket.trailing_delta} '
                        f'is greater than the maximum {max_delta}'
                    ),
                    None,
                )
            new_trailing = max_delta

        if new_trailing == ticket.trailing_delta:
            return None, None

        return None, replace(ticket, trailing_delta=new_trailing)


@dataclass(frozen=True, slots=True)
class NotionalFilter(BaseFilter):
    """
    NOTIONAL filter.

    For MARKET orders, the exchange evaluates notional against the recent
    average market price (`lastPrice` over `avgPriceMins`). Locally we
    cannot reproduce that exactly without live market data, so this filter
    uses the ticket's `estimated_price` as the local approximation. Setting
    `estimated_price` close to current market price is the caller's
    responsibility; the exchange still does the authoritative check on the
    real lastPrice average. `avg_price_mins` is retained for traceability /
    future use but is not consumed by the local check.
    """

    min_notional: Decimal
    apply_min_to_market: bool
    max_notional: Decimal
    apply_max_to_market: bool
    avg_price_mins: int

    def when(
        self,
        ticket: OrderTicket,
    ) -> bool:
        return True

    def apply(
        self,
        ticket: OrderTicket,
        validate_only: bool,
    ) -> FilterResult:
        market_order = ticket.is_a(OrderType.MARKET)

        if (
            market_order
            and not self.apply_min_to_market
            and not self.apply_max_to_market
        ):
            return None, None

        min_notional = self.min_notional
        max_notional = self.max_notional

        if market_order:
            price = ticket.estimated_price

            if not self.apply_min_to_market:
                min_notional = Decimal('0')
            if not self.apply_max_to_market:
                max_notional = Decimal('Infinity')
        else:
            price = ticket.price

        if (
            market_order
            and ticket.quantity_type == MarketQuantityType.QUOTE
        ):
            notional = ticket.quantity
        else:
            notional = price * ticket.quantity

        if notional < min_notional:
            # In this situation, we should not fix the order ticket,
            # or there might be severe side effects.
            return (
                ValueError(
                    f'ticket notional {notional} is less than the minimum '
                    f'{min_notional}'
                ),
                None,
            )

        if notional > max_notional:
            # Similar to the above
            return (
                ValueError(
                    f'ticket notional {notional} is greater than the maximum '
                    f'{max_notional}'
                ),
                None,
            )

        return None, None
