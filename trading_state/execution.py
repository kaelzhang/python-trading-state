from __future__ import annotations

from typing import (
    Optional,
    Protocol,
    Callable,
    TYPE_CHECKING,
    runtime_checkable
)
from decimal import Decimal

from .common import DECIMAL_ZERO
from .enums import (
    MarketQuantityType,
    OrderSide,
    TimeInForce
)
from .order_ticket import (
    LimitOrderTicket,
    MarketOrderTicket,
    OrderTicket
)

if TYPE_CHECKING:
    from .order import Order
    from .symbol import Symbol
    from .target import PositionTarget


@runtime_checkable
class ExecutionStrategy(Protocol):
    """
    Strategy interface used by TradingState to create tickets for a
    position target, and to receive order-tracking callbacks.
    """

    def create_ticket(
        self,
        symbol: 'Symbol',
        quantity: Decimal,
        target: 'PositionTarget',
        side: OrderSide
    ) -> Optional[OrderTicket]:
        """
        Create and return an order ticket.

        Returns:
            - OrderTicket: create order with this ticket
            - None: skip this execution tick
        """

    def track_order(
        self,
        order: 'Order'
    ) -> None:
        """
        Called when an order is completed and tracked by TradingState.
        """


ExecutionStrategyResolver = Callable[
    ['PositionTarget'],
    Optional[ExecutionStrategy]
]


class BaseExecutionStrategy:
    def create_ticket(
        self,
        symbol: 'Symbol',
        quantity: Decimal,
        target: 'PositionTarget',
        side: OrderSide
    ) -> Optional[OrderTicket]:
        raise NotImplementedError(
            'Execution strategy must implement `create_ticket()`'
        )

    def track_order(self, order: 'Order') -> None:
        # Optional hook for custom strategies.
        return


class DefaultExecutionStrategy(BaseExecutionStrategy):
    """
    Keep the previous built-in behavior:
    - urgent=True  => market order using quote quantity
    - urgent=False => limit order using base quantity
    """

    def create_ticket(
        self,
        symbol: 'Symbol',
        quantity: Decimal,
        target: 'PositionTarget',
        side: OrderSide
    ) -> OrderTicket:
        price = target.price

        if target.urgent:
            quote_quantity = quantity * price

            return MarketOrderTicket(
                symbol=symbol,
                side=side,
                quantity=quote_quantity,
                # Use quote quantity 'quoteOrderQty' for market order
                # to avoid -2010 error as much as possible.
                quantity_type=MarketQuantityType.QUOTE,
                estimated_price=price
            )

        return LimitOrderTicket(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            time_in_force=TimeInForce.GTC
        )


def consumed_base_quantity(ticket: OrderTicket) -> Decimal:
    """
    Returns the consumed base quantity represented by a ticket.
    """

    if (
        isinstance(ticket, MarketOrderTicket)
        and ticket.quantity_type is MarketQuantityType.QUOTE
    ):
        estimated_price = ticket.estimated_price
        if estimated_price.is_zero():
            return DECIMAL_ZERO

        return ticket.quantity / estimated_price

    return ticket.quantity
