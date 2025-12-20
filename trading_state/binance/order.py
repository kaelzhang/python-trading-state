
from typing import (
    Tuple
)
from decimal import Decimal
from datetime import datetime

from trading_state import (
    OrderTicket,
    OrderStatus,
    LimitOrderTicket,
    MarketOrderTicket,
    MarketQuantityType
)


def to_order_request(ticket: OrderTicket) -> dict:
    match ticket:
        case LimitOrderTicket():
            kwargs = dict(
                symbol=ticket.symbol.name,
                side=ticket.side,
                type=ticket.type,
                timeInForce=ticket.time_in_force,
                quantity=ticket.quantity,
                price=str(ticket.price)
            )
        case MarketOrderTicket():
            kwargs = dict(
                symbol=ticket.symbol.name,
                side=ticket.side,
                type=ticket.type
            )

            if ticket.quantity_type == MarketQuantityType.BASE:
                kwargs['quantity'] = ticket.quantity
            else:
                kwargs['quoteOrderQty'] = ticket.quantity
        case _:
            # TODO:
            # support other order ticket types
            raise ValueError(f'Unsupported order ticket: {ticket}')

    return kwargs


# Ref
# https://github.com/binance/binance-spot-api-docs/blob/master/user-data-stream.md#order-update
def generate_order_updates(payload: dict) -> Tuple[str, dict]:
    """Generate order updates dict from Binance order update payload

    Args:
        payload (dict): the payload of the event

    Returns:
        Tuple:
        - str: client order id
        - dict: the order updates kwargs for order.update()
    """

    # Current order status
    order_status = payload['X']
    client_order_id = payload['c']
    filled_quantity = Decimal(payload['z'])
    quote_quantity = Decimal(payload['Z'])
    updated_at = datetime.fromtimestamp(payload['T'] / 1000)

    # order = self._state.get_order_by_id(client_order_id)

    # if order is None:
    #     return

    update_kwargs = {
        'filled_quantity': filled_quantity,
        'quote_quantity': quote_quantity,
        'updated_at': updated_at
    }

    if order_status == 'CANCELED':
        update_kwargs['status'] = OrderStatus.CANCELLED
    elif order_status == 'FILLED':
        update_kwargs['status'] = OrderStatus.FILLED

    # order.update(**update_kwargs)
    return client_order_id, update_kwargs
