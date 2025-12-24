
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
from trading_state.common import (
    DECIMAL_ZERO,
    timestamp_to_datetime
)


# Ref:
# https://developers.binance.com/docs/binance-spot-api-docs/rest-api/trading-endpoints#new-order-trade
def encode_order_request(ticket: OrderTicket) -> dict:
    """
    Encode order request to Binance API request payload format

    Args:
        ticket (OrderTicket): the order ticket to encode

    Returns:
        dict: the encoded order request payload
    """

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

    # Get the full response of the order creation
    kwargs['newOrderRespType'] = 'FULL'

    return kwargs


def _decode_order_status(status_str: str) -> OrderStatus:
    """Decode order status from Binance order status string
    """

    match status_str:
        case 'FILLED':
            return OrderStatus.FILLED
        case 'CANCELED':
            return OrderStatus.CANCELLED
        case 'PARTIALLY_FILLED':
            return OrderStatus.CREATED
        case 'NEW':
            return OrderStatus.CREATED
        case 'CANCELED':
            return OrderStatus.CANCELLED
        case 'EXPIRED':
            return OrderStatus.CANCELLED


# Ref:
# https://github.com/binance/binance-spot-api-docs/blob/master/user-data-stream.md#order-create
def decode_order_create_response(response: dict) -> Tuple[str, dict]:
    """Generate kwargs used by state.update_order() from Binance order create response
    """

    updates = dict(
        status=_decode_order_status(response['status']),
        id=response['clientOrderId'],
        created_at=datetime.fromtimestamp(response['transactTime'] / 1000),
        filled_quantity=Decimal(response['executedQty']),
        quote_quantity=Decimal(response['cummulativeQuoteQty']),
    )

    fills = response['fills']
    if fills:
        commission_quantity = DECIMAL_ZERO
        for fill in fills:
            commission_quantity += Decimal(fill['commission'])

        updates['commission_asset'] = fills[0]['commissionAsset']
        updates['commission_quantity'] = commission_quantity

    return updates

# Ref
# https://github.com/binance/binance-spot-api-docs/blob/master/user-data-stream.md#order-update
def decode_order_update_event(
    payload: dict
) -> Tuple[str, dict]:
    """Generate order updates dict from Binance order update payload

    Args:
        payload (dict): the payload of the event

    Returns:
        Tuple:
        - str: client order id
        - dict: the order updates kwargs for state.update_order()
    """

    # Current order status
    order_status = payload['X']
    client_order_id = payload['c']
    filled_quantity = Decimal(payload['z'])
    quote_quantity = Decimal(payload['Z'])
    commission_asset = payload['N'] or None
    commission_quantity = Decimal(payload['n'])
    updated_at = timestamp_to_datetime(payload['T'])
    # event_time = timestamp_to_datetime(payload['E'])

    update_kwargs = {
        'filled_quantity': filled_quantity,
        'quote_quantity': quote_quantity,
        'updated_at': updated_at,
        'commission_asset': commission_asset,
        'commission_quantity': commission_quantity
    }

    if order_status == 'CANCELED':
        update_kwargs['status'] = OrderStatus.CANCELLED
    elif order_status == 'FILLED':
        update_kwargs['status'] = OrderStatus.FILLED

    return client_order_id, update_kwargs
