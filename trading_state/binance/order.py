
from trading_state import (
    OrderTicket,
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
