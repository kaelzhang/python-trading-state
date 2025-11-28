from decimal import Decimal

from trading_state import (
    OrderSide,
    MarketQuantityType,
    TimeInForce,
    OrderStatus,
    OrderType,
)

from .fixtures import (
    init_state,
    BTCUSDC,
    BTCUSDT,
    BTC,
    USDT
)


def test_trading_state():
    state = init_state()

    assert state.position(BTC) == (None, 0.1)

    assert state.expect(
        BTCUSDC,
        position=0.2,
        price=Decimal('10000'),
        asap=False
    ) is None

    assert state.position(BTC) == (None, 0.2)

    orders, orders_to_cancel = state.get_orders()

    assert not orders_to_cancel
    assert len(orders) == 1

    order = next(iter(orders))

    assert order.status == OrderStatus.SUBMITTING
    assert order.id is None
    assert order.filled_quantity == Decimal('0')

    ticket = order.ticket
    assert ticket.type == OrderType.LIMIT
    assert ticket.symbol.name == BTCUSDC
    assert ticket.side == OrderSide.BUY
    assert ticket.quantity == Decimal('1')
    assert ticket.price == Decimal('10000')
    assert ticket.time_in_force == TimeInForce.GTC


