from decimal import Decimal

from trading_state import (
    OrderSide,
    # MarketQuantityType,
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

    assert state.support_symbol(BTCUSDC)
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

    orders, orders_to_cancel = state.get_orders()
    assert not orders
    assert not orders_to_cancel

    # Expect a new position
    assert state.expect(
        BTCUSDC,
        position=0.3,
        # Although the price is provided, it will be ignored
        price=Decimal('10000'),
        asap=True
    ) is None

    assert state.position(BTC) == (None, 0.3)

    # Even we set a new expectation with another symbol,
    # but the previous expectation is equivalent,
    # it will be skipped
    assert state.expect(
        BTCUSDT,
        position=0.3,
        price=Decimal('20000'),
        asap=True
    ) is None

    orders, orders_to_cancel = state.get_orders()

    assert len(orders_to_cancel) == 1
    assert len(orders) == 1

    order = next(iter(orders))
    assert order.status == OrderStatus.SUBMITTING
    assert order.id is None
    assert order.filled_quantity == Decimal('0')

    ticket = order.ticket
    assert ticket.type == OrderType.MARKET
    assert ticket.symbol.name == BTCUSDC
    assert ticket.side == OrderSide.BUY
    assert ticket.quantity == Decimal('2')
    assert not ticket.has('price')
    assert not ticket.has('time_in_force')

    order_to_cancel = next(iter(orders_to_cancel))
    ticket = order_to_cancel.ticket
    assert order_to_cancel.status == OrderStatus.CANCELLING
    assert ticket.quantity == Decimal('1')

    orders, orders_to_cancel = state.get_orders()
    assert not orders
    assert not orders_to_cancel

    order_to_cancel.status = OrderStatus.ABOUT_TO_CANCEL
    orders, orders_to_cancel = state.get_orders()

    assert len(orders_to_cancel) == 1
    assert next(iter(orders_to_cancel)) == order_to_cancel

    state.cancel_order(order)
    state.cancel_order(order_to_cancel)

    orders, orders_to_cancel = state.get_orders()
    assert not orders
    assert len(orders_to_cancel) == 1
    assert order is next(iter(orders_to_cancel))
