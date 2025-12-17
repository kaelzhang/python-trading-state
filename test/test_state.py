from decimal import Decimal

from trading_state import (
    OrderSide,
    MarketQuantityType,
    TimeInForce,
    OrderStatus,
    OrderType,
    Balance,
    PositionTargetStatus
)

from .fixtures import (
    init_state,
    BTCUSDC,
    BTCUSDT,
    BTC,
    # USDT
)


def test_trading_state():
    state = init_state()

    active_value = state.get_account_value()
    assert active_value == Decimal('410000')

    assert state.support_symbol(BTCUSDC)
    assert state.exposure(BTC) == (None, Decimal('0.1'))

    exception, updated = state.expect(
        BTCUSDC,
        exposure=Decimal('0.2'),
        price=Decimal('10000'),
        use_market_order=False
    )
    assert exception is None
    assert updated

    assert state.exposure(BTC) == (None, Decimal('0.2'))

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
    exception, updated = state.expect(
        BTCUSDC,
        exposure=Decimal('0.3'),
        price=Decimal('10000'),
        use_market_order=True
    )
    assert exception is None
    assert updated

    assert state.exposure(BTC) == (None, Decimal('0.3'))

    # Even we set a new expectation with another symbol,
    # but the previous expectation is equivalent,
    # it will be skipped
    exception, updated = state.expect(
        BTCUSDT,
        exposure=Decimal('0.3'),
        price=Decimal('10000'),
        use_market_order=True
    )
    assert exception is None
    assert not updated

    orders, orders_to_cancel = state.get_orders()

    assert len(orders_to_cancel) == 1
    assert len(orders) == 1

    # Just a market order
    order = next(iter(orders))
    assert order.status == OrderStatus.SUBMITTING
    assert order.id is None
    assert order.filled_quantity == Decimal('0')

    ticket = order.ticket
    assert ticket.type == OrderType.MARKET
    assert ticket.symbol.name == BTCUSDC
    assert ticket.side == OrderSide.BUY
    assert ticket.quantity == Decimal('20000')
    assert ticket.estimated_price == Decimal('10000')
    assert ticket.quantity_type == MarketQuantityType.QUOTE
    assert not ticket.has('price')
    assert not ticket.has('time_in_force')

    # The order that created for position 0.2 should be cancelled
    order_to_cancel = next(iter(orders_to_cancel))
    ticket = order_to_cancel.ticket
    assert order_to_cancel.status == OrderStatus.CANCELLING
    assert ticket.quantity == Decimal('1')

    # If we get orders again, there will be no orders to perform
    orders, orders_to_cancel = state.get_orders()
    assert not orders
    assert not orders_to_cancel

    # We set the order status to ABOUT_TO_CANCEL,
    # which usually is triggered by the trader
    # if the order is failed to be canceled from the exchange
    order_to_cancel.update(
        status = OrderStatus.ABOUT_TO_CANCEL
    )
    orders, orders_to_cancel = state.get_orders()

    assert len(orders_to_cancel) == 1
    assert next(iter(orders_to_cancel)) == order_to_cancel

    # We just cancel the market order
    state.cancel_order(order)

    # We could cancel an order which is already canceled
    state.cancel_order(order_to_cancel)

    orders, orders_to_cancel = state.get_orders()
    assert not orders
    assert len(orders_to_cancel) == 1
    assert order is next(iter(orders_to_cancel))

    order.update(
        status = OrderStatus.CANCELLED
    )

    # We should also remove the expectation for the asset
    # to avoid unexpected behavior
    assert BTC not in state._expected


def test_order_filled():
    state = init_state()

    exception, updated = state.expect(
        BTCUSDC,
        exposure=Decimal('0.2'),
        price=Decimal('10000'),
        use_market_order=False
    )
    assert exception is None
    assert updated

    # Same expectation, no need to update
    exception, updated = state.expect(
        BTCUSDC,
        exposure=Decimal('0.2'),
        price=Decimal('10000'),
        use_market_order=False
    )
    assert exception is None
    assert not updated

    orders, _ = state.get_orders()

    order = next(iter(orders))

    order_str = repr(order)

    assert 'side=BUY' in order_str
    assert 'status=SUBMITTING' in order_str
    assert 'quantity=1.00000000' in order_str

    order.update(
        status = OrderStatus.CREATED,
        order_id = 'order-1',
        filled_quantity = Decimal('0.5')
    )

    # Imitate the balance is increased
    state.set_balances([
        Balance(BTC, Decimal('1.5'), Decimal('0'))
    ])

    order.update(
        status = OrderStatus.FILLED
    )

    # The order is filled, so the expectation should marked as achieved,
    # but the balance might not be updated yet,
    # we should keep that expectation
    assert state._expected[BTC].status is PositionTargetStatus.ACHIEVED

    assert state.exposure(BTC) == (None, Decimal('0.2'))

    orders, orders_to_cancel = state.get_orders()
    assert not orders
    assert not orders_to_cancel

    # Although the balance is updated,
    # but the balance is not changed,
    state.set_balances([
        Balance(BTC, Decimal('1.5'), Decimal('0'))
    ])

    state.set_balances([
        Balance(BTC, Decimal('2'), Decimal('0'))
    ])

    # The balance is updated,
    # but the intrinsic position is equal to the expectation,
    # we keep the expectation to improve performance
    assert state._expected[BTC].status is PositionTargetStatus.ACHIEVED

    assert state.exposure(BTC) == (None, Decimal('0.2'))

    # The expectation is equivalent to the current position,
    # no need to update
    exception, updated = state.expect(
        BTCUSDC,
        exposure=Decimal('0.2'),
        price=Decimal('10000'),
        use_market_order=False
    )
    assert exception is None
    assert not updated

    state.set_balances([
        Balance(BTC, Decimal('3'), Decimal('0'))
    ])

    # The balance is updated,
    # but the intrinsic position is not equal to the expectation,
    # we should remove the expectation
    assert BTC not in state._expected

    assert state.exposure(BTC) == (None, Decimal('0.3'))

    # If we freeze the asset, the exposure will be 0
    state.freeze(BTC, Decimal('3'))
    assert state.exposure(BTC) == (None, Decimal('0'))

    # If we unfreeze the asset, the exposure will be the expected value
    state.freeze(BTC, None)
    assert state.exposure(BTC) == (None, Decimal('0.3'))

    # The expectation is already achieved based on calculation
    exception, updated = state.expect(
        BTCUSDC,
        exposure=Decimal('0.3'),
        price=Decimal('10000'),
        use_market_order=False
    )
    assert exception is None
    assert not updated


def test_alt_currencies():
    ...
