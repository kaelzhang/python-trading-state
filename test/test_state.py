from decimal import Decimal
from datetime import datetime, timedelta

from trading_state import (
    OrderSide,
    MarketQuantityType,
    TimeInForce,
    OrderStatus,
    OrderType,
    Balance,
    PositionTargetStatus,
    TradingStateEvent,
)
from trading_state.common import (
    DECIMAL_ZERO,
    DECIMAL_ONE
)

from .fixtures import (
    init_state,
    BTCUSDC,
    USDC,
    BTCUSDT,
    BTC,
    USDT,
    ZUSDT,
    Z
)


BTCUSDC = BTCUSDC.name
BTCUSDT = BTCUSDT.name


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
    state.update_order(
        order_to_cancel,
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

    state.update_order(order, status=OrderStatus.CANCELLED)

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

    state.update_order(
        order,
        status = OrderStatus.CREATED,
        id = 'order-1',
        filled_quantity = Decimal('0.5'),
        quote_quantity = Decimal('5000')
    )

    # Imitate the balance is increased
    state.set_balances([
        Balance(BTC, Decimal('1.5'), Decimal('0'))
    ])

    state.update_order(
        order,
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


def test_no_same_exposure():
    state = init_state()
    state.expect(
        BTCUSDT,
        exposure=Decimal('0.2'),
        price=Decimal('10000'),
        use_market_order=False
    )

    state.set_balances([
        Balance(BTC, Decimal('1'), DECIMAL_ZERO)
    ], delta=True)

    assert BTC in state._expected

    orders, _ = state.get_orders()
    assert not orders

    assert BTC not in state._expected


def test_alt_currencies():
    state = init_state()

    # This will consume USDC and convert to USDT
    state.set_alt_currency_weights(
        (
            (Decimal('1'),),
            (Decimal('0'),)
        )
    )

    # It will create 2 orders, allocated into BTCUSDT and BTCUSDC
    state.expect(
        BTCUSDT,
        exposure=Decimal('0.3'),
        price=Decimal('10000'),
        use_market_order=False
    )

    orders, orders_to_cancel = state.get_orders()

    assert not orders_to_cancel
    order1, order2 = orders

    # with pytest.raises(ValueError, match='order id is required'):
    #     state.update_order(order1, status=OrderStatus.CREATED)

    now = datetime.now()

    state.update_order(
        order1,
        status=OrderStatus.CREATED,
        id='order-1',
        created_at=now
    )

    state.update_order(
        order2,
        status=OrderStatus.CREATED,
        id='order-2',
        created_at=now
    )

    assert order1.created_at == now
    assert order1.updated_at == now

    now += timedelta(seconds=1)
    state.update_order(
        order1,
        updated_at=now
    )
    assert order1.updated_at == now

    for order in state.query_orders():
        assert order.ticket.quantity == Decimal('1')

    assert len(list(state.query_orders(limit=1))) == 1

    assert state.get_order_by_id('order-1') is order1

    result = list(state.query_orders(
        not_exists=True
    ))

    assert len(result) == 0

    result = list(state.query_orders(
        id='order-1'
    ))
    assert result[0] is order1

    def is_order_1(order_id: str, key: str) -> bool:
        assert key == 'id'
        return order_id == 'order-1'

    result = list(state.query_orders(
        id=is_order_1
    ))
    assert result[0] is order1

    result = list(state.query_orders(
        ticket={
            'quantity': Decimal('1')
        }
    ))
    assert len(result) == 2

    assert state.exposure(BTC) == (None, Decimal('0.3'))

    state.update_order(
        order2,
        status = OrderStatus.CANCELLED
    )
    assert state.exposure(BTC) == (None, Decimal('0.2'))

    state.update_order(
        order1,
        status=OrderStatus.CANCELLED
    )
    assert state.exposure(BTC) == (None, Decimal('0.1'))

    # Although the symbol name is BTCUSDC, it will still allocate
    # into BTCUSDT and BTCUSDC
    state.expect(
        BTCUSDC,
        exposure=Decimal('0.3'),
        price=Decimal('10000'),
        use_market_order=False
    )

    state.set_balances([
        Balance(BTC, Decimal('1'), DECIMAL_ZERO)
    ], delta=True)

    assert state.exposure(BTC) == (None, Decimal('0.3'))

    orders, _ = state.get_orders()
    assert len(orders) == 2

    quantity_updated_count = 0
    status_updated_count = 0

    def handler(*args):
        nonlocal quantity_updated_count
        quantity_updated_count += 1

    def status_updated_handler(*args):
        nonlocal status_updated_count
        status_updated_count += 1

    state.on(TradingStateEvent.ORDER_FILLED_QUANTITY_UPDATED, handler)
    state.on(TradingStateEvent.ORDER_STATUS_UPDATED, status_updated_handler)

    for order in orders:
        target = order.target
        state.update_order(
            order,
            filled_quantity=order.ticket.quantity,
            quote_quantity=order.ticket.quantity * target.price
        )
        # It will not trigger quantity updated event, coz no change
        state.update_order(
            order,
            filled_quantity=order.ticket.quantity,
            quote_quantity=order.ticket.quantity * target.price
        )

    for order in orders:
        state.update_order(
            order,
            status=OrderStatus.FILLED
        )
        # It will not trigger status updated event
        state.update_order(
            order,
            status=OrderStatus.FILLED
        )
        target = order.target

    assert quantity_updated_count == 2
    assert status_updated_count == 2

    assert target.status is PositionTargetStatus.ACHIEVED

    state.set_balances([
        Balance(BTC, Decimal('2'), DECIMAL_ZERO)
    ], delta=True)

    assert BTC not in state._expected

    assert state.exposure(BTC) == (None, Decimal('0.4'))


def test_allocate_sell():
    state = init_state()

    # This will consume USDC and convert to USDT
    state.set_alt_currency_weights(
        (
            (Decimal('1'),),
            (Decimal('0'),)
        )
    )

    # Although the symbol name is BTCUSDC,
    # but the weight of USDC is 0, we will only allocate to USDT
    state.expect(
        BTCUSDC,
        exposure=Decimal('0'),
        price=Decimal('10000'),
        use_market_order=False
    )

    orders, _ = state.get_orders()
    order = orders.pop()

    assert order.ticket.symbol.quote_asset == USDT
    assert order.ticket.quantity == Decimal('1')
    assert order.ticket.side == OrderSide.SELL


def test_alt_currencies_edge_cases():
    state = init_state()

    # This will consume USDC and convert to USDT
    state.set_alt_currency_weights(
        (
            (Decimal('1'),),
            (Decimal('1'),)
        )
    )

    # The exposure delta is too small,
    # it is not possible to allocate into two quote currencies,
    # due to the limitation of NOTIONAL
    state.expect(
        BTCUSDC,
        exposure=Decimal('0.10005'),
        price=Decimal('10000'),
        use_market_order=False
    )

    orders, _ = state.get_orders()
    assert len(orders) == 1

    order = orders.pop()

    # Default account currency comes last, so it will allocate to USDT
    assert order.ticket.symbol.quote_asset == USDT
    state.update_order(order, status=OrderStatus.CANCELLED)

    # The exposure delta is too small
    # to create even a single order due to NOTIONAL
    state.expect(
        BTCUSDC,
        exposure=Decimal('0.10004'),
        price=Decimal('10000'),
        use_market_order=False
    )

    orders, _ = state.get_orders()
    assert not orders

    # The target will also be removed
    assert BTC not in state._expected

    # Sell
    state.expect(
        BTCUSDC,
        exposure=Decimal('0'),
        price=Decimal('10000'),
        use_market_order=False
    )

    orders, _ = state.get_orders()

    for order in orders:
        assert order.ticket.quantity == Decimal('0.5')


def test_allocation_not_enough_balance():
    state = init_state()

    # This setting is to convert USDT into USDC
    state.set_alt_currency_weights(
        (
            (Decimal('0'),),
            (Decimal('1'),)
        )
    )

    state.expect(
        BTCUSDC,
        exposure=Decimal('0.2'),
        price=Decimal('10000'),
        use_market_order=False
    )

    state.set_balances([
        Balance(USDT, DECIMAL_ZERO, DECIMAL_ZERO)
    ])

    orders, _ = state.get_orders()
    assert not orders


def test_expect_with_no_notional_limit_and_order_trades():
    state = init_state()

    price = Decimal('10000')

    state.set_notional_limit(Z, None)
    state.set_price(ZUSDT.name, price)
    state.set_symbol(ZUSDT)
    state.set_balances([
        Balance(Z, DECIMAL_ZERO, DECIMAL_ZERO)
    ])

    exception, success = state.expect(
        ZUSDT.name,
        exposure=Decimal('0.1'),
        price=price,
        use_market_order=False
    )

    assert exception is None
    orders, _ = state.get_orders()

    order = orders.pop()

    DECIMAL_20 = Decimal('20')
    DECIMAL_20K = Decimal('200000')

    assert order.ticket.quantity == DECIMAL_20
    assert order.ticket.side == OrderSide.BUY

    state.update_order(
        order,
        filled_quantity=DECIMAL_20,
        quote_quantity=DECIMAL_20K,
        commission_asset=USDC,
        commission_quantity=Decimal('0.02'),
        status=OrderStatus.FILLED
    )

    assert len(order.trades) == 1
    trade = order.trades[0]
    assert trade.base_quantity == DECIMAL_20
    assert trade.base_price == price
    assert trade.quote_quantity == DECIMAL_20K
    assert trade.quote_price == DECIMAL_ONE
    assert trade.commission_cost == Decimal('0.02')

    state.set_balances([
        Balance(Z, DECIMAL_20, DECIMAL_ZERO),
        Balance(USDT, DECIMAL_ZERO, DECIMAL_ZERO)
    ])

    state.expect(
        ZUSDT.name,
        exposure=DECIMAL_ZERO,
        price=price,
        use_market_order=False
    )

    orders, _ = state.get_orders()
    order = orders.pop()

    assert order.ticket.quantity == Decimal('20')
    assert order.ticket.side == OrderSide.SELL

    state.update_order(
        order,
        status=OrderStatus.REJECTED
    )

    assert not state._expected
