from decimal import Decimal
from datetime import datetime, timedelta

from trading_state import (
    Balance,
    LimitOrderTicket,
    MarketOrderTicket,
    MarketQuantityType,
    OrderSide,
    OrderStatus,
    StaleUpdate,
    Symbol,
    TimeInForce,
    TradingStateEvent,
)
from trading_state.common import (
    DECIMAL_ZERO,
    DECIMAL_ONE,
)

from .fixtures import (
    init_state,
    balance_time,
    BTCUSDC,
    USDC,
    BTCUSDT,
    BTC,
    USDT,
    ZUSDT,
    Z,
)


BTCUSDC_NAME = BTCUSDC.name
BTCUSDT_NAME = BTCUSDT.name


def _make_buy_limit_ticket(state, symbol_name, quantity, price):
    sym = state.get_symbol(symbol_name)
    return LimitOrderTicket(
        symbol=sym,
        side=OrderSide.BUY,
        quantity=quantity,
        price=price,
        time_in_force=TimeInForce.GTC,
    )


def test_trading_state_basics():
    from trading_state import AccountAssetHasNoExposureError

    state = init_state()

    assert state.get_account_value() == Decimal('410000')
    assert state.support_symbol(BTCUSDC_NAME)

    # Account currencies have no exposure (they are the unit of
    # measurement, not a position).
    exc, exp = state.exposure(
        USDT,
        include_unsettled_inflow=False,
        include_unsettled_outflow=False,
    )
    assert exp is None
    assert isinstance(exc, AccountAssetHasNoExposureError)
    assert exc.asset == USDT

    # BTC exposure with no orders in flight: balance (1) / limit (100k)
    # at price 10k = 0.1.
    exc, exp = state.exposure(
        BTC,
        include_unsettled_inflow=False,
        include_unsettled_outflow=False,
    )
    assert exc is None
    assert exp.ratio == Decimal('0.1')
    assert exp.holding == Decimal('1')
    assert exp.valuation_price == Decimal('10000')
    assert exp.notional_limit == Decimal('100000')
    assert exp.notional_value == Decimal('10000')
    assert exp.headroom_notional == Decimal('90000')
    assert exp.headroom_quantity == Decimal('9')


def test_add_order_returns_init_order_then_caller_drives_state_machine():
    state = init_state()

    ticket = _make_buy_limit_ticket(
        state, BTCUSDC_NAME, Decimal('1'), Decimal('10000')
    )

    exc, order = state.add_order(ticket, data={'strategy': 'momentum'})

    assert exc is None
    assert order is not None
    assert order.status is OrderStatus.INIT
    assert order.id is None
    assert order.filled_quantity == Decimal('0')
    assert order.data == {'strategy': 'momentum'}

    # Repr stable: side=BUY (not <OrderSide.BUY: ...>) and
    # quantity=1.00000000 (PrecisionFilter quantize)
    s = repr(order)
    assert 'side=BUY' in s
    assert 'status=INIT' in s
    assert 'quantity=1.00000000' in s

    # Caller-driven progression
    t0 = datetime(2024, 1, 1, 0, 0, 0)
    state.update_order(
        order,
        status=OrderStatus.SUBMITTING,
        updated_at=None,
        id=None,
        filled_quantity=None,
        quote_quantity=None,
        commission_asset=None,
        commission_quantity=None,
    )
    assert order.status is OrderStatus.SUBMITTING

    state.update_order(
        order,
        status=OrderStatus.CREATED,
        updated_at=t0,
        id='order-1',
        filled_quantity=Decimal('0.5'),
        quote_quantity=Decimal('5000'),
        commission_asset=None,
        commission_quantity=None,
    )
    assert order.status is OrderStatus.CREATED
    assert order.id == 'order-1'
    assert order.filled_quantity == Decimal('0.5')
    assert order.created_at == t0
    assert order.updated_at == t0
    assert state.get_order_by_id('order-1') is order


def test_filter_rejection_returns_exception_no_state_change():
    """An invalid ticket (notional too small) is rejected by add_order
    and never reaches the order manager / reconciliation."""
    state = init_state()

    ticket = _make_buy_limit_ticket(
        state, BTCUSDC_NAME, Decimal('0.00001'), Decimal('10000')
    )
    exc, order = state.add_order(ticket)
    assert order is None
    assert exc is not None
    assert isinstance(exc, ValueError)
    assert len(list(state.query_orders())) == 0


def test_update_order_silently_drops_status_regression_and_emits_stale():
    state = init_state()

    ticket = _make_buy_limit_ticket(
        state, BTCUSDC_NAME, Decimal('1'), Decimal('10000')
    )
    _, order = state.add_order(ticket)

    t0 = datetime(2024, 1, 1, 0, 0, 0)
    state.update_order(
        order,
        status=OrderStatus.CREATED,
        updated_at=t0,
        id='order-1',
        filled_quantity=None,
        quote_quantity=None,
        commission_asset=None,
        commission_quantity=None,
    )
    assert order.status is OrderStatus.CREATED

    captured = []
    state.on(
        TradingStateEvent.STALE_UPDATE,
        lambda event: captured.append(event),
    )

    # Attempt to regress to INIT
    state.update_order(
        order,
        status=OrderStatus.INIT,
        updated_at=None,
        id=None,
        filled_quantity=None,
        quote_quantity=None,
        commission_asset=None,
        commission_quantity=None,
    )

    assert order.status is OrderStatus.CREATED  # not regressed
    assert len(captured) == 1
    event = captured[0]
    assert isinstance(event, StaleUpdate)
    assert event.kind == 'order_status_regress'
    assert event.order is order
    assert event.incoming_value is OrderStatus.INIT
    assert event.current_value is OrderStatus.CREATED


def test_update_order_silently_drops_filled_regression():
    state = init_state()
    ticket = _make_buy_limit_ticket(
        state, BTCUSDC_NAME, Decimal('1'), Decimal('10000')
    )
    _, order = state.add_order(ticket)

    t0 = datetime(2024, 1, 1)
    state.update_order(
        order,
        status=OrderStatus.CREATED,
        updated_at=t0,
        id='o',
        filled_quantity=Decimal('0.5'),
        quote_quantity=Decimal('5000'),
        commission_asset=None,
        commission_quantity=None,
    )
    assert order.filled_quantity == Decimal('0.5')

    captured = []
    state.on(
        TradingStateEvent.STALE_UPDATE,
        lambda event: captured.append(event),
    )
    state.update_order(
        order,
        status=None,
        updated_at=None,
        id=None,
        filled_quantity=Decimal('0.3'),
        quote_quantity=None,
        commission_asset=None,
        commission_quantity=None,
    )
    assert order.filled_quantity == Decimal('0.5')
    assert captured[0].kind == 'order_filled_regress'


def test_update_order_silently_drops_time_regression():
    state = init_state()
    ticket = _make_buy_limit_ticket(
        state, BTCUSDC_NAME, Decimal('1'), Decimal('10000')
    )
    _, order = state.add_order(ticket)

    t1 = datetime(2024, 1, 1, 0, 0, 0)
    state.update_order(
        order,
        status=OrderStatus.CREATED,
        updated_at=t1,
        id='o',
        filled_quantity=None,
        quote_quantity=None,
        commission_asset=None,
        commission_quantity=None,
    )
    assert order.updated_at == t1

    captured = []
    state.on(
        TradingStateEvent.STALE_UPDATE,
        lambda event: captured.append(event),
    )

    earlier = t1 - timedelta(seconds=1)
    state.update_order(
        order,
        status=None,
        updated_at=earlier,
        id=None,
        filled_quantity=None,
        quote_quantity=None,
        commission_asset=None,
        commission_quantity=None,
    )
    assert order.updated_at == t1
    assert captured[0].kind == 'order_time_regress'


def test_set_balances_drops_stale_time_and_emits_stale():
    state = init_state()

    t_earlier = balance_time(-10)

    # init_state already wrote balances at balance_time(0); pushing an
    # earlier time must be rejected.
    captured = []
    state.on(
        TradingStateEvent.STALE_UPDATE,
        lambda event: captured.append(event),
    )

    state.set_balances([
        Balance(BTC, Decimal('999'), Decimal('0'), t_earlier),
    ])

    # current BTC balance unchanged
    assert state._balances.get_balance(BTC).free == Decimal('1')
    assert captured[0].kind == 'balance_time_regress'
    assert captured[0].asset == BTC


def test_unsettled_inflow_outflow_and_exposure_modes():
    """
    Order shows a fill via update_order; balance hasn't caught up yet.
    The unsettled inflow / outflow flags on exposure should toggle the
    reconciled component appropriately.
    """
    state = init_state()
    state.set_balances([
        Balance(BTC, Decimal('1'), Decimal('0'), balance_time(0)),
    ])

    # confirmed = 1; before any orders, no unsettled. exposure = 0.1
    exc, exp = state.exposure(
        BTC,
        include_unsettled_inflow=True,
        include_unsettled_outflow=True,
    )
    assert exc is None
    assert exp.ratio == Decimal('0.1')

    # Add a BUY order and fill it via order channel — balance NOT yet
    # updated.
    ticket = _make_buy_limit_ticket(
        state, BTCUSDC_NAME, Decimal('1'), Decimal('10000')
    )
    _, order = state.add_order(ticket)

    t_fill = balance_time(60)  # after current balance
    state.update_order(
        order,
        status=OrderStatus.FILLED,
        updated_at=t_fill,
        id='o-1',
        filled_quantity=Decimal('1'),
        quote_quantity=Decimal('10000'),
        commission_asset=None,
        commission_quantity=None,
    )

    # Confirmed BTC balance still 1 (no balance update yet).
    exc, base_exp = state.exposure(
        BTC,
        include_unsettled_inflow=False,
        include_unsettled_outflow=False,
    )
    assert exc is None
    assert base_exp.ratio == Decimal('0.1')

    # With inflow included: 2 BTC / 100k notional * 10k price = 0.2
    exc, inflow_exp = state.exposure(
        BTC,
        include_unsettled_inflow=True,
        include_unsettled_outflow=False,
    )
    assert exc is None
    assert inflow_exp.ratio == Decimal('0.2')

    # unsettled(BTC) should reflect the +1 BTC inflow
    exc, flow = state.unsettled(BTC)
    assert exc is None
    assert flow.inflow == Decimal('1')
    assert flow.outflow == Decimal('0')

    # Balance arriving later (time > order updated_at) closes the diff
    t_balance = t_fill + timedelta(seconds=1)
    state.set_balances([
        Balance(BTC, Decimal('2'), Decimal('0'), t_balance),
    ])
    exc, base_exp_after = state.exposure(
        BTC,
        include_unsettled_inflow=True,
        include_unsettled_outflow=False,
    )
    assert exc is None
    assert base_exp_after.ratio == Decimal('0.2')
    # unsettled is now 0
    _, flow_after = state.unsettled(BTC)
    assert flow_after.inflow == Decimal('0')
    assert flow_after.outflow == Decimal('0')


def test_cancel_order_sets_cancelling_then_cancelled_purges():
    state = init_state()
    ticket = _make_buy_limit_ticket(
        state, BTCUSDC_NAME, Decimal('1'), Decimal('10000')
    )
    _, order = state.add_order(ticket)

    t0 = datetime(2024, 1, 1)
    state.update_order(
        order,
        status=OrderStatus.CREATED,
        updated_at=t0,
        id='o-1',
        filled_quantity=None,
        quote_quantity=None,
        commission_asset=None,
        commission_quantity=None,
    )
    assert order in state._orders.open_orders

    state.cancel_order(order)
    assert order.status is OrderStatus.CANCELLING
    # Still open until exchange confirms CANCELLED
    assert order in state._orders.open_orders

    # Idempotent: another cancel is no-op
    state.cancel_order(order)
    assert order.status is OrderStatus.CANCELLING

    state.update_order(
        order,
        status=OrderStatus.CANCELLED,
        updated_at=t0 + timedelta(seconds=1),
        id=None,
        filled_quantity=None,
        quote_quantity=None,
        commission_asset=None,
        commission_quantity=None,
    )
    assert order not in state._orders.open_orders


def test_query_orders_by_data_subset():
    state = init_state()

    for tag in ['a', 'b']:
        ticket = _make_buy_limit_ticket(
            state, BTCUSDC_NAME, Decimal('1'), Decimal('10000')
        )
        _, order = state.add_order(ticket, data={'tag': tag})
        state.update_order(
            order,
            status=OrderStatus.CREATED,
            updated_at=datetime(2024, 1, 1),
            id=f'o-{tag}',
            filled_quantity=None,
            quote_quantity=None,
            commission_asset=None,
            commission_quantity=None,
        )

    matched = list(state.query_orders(data={'tag': 'a'}))
    assert len(matched) == 1
    assert matched[0].data['tag'] == 'a'

    # ticket subset match still works
    btc_buys = list(state.query_orders(ticket={'side': OrderSide.BUY}))
    assert len(btc_buys) == 2

    # callable predicate
    only_a = list(state.query_orders(
        id=lambda v, _: v == 'o-a',
    ))
    assert len(only_a) == 1


def test_allocate_requires_weights_to_be_set():
    from trading_state import AllocationWeightsNotSetError

    state = init_state()
    ticket = _make_buy_limit_ticket(
        state, BTCUSDT_NAME, Decimal('1'), Decimal('10000')
    )
    exc, out = state.allocate(ticket)
    assert out is None
    assert isinstance(exc, AllocationWeightsNotSetError)


def test_allocate_buy_across_alt_currencies():
    state = init_state()
    state.set_alt_currency_weights((
        (Decimal('1'),),  # BUY weights: USDC weight 1 (primary USDT also implicit 1)
        (Decimal('0'),),  # SELL weights
    ))

    canonical = _make_buy_limit_ticket(
        state, BTCUSDT_NAME, Decimal('1'), Decimal('10000')
    )
    exc, tickets = state.allocate(canonical)

    # Both USDC and USDT have balance, weights both > 0 → expect 2 tickets
    assert exc is None
    assert len(tickets) == 2
    quote_assets = {t.symbol.quote_asset for t in tickets}
    assert quote_assets == {USDC, USDT}
    for t in tickets:
        # Filter-applied: quantity quantized to 8-decimal precision
        assert t.quantity > Decimal('0')
        assert t.side is OrderSide.BUY


def test_allocate_sell_across_alt_currencies():
    state = init_state()
    state.set_alt_currency_weights((
        (Decimal('0'),),
        (Decimal('1'),),  # SELL weights: USDC weight 1
    ))

    canonical = LimitOrderTicket(
        symbol=state.get_symbol(BTCUSDT_NAME),
        side=OrderSide.SELL,
        quantity=Decimal('1'),
        price=Decimal('10000'),
        time_in_force=TimeInForce.GTC,
    )
    exc, tickets = state.allocate(canonical)

    assert exc is None
    assert len(tickets) == 2
    for t in tickets:
        assert t.side is OrderSide.SELL


def test_allocate_passthrough_for_market_quote_quantity():
    state = init_state()
    state.set_alt_currency_weights((
        (Decimal('1'),),
        (Decimal('0'),),
    ))
    sym = state.get_symbol(BTCUSDT_NAME)
    quote_market = MarketOrderTicket(
        symbol=sym,
        side=OrderSide.BUY,
        quantity=Decimal('1000'),
        quantity_type=MarketQuantityType.QUOTE,
        estimated_price=Decimal('10000'),
    )
    exc, out = state.allocate(quote_market)
    assert exc is None
    assert out == [quote_market]


def test_order_fill_records_trade_and_pnl():
    state = init_state()
    # Use Decimal('Infinity') to say "no effective cap" while still
    # satisfying the always-set invariant on notional_limit.
    state.set_notional_limit(Z, Decimal('Infinity'))
    state.set_price(ZUSDT.name, Decimal('10000'))
    state.set_symbol(ZUSDT)
    state.set_balances([
        Balance(Z, DECIMAL_ZERO, DECIMAL_ZERO, balance_time()),
    ])

    sym = state.get_symbol(ZUSDT.name)
    ticket = LimitOrderTicket(
        symbol=sym,
        side=OrderSide.BUY,
        quantity=Decimal('20'),
        price=Decimal('10000'),
        time_in_force=TimeInForce.GTC,
    )
    _, order = state.add_order(ticket)
    assert order is not None
    assert order.ticket.quantity == Decimal('20')

    t = datetime(2024, 1, 1)
    state.update_order(
        order,
        status=OrderStatus.FILLED,
        updated_at=t,
        id='ord',
        filled_quantity=Decimal('20'),
        quote_quantity=Decimal('200000'),
        commission_asset=USDC,
        commission_quantity=Decimal('0.02'),
    )

    assert len(order.trades) == 1
    trade = order.trades[0]
    assert trade.base_quantity == Decimal('20')
    assert trade.base_price == Decimal('10000')
    assert trade.quote_quantity == Decimal('200000')
    assert trade.quote_price == DECIMAL_ONE
    assert trade.commission_cost == Decimal('0.02')


def test_freeze_set_and_clear():
    state = init_state()

    # exposure baseline: 1 BTC / 100k limit at 10k price = 0.1
    exc, e0 = state.exposure(
        BTC,
        include_unsettled_inflow=False,
        include_unsettled_outflow=False,
    )
    assert exc is None and e0.ratio == Decimal('0.1')

    # Freezing all of it makes the available holding 0 → exposure 0
    state.freeze(BTC, Decimal('1'))
    exc, e1 = state.exposure(
        BTC,
        include_unsettled_inflow=False,
        include_unsettled_outflow=False,
    )
    assert exc is None and e1.ratio == Decimal('0')

    # Passing quantity=None clears the freeze
    state.freeze(BTC, None)
    exc, e2 = state.exposure(
        BTC,
        include_unsettled_inflow=False,
        include_unsettled_outflow=False,
    )
    assert exc is None and e2.ratio == Decimal('0.1')


def test_unsettled_propagates_asset_not_ready_error():
    state = init_state()
    exc, value = state.unsettled('UNKNOWN_ASSET')
    assert value is None
    assert exc is not None


def test_query_orders_unknown_criterion_matches_nothing():
    state = init_state()
    ticket = _make_buy_limit_ticket(
        state, BTCUSDC_NAME, Decimal('1'), Decimal('10000')
    )
    _, order = state.add_order(ticket)
    state.update_order(
        order,
        status=OrderStatus.CREATED,
        updated_at=datetime(2024, 1, 1),
        id='o',
        filled_quantity=None,
        quote_quantity=None,
        commission_asset=None,
        commission_quantity=None,
    )

    # criterion references an attribute that doesn't exist on Order:
    # _compare returns False, query returns no matches.
    assert list(state.query_orders(not_a_real_field='x')) == []

    # No-criteria query with a limit exercises the empty-criteria branch.
    assert len(list(state.query_orders(limit=1))) == 1


def test_order_rejected_path():
    state = init_state()
    ticket = _make_buy_limit_ticket(
        state, BTCUSDC_NAME, Decimal('1'), Decimal('10000')
    )
    _, order = state.add_order(ticket)
    state.update_order(
        order,
        status=OrderStatus.CREATED,
        updated_at=datetime(2024, 1, 1),
        id='o',
        filled_quantity=None,
        quote_quantity=None,
        commission_asset=None,
        commission_quantity=None,
    )
    assert order in state._orders.open_orders

    state.update_order(
        order,
        status=OrderStatus.REJECTED,
        updated_at=datetime(2024, 1, 1, 0, 0, 1),
        id=None,
        filled_quantity=None,
        quote_quantity=None,
        commission_asset=None,
        commission_quantity=None,
    )
    assert order not in state._orders.open_orders


def test_allocate_stop_loss_passes_through_with_weights_set():
    state = init_state()
    state.set_alt_currency_weights((
        (Decimal('0'),),
        (Decimal('0'),),
    ))
    # Stop-loss / take-profit families are out of scope of the
    # account-currency split logic; allocate passes them through as a
    # single sub-ticket without running the exposure pre-check.
    from trading_state import StopLossOrderTicket
    sym = state.get_symbol(BTCUSDT_NAME)
    sl = StopLossOrderTicket(
        symbol=sym,
        side=OrderSide.SELL,
        quantity=Decimal('1'),
        stop_price=Decimal('5000'),
    )
    exc, out = state.allocate(sl)
    assert exc is None
    assert out == [sl]


def test_allocate_buy_unset_notional_limit_propagates_exposure_error():
    """A BUY whose base asset has no notional_limit set surfaces the
    exposure pre-check error (NotionalLimitNotSetError)."""
    from trading_state import NotionalLimitNotSetError, Symbol
    state = init_state()
    state.set_alt_currency_weights((
        (Decimal('1'),),
        (Decimal('1'),),
    ))
    # Register a new symbol but skip set_notional_limit for its base.
    state.set_symbol(Symbol('FOOZZZ', 'FOO', 'ZZZ'))
    sym = state.get_symbol('FOOZZZ')
    ticket = LimitOrderTicket(
        symbol=sym,
        side=OrderSide.BUY,
        quantity=Decimal('1'),
        price=Decimal('10'),
        time_in_force=TimeInForce.GTC,
    )
    exc, out = state.allocate(ticket)
    assert out is None
    assert isinstance(exc, NotionalLimitNotSetError)


def test_allocate_sell_skips_missing_alt_symbol_buckets():
    """A SELL whose base asset has a symbol only in some weighted
    account currencies: the missing-symbol buckets are silently
    skipped and the present ones still produce sub-tickets."""
    state = init_state()
    state.set_alt_currency_weights((
        (Decimal('0'),),
        (Decimal('1'),),  # SELL: USDC weight 1, USDT implicit 1
    ))
    # Register a new base whose symbol exists only against USDT, not
    # USDC. The USDC bucket exercises the `alt_symbol is None: continue`
    # path inside state.allocate's resource gathering.
    from trading_state import Symbol
    state.set_symbol(Symbol('FOOUSDT', 'FOO', USDT))
    state.set_notional_limit('FOO', Decimal('100000'))
    state.set_price('FOOUSDT', Decimal('10'))
    state.set_balances([
        Balance('FOO', Decimal('100'), Decimal('0'), balance_time(0)),
    ])

    sym = state.get_symbol('FOOUSDT')
    ticket = LimitOrderTicket(
        symbol=sym,
        side=OrderSide.SELL,
        quantity=Decimal('10'),
        price=Decimal('10'),
        time_in_force=TimeInForce.GTC,
    )
    exc, out = state.allocate(ticket)
    assert exc is None
    assert len(out) == 1
    assert out[0].symbol.quote_asset == USDT


def test_allocate_sell_with_no_registered_symbols_falls_back_to_passthrough():
    """A SELL whose base asset has no symbol in any weighted account
    currency falls back to a single-element passthrough rather than
    raising an `InsufficientFreeBalanceError` (which is BUY-specific)."""
    state = init_state()
    state.set_alt_currency_weights((
        (Decimal('0'),),
        (Decimal('1'),),
    ))
    from trading_state import Symbol
    state.set_symbol(Symbol('FOOZZZ', 'FOO', 'ZZZ'))
    sym = state.get_symbol('FOOZZZ')
    ticket = LimitOrderTicket(
        symbol=sym,
        side=OrderSide.SELL,
        quantity=Decimal('1'),
        price=Decimal('10'),
        time_in_force=TimeInForce.GTC,
    )
    exc, out = state.allocate(ticket)
    assert exc is None
    assert out == [ticket]


def test_allocate_buy_zero_balance_returns_insufficient_free_balance():
    """A BUY allocation with every weighted account-currency bucket at
    zero free balance returns InsufficientFreeBalanceError."""
    from trading_state import InsufficientFreeBalanceError, TradingConfig
    config = TradingConfig(
        account_currency=USDT,
        alt_account_currencies=(USDC,),
    )
    state = init_state(config=config)
    state.set_alt_currency_weights((
        (Decimal('1'),),
        (Decimal('1'),),
    ))
    # Zero out both balances so every BUY bucket is ineligible.
    t = balance_time(100)
    state.set_balances([
        Balance(USDC, Decimal('0'), Decimal('0'), t),
        Balance(USDT, Decimal('0'), Decimal('0'), t),
    ])

    ticket = _make_buy_limit_ticket(
        state, BTCUSDT_NAME, Decimal('1'), Decimal('10000')
    )
    exc, out = state.allocate(ticket)
    assert out is None
    assert isinstance(exc, InsufficientFreeBalanceError)
    assert exc.asset == BTC


def test_unsettled_outflow_from_sell_fill():
    """SELL fill ahead of balance update produces outflow on the base
    asset."""
    state = init_state()

    # Pre-existing balance: 1 BTC. Then a SELL of 0.5 BTC is filled but
    # the balance update reflecting the decrease hasn't landed yet.
    sym = state.get_symbol(BTCUSDC_NAME)
    ticket = LimitOrderTicket(
        symbol=sym,
        side=OrderSide.SELL,
        quantity=Decimal('0.5'),
        price=Decimal('10000'),
        time_in_force=TimeInForce.GTC,
    )
    _, order = state.add_order(ticket)
    t_fill = balance_time(60)
    state.update_order(
        order,
        status=OrderStatus.FILLED,
        updated_at=t_fill,
        id='sell-1',
        filled_quantity=Decimal('0.5'),
        quote_quantity=Decimal('5000'),
        commission_asset=None,
        commission_quantity=None,
    )

    _, flow = state.unsettled(BTC)
    assert flow.inflow == Decimal('0')
    assert flow.outflow == Decimal('0.5')


def test_unsettled_handles_commission_asset_and_unrelated_balance_update():
    """An order with a commission paid in a third asset (BNB-style)
    creates an unsettled outflow on that third asset; balance updates
    on unrelated assets exercise the order_touches=False branches."""
    state = init_state()

    BNB = 'BNB'
    BNBUSDT = 'BNBUSDT'
    BNBUSDC = 'BNBUSDC'
    state.set_symbol(Symbol(BNBUSDT, BNB, USDT))
    state.set_symbol(Symbol(BNBUSDC, BNB, USDC))
    state.set_price(BNBUSDT, Decimal('500'))
    state.set_price(BNBUSDC, Decimal('500'))
    state.set_notional_limit(BNB, Decimal('5000'))
    state.set_balances([
        Balance(BNB, Decimal('10'), Decimal('0'), balance_time(0)),
    ])

    # Add a BUY order on BTCUSDC; commission paid in BNB.
    ticket = _make_buy_limit_ticket(
        state, BTCUSDC_NAME, Decimal('1'), Decimal('10000'),
    )
    _, order = state.add_order(ticket)
    state.update_order(
        order,
        status=OrderStatus.FILLED,
        updated_at=balance_time(60),
        id='o',
        filled_quantity=Decimal('1'),
        quote_quantity=Decimal('10000'),
        commission_asset=BNB,
        commission_quantity=Decimal('0.05'),
    )

    # BNB unsettled outflow reflects the commission, even though BNB is
    # neither base nor quote of the ticket.
    _, bnb_flow = state.unsettled(BNB)
    assert bnb_flow.outflow == Decimal('0.05')
    assert bnb_flow.inflow == Decimal('0')

    # A balance update on a totally unrelated asset traverses both the
    # on_balance_set and the implicit unsettled paths without touching
    # this order (order_touches returns False for ETHUSDT-base 'ETH').
    state.set_symbol(Symbol('ETHUSDT', 'ETH', USDT))
    state.set_price('ETHUSDT', Decimal('3000'))
    state.set_notional_limit('ETH', Decimal('30000'))
    state.set_balances([
        Balance('ETH', Decimal('1'), Decimal('0'), balance_time(120)),
    ])

    # unsettled for ETH yields zero flows — no order touches it.
    exc, eth_flow = state.unsettled('ETH')
    assert exc is None
    assert eth_flow.inflow == Decimal('0')
    assert eth_flow.outflow == Decimal('0')


def test_purge_fully_settled_after_balance_catches_up_with_commission():
    """After a balance update reflects the full fill (including
    commission), the order is purged from the reconciliation manager."""
    state = init_state()
    BNB = 'BNB'
    state.set_symbol(Symbol('BNBUSDT', BNB, USDT))
    state.set_symbol(Symbol('BNBUSDC', BNB, USDC))
    state.set_price('BNBUSDT', Decimal('500'))
    state.set_price('BNBUSDC', Decimal('500'))
    state.set_notional_limit(BNB, Decimal('5000'))
    state.set_balances([
        Balance(BNB, Decimal('10'), Decimal('0'), balance_time(0)),
    ])

    ticket = _make_buy_limit_ticket(
        state, BTCUSDC_NAME, Decimal('1'), Decimal('10000'),
    )
    _, order = state.add_order(ticket)
    t_fill = balance_time(60)
    state.update_order(
        order,
        status=OrderStatus.FILLED,
        updated_at=t_fill,
        id='o',
        filled_quantity=Decimal('1'),
        quote_quantity=Decimal('10000'),
        commission_asset=BNB,
        commission_quantity=Decimal('0.05'),
    )
    assert order in state._recon._orders

    # Land balance updates that reflect every leg of the fill at or
    # after the order's update time → the order is fully settled and
    # the recon manager drops it.
    t_settle = t_fill
    state.set_balances([
        Balance(BTC, Decimal('2'), Decimal('0'), t_settle),
    ])
    state.set_balances([
        Balance(USDC, Decimal('190000'), Decimal('0'), t_settle),
    ])
    state.set_balances([
        Balance(BNB, Decimal('9.95'), Decimal('0'), t_settle),
    ])

    assert order not in state._recon._orders


def test_allocate_with_market_base_ticket():
    """MarketOrderTicket with BASE quantity_type goes through the
    normal allocate path — estimated_price is used as reference."""
    state = init_state()
    state.set_alt_currency_weights((
        (Decimal('1'),),
        (Decimal('0'),),
    ))
    sym = state.get_symbol(BTCUSDT_NAME)
    ticket = MarketOrderTicket(
        symbol=sym,
        side=OrderSide.BUY,
        quantity=Decimal('1'),
        quantity_type=MarketQuantityType.BASE,
        estimated_price=Decimal('10000'),
    )
    exc, out = state.allocate(ticket)
    assert exc is None
    assert len(out) >= 1
    for t in out:
        assert isinstance(t, MarketOrderTicket)


def test_allocate_skips_zero_weight_bucket():
    state = init_state()
    # USDC weight 0, primary USDT implicit 1 -> only USDT bucket used
    state.set_alt_currency_weights((
        (Decimal('0'),),
        (Decimal('0'),),
    ))
    ticket = _make_buy_limit_ticket(
        state, BTCUSDT_NAME, Decimal('1'), Decimal('10000'),
    )
    exc, out = state.allocate(ticket)
    # Only the primary bucket got a sub-ticket
    assert exc is None
    assert len(out) == 1
    assert out[0].symbol.quote_asset == USDT


def test_allocate_sell_zero_quantity_surfaces_value_error():
    """A SELL with quantity == 0 produces no sub-ticket — under the
    new allocate contract that surfaces as a ValueError rather than a
    silent passthrough."""
    state = init_state()
    state.set_alt_currency_weights((
        (Decimal('0'),),
        (Decimal('1'),),
    ))
    sym = state.get_symbol(BTCUSDT_NAME)
    ticket = LimitOrderTicket(
        symbol=sym,
        side=OrderSide.SELL,
        quantity=Decimal('0'),
        price=Decimal('10000'),
        time_in_force=TimeInForce.GTC,
    )
    exc, out = state.allocate(ticket)
    assert out is None
    assert isinstance(exc, ValueError)


def test_allocate_filter_rejection_surfaces_filter_error():
    """When every candidate sub-ticket is rejected by the symbol's
    filters, allocate returns the first such filter exception rather
    than silently passing the original ticket through."""
    state = init_state()
    state.set_alt_currency_weights((
        (Decimal('1'),),
        (Decimal('0'),),
    ))
    sym = state.get_symbol(BTCUSDT_NAME)
    # Quantity intentionally too small to clear the symbol's NOTIONAL
    # filter so every candidate gets rejected.
    ticket = LimitOrderTicket(
        symbol=sym,
        side=OrderSide.BUY,
        quantity=Decimal('0.0000001'),
        price=Decimal('10000'),
        time_in_force=TimeInForce.GTC,
    )
    exc, out = state.allocate(ticket)
    assert out is None
    assert isinstance(exc, ValueError)


def test_completed_order_rejects_further_updates():
    state = init_state()
    ticket = _make_buy_limit_ticket(
        state, BTCUSDC_NAME, Decimal('1'), Decimal('10000')
    )
    _, order = state.add_order(ticket)
    t = datetime(2024, 1, 1)
    state.update_order(
        order,
        status=OrderStatus.FILLED,
        updated_at=t,
        id='o',
        filled_quantity=Decimal('1'),
        quote_quantity=Decimal('10000'),
        commission_asset=None,
        commission_quantity=None,
    )
    assert order.status is OrderStatus.FILLED
    # Further update is silently ignored by Order.update (status already
    # completed). No regression event fires either, because filled_quantity
    # is not regressing.
    state.update_order(
        order,
        status=OrderStatus.REJECTED,
        updated_at=t,
        id=None,
        filled_quantity=Decimal('1'),
        quote_quantity=None,
        commission_asset=None,
        commission_quantity=None,
    )
    assert order.status is OrderStatus.FILLED


def test_set_notional_limit_rejects_none_and_non_positive():
    import pytest as _pytest
    from trading_state import TradingConfig, TradingState
    state = TradingState(TradingConfig(account_currency=USDT))
    with _pytest.raises(TypeError, match='must be a Decimal'):
        state.set_notional_limit('BTC', None)  # type: ignore[arg-type]
    with _pytest.raises(ValueError, match='must be > 0'):
        state.set_notional_limit('BTC', Decimal('0'))
    with _pytest.raises(ValueError, match='must be > 0'):
        state.set_notional_limit('BTC', Decimal('-1'))


def test_allocate_buy_exceeding_notional_limit_returns_error():
    from trading_state import NotionalLimitExceededError
    state = init_state()
    state.set_alt_currency_weights((
        (Decimal('1'),),
        (Decimal('0'),),
    ))
    # init_state fixture sets BTC notional_limit=100k. Holding is 1 BTC
    # at price 10k → current notional 10k. A BUY of 10 BTC at 10k would
    # add 100k, totalling 110k > 100k cap.
    ticket = _make_buy_limit_ticket(
        state, BTCUSDT_NAME, Decimal('10'), Decimal('10000'),
    )
    exc, out = state.allocate(ticket)
    assert out is None
    assert isinstance(exc, NotionalLimitExceededError)
    assert exc.asset == BTC
    assert exc.notional_limit == Decimal('100000')


def test_user_order_listener_survives_terminal_status():
    """A listener the user attached directly to the Order should keep
    firing as long as the order continues to receive updates — the
    library's own internal cleanup at terminal status must only
    detach the library's own subscription, not the user's."""
    from trading_state.order import OrderUpdatedType
    state = init_state()
    ticket = _make_buy_limit_ticket(
        state, BTCUSDC_NAME, Decimal('1'), Decimal('10000')
    )
    _, order = state.add_order(ticket)

    seen_statuses = []
    order.on(
        OrderUpdatedType.STATUS_UPDATED,
        lambda _o, s: seen_statuses.append(s),
    )

    t = datetime(2024, 1, 1)
    state.update_order(
        order,
        status=OrderStatus.CREATED,
        updated_at=t,
        id='o',
        filled_quantity=None,
        quote_quantity=None,
        commission_asset=None,
        commission_quantity=None,
    )
    state.update_order(
        order,
        status=OrderStatus.FILLED,
        updated_at=t,
        id=None,
        filled_quantity=Decimal('1'),
        quote_quantity=Decimal('10000'),
        commission_asset=None,
        commission_quantity=None,
    )
    # Both transitions reached the user listener — the prior blanket
    # `order.off()` at FILLED would have left it with just the CREATED
    # entry.
    assert seen_statuses == [OrderStatus.CREATED, OrderStatus.FILLED]


def test_exposure_infinity_notional_limit_yields_zero_ratio():
    """Decimal('Infinity') as notional_limit is the documented escape
    hatch for "no effective cap"; the ratio collapses to 0 and
    headroom is unbounded."""
    state = init_state()
    state.set_notional_limit(BTC, Decimal('Infinity'))
    exc, exp = state.exposure(
        BTC,
        include_unsettled_inflow=False,
        include_unsettled_outflow=False,
    )
    assert exc is None
    assert exp.notional_limit == Decimal('Infinity')
    assert exp.ratio == Decimal('0')
    assert exp.headroom_notional == Decimal('Infinity')
    assert exp.headroom_quantity == Decimal('Infinity')
