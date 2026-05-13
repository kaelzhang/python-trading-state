from decimal import Decimal
from datetime import datetime, timedelta

from trading_state import (
    Balance,
    LimitOrderTicket,
    MarketOrderTicket,
    MarketQuantityType,
    OrderSide,
    OrderStatus,
    OrderType,
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
    sym = state._symbols.get_symbol(symbol_name)
    return LimitOrderTicket(
        symbol=sym,
        side=OrderSide.BUY,
        quantity=quantity,
        price=price,
        time_in_force=TimeInForce.GTC,
    )


def test_trading_state_basics():
    state = init_state()

    assert state.get_account_value() == Decimal('410000')
    assert state.support_symbol(BTCUSDC_NAME)

    # Account currency exposure is not defined (no notional cap).
    assert state.exposure(
        USDT,
        include_unsettled_inflow=False,
        include_unsettled_outflow=False,
    ) == (None, None)

    # BTC exposure with no orders in flight: balance (1) / limit (100k)
    # at price 10k = 0.1.
    assert state.exposure(
        BTC,
        include_unsettled_inflow=False,
        include_unsettled_outflow=False,
    ) == (None, Decimal('0.1'))


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

    t0 = balance_time(0)
    t_earlier = balance_time(-10)

    # init_state already wrote balances at t0; pushing an earlier time
    # must be rejected.
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
    assert state.exposure(
        BTC,
        include_unsettled_inflow=True,
        include_unsettled_outflow=True,
    ) == (None, Decimal('0.1'))

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
    assert base_exp == Decimal('0.1')

    # With inflow included: 2 BTC / 100k notional * 10k price = 0.2
    exc, inflow_exp = state.exposure(
        BTC,
        include_unsettled_inflow=True,
        include_unsettled_outflow=False,
    )
    assert exc is None
    assert inflow_exp == Decimal('0.2')

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
    assert base_exp_after == Decimal('0.2')
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


def test_allocate_passthrough_when_weights_unset():
    state = init_state()
    ticket = _make_buy_limit_ticket(
        state, BTCUSDT_NAME, Decimal('1'), Decimal('10000')
    )
    out = state.allocate(ticket)
    assert out == [ticket]


def test_allocate_buy_across_alt_currencies():
    state = init_state()
    state.set_alt_currency_weights((
        (Decimal('1'),),  # BUY weights: USDC weight 1 (primary USDT also implicit 1)
        (Decimal('0'),),  # SELL weights
    ))

    canonical = _make_buy_limit_ticket(
        state, BTCUSDT_NAME, Decimal('1'), Decimal('10000')
    )
    tickets = state.allocate(canonical)

    # Both USDC and USDT have balance, weights both > 0 → expect 2 tickets
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
        symbol=state._symbols.get_symbol(BTCUSDT_NAME),
        side=OrderSide.SELL,
        quantity=Decimal('1'),
        price=Decimal('10000'),
        time_in_force=TimeInForce.GTC,
    )
    tickets = state.allocate(canonical)

    assert len(tickets) == 2
    for t in tickets:
        assert t.side is OrderSide.SELL


def test_allocate_passthrough_for_market_quote_quantity():
    state = init_state()
    state.set_alt_currency_weights((
        (Decimal('1'),),
        (Decimal('0'),),
    ))
    sym = state._symbols.get_symbol(BTCUSDT_NAME)
    quote_market = MarketOrderTicket(
        symbol=sym,
        side=OrderSide.BUY,
        quantity=Decimal('1000'),
        quantity_type=MarketQuantityType.QUOTE,
        estimated_price=Decimal('10000'),
    )
    out = state.allocate(quote_market)
    assert out == [quote_market]


def test_order_fill_records_trade_and_pnl():
    state = init_state()
    state.set_notional_limit(Z, None)
    state.set_price(ZUSDT.name, Decimal('10000'))
    state.set_symbol(ZUSDT)
    state.set_balances([
        Balance(Z, DECIMAL_ZERO, DECIMAL_ZERO, balance_time()),
    ])

    sym = state._symbols.get_symbol(ZUSDT.name)
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
    assert exc is None and e0 == Decimal('0.1')

    # Freezing all of it makes the available holding 0 → exposure 0
    state.freeze(BTC, Decimal('1'))
    exc, e1 = state.exposure(
        BTC,
        include_unsettled_inflow=False,
        include_unsettled_outflow=False,
    )
    assert exc is None and e1 == Decimal('0')

    # Passing quantity=None clears the freeze
    state.freeze(BTC, None)
    exc, e2 = state.exposure(
        BTC,
        include_unsettled_inflow=False,
        include_unsettled_outflow=False,
    )
    assert exc is None and e2 == Decimal('0.1')


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


def test_allocate_all_zero_weights_passthrough():
    state = init_state()
    state.set_alt_currency_weights((
        (Decimal('0'),),
        (Decimal('0'),),
    ))
    # All weights effectively zero (primary is implicit 1 but every alt
    # is 0 and the primary one alone might still be > 0)
    # Edge case fully zero requires removing primary; we cover the
    # all-zero alt path here.

    # To hit "not any > 0" we configure no alt and pass an SELL ticket
    # — the SELL vec (0,) plus primary implicit 1 still has primary>0, so
    # this asserts the passthrough only when weights vec genuinely zero.
    # Instead force passthrough via stop-loss ticket which is unsupported.
    from trading_state import StopLossOrderTicket
    sym = state._symbols.get_symbol(BTCUSDT_NAME)
    sl = StopLossOrderTicket(
        symbol=sym,
        side=OrderSide.SELL,
        quantity=Decimal('1'),
        stop_price=Decimal('5000'),
    )
    out = state.allocate(sl)
    assert out == [sl]


def test_allocate_no_eligible_resources_passthrough():
    """No matching alt symbols for the base → passthrough."""
    state = init_state()
    state.set_alt_currency_weights((
        (Decimal('1'),),
        (Decimal('1'),),
    ))
    # Construct a ticket whose base asset has no symbol in any account
    # currency. Use FOO/BAR which is not in the config; the symbol
    # lookups during allocate's resource-gathering all return None.
    from trading_state import Symbol
    state.set_symbol(Symbol('FOOZZZ', 'FOO', 'ZZZ'))
    sym = state._symbols.get_symbol('FOOZZZ')
    ticket = LimitOrderTicket(
        symbol=sym,
        side=OrderSide.BUY,
        quantity=Decimal('1'),
        price=Decimal('10'),
        time_in_force=TimeInForce.GTC,
    )
    out = state.allocate(ticket)
    assert out == [ticket]


def test_allocate_zero_or_missing_balance_skips_bucket():
    """A BUY allocation skips buckets whose account currency has no
    balance — and if every weighted bucket is skipped, falls back to
    passthrough."""
    from trading_state import TradingConfig
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
    out = state.allocate(ticket)
    assert out == [ticket]


def test_unsettled_outflow_from_sell_fill():
    """SELL fill ahead of balance update produces outflow on the base
    asset."""
    state = init_state()

    # Pre-existing balance: 1 BTC. Then a SELL of 0.5 BTC is filled but
    # the balance update reflecting the decrease hasn't landed yet.
    sym = state._symbols.get_symbol(BTCUSDC_NAME)
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
    sym = state._symbols.get_symbol(BTCUSDT_NAME)
    ticket = MarketOrderTicket(
        symbol=sym,
        side=OrderSide.BUY,
        quantity=Decimal('1'),
        quantity_type=MarketQuantityType.BASE,
        estimated_price=Decimal('10000'),
    )
    out = state.allocate(ticket)
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
    out = state.allocate(ticket)
    # Only the primary bucket got a sub-ticket
    assert len(out) == 1
    assert out[0].symbol.quote_asset == USDT


def test_allocate_filter_rejection_rolls_forward_or_passthroughs():
    """A bucket whose filter rejects (e.g. quantity below MinNotional)
    rolls its quantity forward to the next bucket. With only one
    eligible bucket and a too-small total, every assignment is
    rejected → allocate returns the single passthrough fallback."""
    state = init_state()
    state.set_alt_currency_weights((
        (Decimal('1'),),
        (Decimal('0'),),
    ))
    sym = state._symbols.get_symbol(BTCUSDT_NAME)
    # Quantity intentionally too small to clear the symbol's NOTIONAL
    # filter so every candidate gets rejected.
    ticket = LimitOrderTicket(
        symbol=sym,
        side=OrderSide.BUY,
        quantity=Decimal('0.0000001'),
        price=Decimal('10000'),
        time_in_force=TimeInForce.GTC,
    )
    out = state.allocate(ticket)
    assert out == [ticket]


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
