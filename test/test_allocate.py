from decimal import Decimal

from trading_state import (
    Balance,
    LimitOrderTicket,
    MarketOrderTicket,
    MarketQuantityType,
    OrderSide,
    OrderStatus,
    Symbol,
    TimeInForce,
)
from trading_state.allocate import (
    buy_allocate,
    sell_allocate,
    AllocationResource,
)

from .fixtures import (
    init_state,
    balance_time,
    BTCUSDT,
    BTCUSDC,
    BTCFDUSD,
    USDC,
    USDT,
)


BTCUSDT_NAME = BTCUSDT.name
BTCUSDC_NAME = BTCUSDC.name


def _make_buy_limit_ticket(state, symbol_name, quantity, price):
    sym = state.get_symbol(symbol_name)
    return LimitOrderTicket(
        symbol=sym,
        side=OrderSide.BUY,
        quantity=quantity,
        price=price,
        time_in_force=TimeInForce.GTC,
    )


resources = [
    AllocationResource(
        BTCUSDT,
        free=Decimal('10000'),
        weight=Decimal('1'),
    ),
    AllocationResource(
        BTCUSDC,
        free=Decimal('10000'),
        weight=Decimal('1.5'),
    ),
    AllocationResource(
        BTCFDUSD,
        free=Decimal('10000'),
        weight=Decimal('2.5'),
    ),
]

resources = sorted(resources, key=lambda r: r.symbol.name)
price = Decimal('10000')

results = []


def assign(symbol: Symbol, quantity: Decimal) -> Decimal:
    """
    A test stub for the Assigner callback. Forces a small "leftover"
    return when quantity is below 0.5 so we can verify the compensate
    chain wiring without involving a real ticket / filter path.
    """
    ret = Decimal('0')
    if quantity <= Decimal('0.5'):
        ret = Decimal('0.1')

    quantity -= ret

    results.append((symbol, quantity, ret))
    return ret


def match_results(prefix, quantities, returns):
    for i, (s, q, r) in enumerate(
        sorted(results, key=lambda r: r[0].name)
    ):
        assert s == resources[i].symbol, f'{prefix}: symbol'
        assert q == quantities[i], f'{prefix}: quantity'
        assert r == returns[i], f'{prefix}: return'


def test_buy_allocate():
    def run(take: Decimal):
        results.clear()
        buy_allocate(
            resources,
            take=take,
            reference_price=price,
            assign=assign,
        )

    # Buy 5 BTC, but quote balance is not enough
    run(Decimal('5'))
    match_results(
        '5',
        [Decimal('1')] * 3,
        [Decimal('0')] * 3,
    )

    # Buy 2 BTC, enough but with returns
    run(Decimal('2'))
    match_results(
        '2',
        [Decimal('1'), Decimal('0.6'), Decimal('0.3')],
        [Decimal('0'), Decimal('0'), Decimal('0.1')],
    )

    # Buy 1 BTC, enough but with multiple returns
    run(Decimal('1'))
    match_results(
        '1',
        [Decimal('0.4'), Decimal('0.3'), Decimal('0.2')],
        [Decimal('0.1'), Decimal('0.1'), Decimal('0.1')],
    )

    run(Decimal('2.5'))
    match_results(
        '2.5',
        [Decimal('1'), Decimal('0.9'), Decimal('0.6')],
        [Decimal('0'), Decimal('0'), Decimal('0')],
    )


def test_sell_allocate():
    def run(take: Decimal):
        results.clear()
        sell_allocate(
            resources,
            take=take,
            assign=assign,
        )

    # No returns
    run(Decimal('5'))
    match_results(
        '5',
        [Decimal('2.5'), Decimal('1.5'), Decimal('1')],
        [Decimal('0')] * 3,
    )

    # Single return
    run(Decimal('2'))
    match_results(
        '2',
        [Decimal('1'), Decimal('0.7'), Decimal('0.3')],
        [Decimal('0'), Decimal('0'), Decimal('0.1')],
    )

    # Multiple returns
    run(Decimal('1'))
    match_results(
        '1',
        [Decimal('0.6'), Decimal('0.3'), Decimal('0.1')],
        [Decimal('0'), Decimal('0.1'), Decimal('0.1')],
    )


# State-level allocate flows ---------------------------------------
#
# Tests below exercise `state.allocate` end-to-end (cross-currency
# split, passthrough flow, fail-fast vs best-effort outcomes) rather
# than the lower-level `buy_allocate` / `sell_allocate` math above.


def test_create_order_invalid_allocate_weights_length():
    """`allocate=` length must match config.alt_account_currencies;
    a mismatch fails fast with InvalidAllocationWeightsError."""
    from trading_state import InvalidAllocationWeightsError
    state = init_state()
    ticket = _make_buy_limit_ticket(
        state, BTCUSDT_NAME, Decimal('1'), Decimal('10000')
    )
    exc, out = state.create_order(
        ticket,
        allocate=(Decimal('0.5'), Decimal('0.5')),  # 2 entries, config has 1 alt
    )
    assert out is None
    assert isinstance(exc, InvalidAllocationWeightsError)


def test_create_order_invalid_allocate_negative_weight():
    from trading_state import InvalidAllocationWeightsError
    state = init_state()
    ticket = _make_buy_limit_ticket(
        state, BTCUSDT_NAME, Decimal('1'), Decimal('10000')
    )
    exc, out = state.create_order(
        ticket,
        allocate=(Decimal('-0.5'),),
    )
    assert out is None
    assert isinstance(exc, InvalidAllocationWeightsError)


def test_create_order_no_allocate_keyword_raises_type_error():
    """`allocate` is a required keyword on create_order; omitting it
    is a Python-level TypeError, not a returned exception."""
    import pytest
    state = init_state()
    ticket = _make_buy_limit_ticket(
        state, BTCUSDT_NAME, Decimal('1'), Decimal('10000')
    )
    with pytest.raises(TypeError):
        state.create_order(ticket)  # type: ignore[call-arg]


def test_create_order_allocate_none_preserves_ticket_symbol():
    """With allocate=None there is no cross-currency split and the
    Order's ticket.symbol is preserved (no primary-symbol
    substitution)."""
    state = init_state()
    sym = state.get_symbol(BTCUSDC.name)
    ticket = LimitOrderTicket(
        symbol=sym,
        side=OrderSide.BUY,
        quantity=Decimal('1'),
        price=Decimal('10000'),
        time_in_force=TimeInForce.GTC,
    )
    exc, orders = state.create_order(ticket, allocate=None)
    assert exc is None
    assert len(orders) == 1
    assert orders[0].ticket.symbol is sym
    assert orders[0].ticket.symbol.quote_asset == USDC


def test_create_order_buy_split_across_alt_currencies():
    state = init_state()
    canonical = _make_buy_limit_ticket(
        state, BTCUSDT_NAME, Decimal('1'), Decimal('10000')
    )
    exc, orders = state.create_order(
        canonical,
        allocate=(Decimal('1'),),  # USDC weight 1 (USDT primary implicit 1)
    )
    assert exc is None
    assert len(orders) == 2
    quote_assets = {o.ticket.symbol.quote_asset for o in orders}
    assert quote_assets == {USDC, USDT}
    for order in orders:
        assert order.ticket.quantity > Decimal('0')
        assert order.ticket.side is OrderSide.BUY
        assert order.status is OrderStatus.INIT


def test_create_order_sell_split_across_alt_currencies():
    state = init_state()
    canonical = LimitOrderTicket(
        symbol=state.get_symbol(BTCUSDT_NAME),
        side=OrderSide.SELL,
        quantity=Decimal('1'),
        price=Decimal('10000'),
        time_in_force=TimeInForce.GTC,
    )
    exc, orders = state.create_order(
        canonical,
        allocate=(Decimal('1'),),  # USDC weight 1
    )
    assert exc is None
    assert len(orders) == 2
    for order in orders:
        assert order.ticket.side is OrderSide.SELL
        assert order.status is OrderStatus.INIT


def test_create_order_market_quote_buy_splits_across_alt_currencies():
    """MARKET(QUOTE) BUY is split across alt account currencies via
    estimated_price (used as the base/quote conversion factor).
    The split assumes stablecoin parity between primary and alts."""
    state = init_state()
    sym = state.get_symbol(BTCUSDT_NAME)
    quote_market = MarketOrderTicket(
        symbol=sym,
        side=OrderSide.BUY,
        quantity=Decimal('1000'),   # spend 1000 USDT total
        quantity_type=MarketQuantityType.QUOTE,
        estimated_price=Decimal('10000'),
    )
    exc, orders = state.create_order(
        quote_market, allocate=(Decimal('1'),),  # USDC weight 1
    )
    assert exc is None
    assert len(orders) == 2

    # Equal weights -> equal split of the base equivalent
    # (1000/10000 = 0.1 BTC total) -> per bucket sub_quote ~= 500.
    quote_assets = {o.ticket.symbol.quote_asset for o in orders}
    assert quote_assets == {USDC, USDT}
    for order in orders:
        assert isinstance(order.ticket, MarketOrderTicket)
        assert order.ticket.quantity_type is MarketQuantityType.QUOTE
        assert order.ticket.quantity == Decimal('500')
        assert order.ticket.side is OrderSide.BUY


def test_create_order_market_quote_sell_splits_with_aggregate_base_check():
    """MARKET(QUOTE) SELL splits the implied base quantity
    (quote / estimated_price) across buckets. The aggregate
    free-base pre-flight rejects when the caller's base balance
    cannot cover the implied base across all sub-orders."""
    state = init_state()
    sym = state.get_symbol(BTCUSDT_NAME)

    # init_state seeds 1 BTC free. quantity=10000 USDT implies
    # base_total = 1.0 BTC at estimated_price=10000 -> exactly fits.
    quote_market = MarketOrderTicket(
        symbol=sym,
        side=OrderSide.SELL,
        quantity=Decimal('10000'),
        quantity_type=MarketQuantityType.QUOTE,
        estimated_price=Decimal('10000'),
    )
    exc, orders = state.create_order(
        quote_market, allocate=(Decimal('1'),),  # USDC weight 1
    )
    assert exc is None
    assert len(orders) == 2
    quote_assets = {o.ticket.symbol.quote_asset for o in orders}
    assert quote_assets == {USDC, USDT}


def test_create_order_market_quote_sell_insufficient_base_returns_empty():
    state = init_state()
    sym = state.get_symbol(BTCUSDT_NAME)
    # 1 BTC held but caller asks for 50000 USDT -> implies 5 BTC needed.
    quote_market = MarketOrderTicket(
        symbol=sym,
        side=OrderSide.SELL,
        quantity=Decimal('50000'),
        quantity_type=MarketQuantityType.QUOTE,
        estimated_price=Decimal('10000'),
    )
    exc, orders = state.create_order(
        quote_market, allocate=(Decimal('1'),),
    )
    assert exc is None
    assert orders == []


def test_create_order_stop_loss_splits_across_alt_currencies():
    """A bare StopLoss SELL splits the base quantity across alt
    account currencies. stop_price is preserved on every sub-ticket."""
    from trading_state import StopLossOrderTicket
    state = init_state()
    sym = state.get_symbol(BTCUSDT_NAME)
    sl = StopLossOrderTicket(
        symbol=sym,
        side=OrderSide.SELL,
        quantity=Decimal('1'),
        stop_price=Decimal('5000'),
    )
    exc, orders = state.create_order(
        sl, allocate=(Decimal('1'),),  # USDC weight 1
    )
    assert exc is None
    assert len(orders) == 2
    quote_assets = {o.ticket.symbol.quote_asset for o in orders}
    assert quote_assets == {USDC, USDT}
    for order in orders:
        assert isinstance(order.ticket, StopLossOrderTicket)
        assert order.ticket.stop_price == Decimal('5000')
        assert order.ticket.side is OrderSide.SELL


def test_create_order_stop_loss_allocate_none_keeps_caller_symbol():
    """With allocate=None, the bare stop-loss creates exactly one
    Order on the caller's chosen symbol."""
    from trading_state import StopLossOrderTicket
    state = init_state()
    sym = state.get_symbol(BTCUSDT_NAME)
    sl = StopLossOrderTicket(
        symbol=sym,
        side=OrderSide.SELL,
        quantity=Decimal('1'),
        stop_price=Decimal('5000'),
    )
    exc, orders = state.create_order(sl, allocate=None)
    assert exc is None
    assert len(orders) == 1
    assert orders[0].ticket.symbol is sym


def test_create_order_stop_loss_limit_buy_splits_with_quote_balance_check():
    from trading_state import StopLossLimitOrderTicket
    state = init_state()
    sym = state.get_symbol(BTCUSDT_NAME)
    sll = StopLossLimitOrderTicket(
        symbol=sym,
        side=OrderSide.BUY,
        quantity=Decimal('1'),
        price=Decimal('10000'),
        stop_price=Decimal('11000'),
        time_in_force=TimeInForce.GTC,
    )
    exc, orders = state.create_order(sll, allocate=(Decimal('1'),))
    assert exc is None
    assert len(orders) == 2
    for order in orders:
        assert isinstance(order.ticket, StopLossLimitOrderTicket)
        assert order.ticket.price == Decimal('10000')
        assert order.ticket.stop_price == Decimal('11000')
        assert order.ticket.time_in_force is TimeInForce.GTC


def test_create_order_take_profit_limit_sell_splits_preserves_fields():
    from trading_state import TakeProfitLimitOrderTicket
    state = init_state()
    sym = state.get_symbol(BTCUSDT_NAME)
    tpl = TakeProfitLimitOrderTicket(
        symbol=sym,
        side=OrderSide.SELL,
        quantity=Decimal('1'),
        price=Decimal('40000'),
        stop_price=Decimal('39000'),
        time_in_force=TimeInForce.GTC,
    )
    exc, orders = state.create_order(tpl, allocate=(Decimal('1'),))
    assert exc is None
    assert len(orders) == 2
    for order in orders:
        assert isinstance(order.ticket, TakeProfitLimitOrderTicket)
        assert order.ticket.price == Decimal('40000')
        assert order.ticket.stop_price == Decimal('39000')


def test_create_order_trailing_delta_stop_falls_back_to_symbol_price():
    """A bare StopLoss with trailing_delta (no fixed stop_price) can
    still split — the split reference falls back to the symbol's
    last set_price."""
    from trading_state import StopLossOrderTicket
    state = init_state()
    sym = state.get_symbol(BTCUSDT_NAME)
    sl = StopLossOrderTicket(
        symbol=sym,
        side=OrderSide.SELL,
        quantity=Decimal('1'),
        trailing_delta=Decimal('100'),  # bps
    )
    exc, orders = state.create_order(sl, allocate=(Decimal('1'),))
    assert exc is None
    assert len(orders) == 2
    for order in orders:
        assert order.ticket.trailing_delta == Decimal('100')
        assert order.ticket.stop_price is None


def test_create_order_trailing_delta_stop_fail_fasts_without_symbol_price():
    """A trailing-delta stop on a symbol with no set_price has no
    reference for the split math; create_order fail-fasts with
    SymbolPriceNotReadyError."""
    from trading_state import (
        Balance,
        StopLossOrderTicket,
        SymbolPriceNotReadyError,
    )
    state = init_state()
    state.set_symbol(Symbol('FOOUSDT', 'FOO', USDT))
    state.set_notional_limit('FOO', Decimal('100000'))
    state.set_balances([
        Balance('FOO', Decimal('10'), Decimal('0'), balance_time(0)),
    ])
    sym = state.get_symbol('FOOUSDT')
    sl = StopLossOrderTicket(
        symbol=sym,
        side=OrderSide.SELL,
        quantity=Decimal('1'),
        trailing_delta=Decimal('100'),
    )
    exc, orders = state.create_order(sl, allocate=(Decimal('1'),))
    assert orders is None
    assert isinstance(exc, SymbolPriceNotReadyError)


def test_create_order_bare_stop_buy_notional_uses_stop_price_estimate():
    """bare StopLoss BUY notional pre-flight uses stop_price as an
    estimate. Estimated overshoot returns an empty list."""
    from trading_state import StopLossOrderTicket
    state = init_state()
    sym = state.get_symbol(BTCUSDT_NAME)
    # BTC notional_limit = 100000 (init_state). Holding 1 BTC at
    # price 10000 -> current notional 10000. A BUY 10 BTC at
    # stop_price 10000 would project to 110000 > 100000.
    sl = StopLossOrderTicket(
        symbol=sym,
        side=OrderSide.BUY,
        quantity=Decimal('10'),
        stop_price=Decimal('10000'),
    )
    exc, orders = state.create_order(sl, allocate=None)
    assert exc is None
    assert orders == []


def test_create_order_buy_unset_notional_limit_propagates_exposure_error():
    """A BUY whose base asset has no notional_limit set surfaces the
    exposure pre-check error (NotionalLimitNotSetError)."""
    from trading_state import NotionalLimitNotSetError, Symbol
    state = init_state()
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
    exc, out = state.create_order(ticket, allocate=(Decimal('1'),))
    assert out is None
    assert isinstance(exc, NotionalLimitNotSetError)


def test_create_order_sell_skips_missing_alt_symbol_buckets():
    """A SELL whose base asset has a symbol only in some weighted
    account currencies: the missing-symbol buckets are silently
    skipped and the present ones still produce sub-orders."""
    state = init_state()
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
    exc, out = state.create_order(ticket, allocate=(Decimal('1'),))
    assert exc is None
    assert len(out) == 1
    assert out[0].ticket.symbol.quote_asset == USDT


def test_create_order_sell_with_no_registered_symbols_returns_empty_list():
    state = init_state()
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
    exc, out = state.create_order(ticket, allocate=(Decimal('1'),))
    assert exc is None
    assert out == []


def test_create_order_buy_zero_balance_returns_empty_list():
    """A BUY allocation with every account-currency bucket at zero
    free balance: every bucket fails the eligibility check, returns
    empty list (best-effort)."""
    from trading_state import TradingConfig
    config = TradingConfig(
        account_currency=USDT,
        alt_account_currencies=(USDC,),
    )
    state = init_state(config=config)
    # Zero out both balances so every BUY bucket is ineligible.
    t = balance_time(100)
    state.set_balances([
        Balance(USDC, Decimal('0'), Decimal('0'), t),
        Balance(USDT, Decimal('0'), Decimal('0'), t),
    ])

    ticket = _make_buy_limit_ticket(
        state, BTCUSDT_NAME, Decimal('1'), Decimal('10000')
    )
    exc, out = state.create_order(ticket, allocate=(Decimal('1'),))
    assert exc is None
    assert out == []


def test_create_order_market_base_ticket_splits():
    state = init_state()
    sym = state.get_symbol(BTCUSDT_NAME)
    ticket = MarketOrderTicket(
        symbol=sym,
        side=OrderSide.BUY,
        quantity=Decimal('1'),
        quantity_type=MarketQuantityType.BASE,
        estimated_price=Decimal('10000'),
    )
    exc, out = state.create_order(ticket, allocate=(Decimal('1'),))
    assert exc is None
    assert len(out) >= 1
    for order in out:
        assert isinstance(order.ticket, MarketOrderTicket)


def test_create_order_split_skips_zero_weight_bucket():
    """USDC weight 0 with allocate -> only USDT primary bucket
    produces an Order. ticket.symbol is rewritten to the primary
    pair (because we asked for split, not allocate=None)."""
    state = init_state()
    ticket = _make_buy_limit_ticket(
        state, BTCUSDT_NAME, Decimal('1'), Decimal('10000'),
    )
    exc, out = state.create_order(ticket, allocate=(Decimal('0'),))
    assert exc is None
    assert len(out) == 1
    assert out[0].ticket.symbol.quote_asset == USDT


def test_create_order_sell_zero_quantity_returns_empty_list():
    """A SELL with quantity == 0 produces no Orders."""
    state = init_state()
    sym = state.get_symbol(BTCUSDT_NAME)
    ticket = LimitOrderTicket(
        symbol=sym,
        side=OrderSide.SELL,
        quantity=Decimal('0'),
        price=Decimal('10000'),
        time_in_force=TimeInForce.GTC,
    )
    exc, out = state.create_order(ticket, allocate=(Decimal('1'),))
    assert exc is None
    assert out == []


def test_create_order_filter_rejection_returns_empty_list():
    """Every candidate sub-ticket rejected by NOTIONAL filter ->
    empty list, no exception. State unchanged."""
    state = init_state()
    sym = state.get_symbol(BTCUSDT_NAME)
    ticket = LimitOrderTicket(
        symbol=sym,
        side=OrderSide.BUY,
        quantity=Decimal('0.0000001'),
        price=Decimal('10000'),
        time_in_force=TimeInForce.GTC,
    )
    exc, out = state.create_order(ticket, allocate=(Decimal('1'),))
    assert exc is None
    assert out == []
    assert len(list(state.query_orders())) == 0


def test_create_order_buy_exceeding_notional_limit_returns_empty_list():
    """init_state sets BTC notional_limit=100k. Holding 1 BTC at
    price 10k -> current notional 10k. A BUY of 10 BTC at 10k would
    add 100k, totalling 110k > 100k cap. Aggregate notional violation
    is best-effort: empty list, no exception."""
    state = init_state()
    ticket = _make_buy_limit_ticket(
        state, BTCUSDT_NAME, Decimal('10'), Decimal('10000'),
    )
    exc, out = state.create_order(ticket, allocate=(Decimal('1'),))
    assert exc is None
    assert out == []
    assert len(list(state.query_orders())) == 0


