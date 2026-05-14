from datetime import datetime
from decimal import Decimal

from trading_state import (
    TradingConfig,
    Balance,
    CashFlow,
    LimitOrderTicket,
    OrderSide,
    OrderStatus,
    Symbol,
    TimeInForce,
)

from trading_state.common import (
    DECIMAL_ZERO,
)

from .fixtures import (
    init_state,
    balance_time,
    BTC,
    ETH,
    USDT,
    USDC,
    Z,
    X,
    ZUSDT,
    BTCUSDT,
    ETHUSDT,
    DEFAULT_CONFIG_KWARGS,
)


def _buy_limit(state, symbol_name, quantity, price):
    sym = state.get_symbol(symbol_name)
    return LimitOrderTicket(
        symbol=sym,
        side=OrderSide.BUY,
        quantity=quantity,
        price=price,
        time_in_force=TimeInForce.GTC,
    )


def _sell_limit(state, symbol_name, quantity, price):
    sym = state.get_symbol(symbol_name)
    return LimitOrderTicket(
        symbol=sym,
        side=OrderSide.SELL,
        quantity=quantity,
        price=price,
        time_in_force=TimeInForce.GTC,
    )


def test_pnl():
    config = TradingConfig(
        benchmark_assets=(BTC, ETH),
        **DEFAULT_CONFIG_KWARGS,
    )
    state = init_state(config=config)

    state.set_symbol(ZUSDT)

    now = datetime.now()
    t = balance_time(0)

    # Cash flow before balance is set and before init: skip
    cash_flow_z = CashFlow(Z, Decimal('0.5'), now)
    state.set_cash_flow(cash_flow_z)

    state.set_balances([
        # invalid balance, asset not found
        Balance('invalid', Decimal('1'), Decimal('0'), t),
        # Zero balance, which is actually invalid
        Balance(X, DECIMAL_ZERO, DECIMAL_ZERO, t),
    ])

    # Initial record (BTC: $10000)
    node = state.record(time=now)

    # Cash flow before balance is set, skip
    state.set_cash_flow(cash_flow_z)

    state.set_balances([
        Balance(Z, Decimal('1'), Decimal('0'), t),
    ])

    # Set the same cash flow before price is ready, skip
    state.set_cash_flow(cash_flow_z)

    assert node.time == now
    assert node.realized_pnl == DECIMAL_ZERO
    assert node.unrealized_pnl == DECIMAL_ZERO

    assert BTC in node.positions
    # Price not ready yet
    assert Z not in node.positions

    # Account currencies don't have positions
    assert USDT not in node.positions
    assert USDC not in node.positions

    BTC_position = node.positions[BTC]
    assert BTC_position.quantity == Decimal('1')
    assert BTC_position.cost == Decimal('10000')
    assert BTC_position.valuation_price == Decimal('10000')

    assert node.unrealized_pnl == DECIMAL_ZERO

    # Price increased => unrealized PnL increased
    price = Decimal('20000')
    state.set_price(BTCUSDT.name, price)

    now2 = datetime.now()
    node2 = state.record(time=now2)

    assert node2.positions[BTC].unrealized_pnl == Decimal('10000')
    assert node2.unrealized_pnl == Decimal('10000')

    now3 = datetime.now()
    t3 = balance_time(60)

    # Cash flow of BTC, + $20000
    state.set_balances([
        Balance(BTC, Decimal('1'), Decimal('0'), t3),
    ], delta=True)

    cash_flow = CashFlow(BTC, Decimal('1'), now3)
    state.set_cash_flow(cash_flow)
    # Same cash flow again -> ignored
    state.set_cash_flow(cash_flow)

    assert 'CashFlow(BTC' in repr(cash_flow)

    node3 = state.record(time=now3)

    BTC_position3 = node3.positions[BTC]
    assert BTC_position3.quantity == Decimal('2')
    assert BTC_position3.cost == Decimal('30000')
    assert node3.unrealized_pnl == Decimal('10000')
    assert node3.net_cash_flow == Decimal('20000')

    _, exposure = state.exposure(
        BTC,
        include_unsettled_inflow=False,
        include_unsettled_outflow=False,
    )
    assert exposure.ratio == Decimal('0.4')

    # Buy via add_order
    ticket = _buy_limit(state, BTCUSDT.name, Decimal('0.5'), price)
    _, order = state.add_order(ticket)
    assert order is not None
    assert order.ticket.symbol.name == BTCUSDT.name
    assert order.ticket.side is OrderSide.BUY
    assert order.ticket.quantity == Decimal('0.5')
    assert order.ticket.price == Decimal('20000')

    state.update_order(
        order,
        status=OrderStatus.FILLED,
        updated_at=datetime.now(),
        id='buy-1',
        filled_quantity=Decimal('0.5'),
        # The USDT used is less than the expected quote quantity
        quote_quantity=Decimal('5000'),
        commission_asset=None,
        commission_quantity=None,
    )

    # Price decreased significantly
    price2 = Decimal('5000')
    state.set_price(BTCUSDT.name, price2)

    assert state.get_price(ETHUSDT.name) is None
    state.set_price(ETHUSDT.name, price2)
    assert state.get_price(ETHUSDT.name) == price2

    now5 = datetime.now()
    node5 = state.record(time=now5)

    BTC_position5 = node5.positions[BTC]
    assert BTC_position5.unrealized_pnl == Decimal('-22500')
    assert BTC_position5.quantity == Decimal('2.5')
    assert node5.unrealized_pnl == Decimal('-22500')

    # Set price of Z; balance of Z should be treated as cash flow.
    state.set_price(ZUSDT.name, Decimal('10000'))

    now6 = datetime.now()
    node6 = state.record(time=now6)

    assert node6.net_cash_flow == Decimal('30000')
    assert node6.benchmarks[BTC].benchmark_return == Decimal('-0.5')

    ETH_benchmark6 = node6.benchmarks[ETH]
    assert ETH_benchmark6.benchmark_return == Decimal('0')
    assert ETH_benchmark6.price == Decimal('5000')
    assert ETH_benchmark6.asset == ETH

    # Sell
    ticket = _sell_limit(state, BTCUSDT.name, Decimal('2'), price2)
    _, order = state.add_order(ticket)
    assert order is not None
    state.update_order(
        order,
        status=OrderStatus.FILLED,
        updated_at=datetime.now(),
        id='sell-1',
        filled_quantity=Decimal('2'),
        # The USDT used is less than the expected quote quantity
        quote_quantity=Decimal('15000'),
        commission_asset=None,
        commission_quantity=None,
    )

    now7 = datetime.now()
    node7 = state.record(time=now7)

    assert node7.realized_pnl == Decimal('-15000')
    assert node7.unrealized_pnl == Decimal('-2500')

    # Sell more
    ticket = _sell_limit(state, BTCUSDT.name, Decimal('0.1'), price2)
    _, order = state.add_order(ticket)
    assert order is not None
    state.update_order(
        order,
        status=OrderStatus.FILLED,
        updated_at=datetime.now(),
        id='sell-2',
        filled_quantity=Decimal('0.1'),
        quote_quantity=Decimal('500'),
        commission_asset=None,
        commission_quantity=None,
    )

    now8 = datetime.now()
    node8 = state.record(time=now8)

    assert node8.realized_pnl == Decimal('-15500')

    # External cash flow for Z (withdrawal)
    state.set_cash_flow(
        CashFlow(Z, Decimal('-2'), datetime.now())
    )
    # Out-of-order cash flow: ignored
    state.set_cash_flow(
        CashFlow(Z, Decimal('-1'), datetime.now())
    )
    # Zero cash flow: invalid in practice but state should swallow it
    state.set_cash_flow(
        CashFlow(Z, Decimal('0'), datetime.now())
    )

    now9 = datetime.now()
    node9 = state.record(time=now9)

    assert node9.positions[Z].quantity == Decimal('0')


def test_buy_commission_in_base_shrinks_tracked_position():
    """BUY where the commission is paid in the base asset: the
    exchange delivers (filled − commission) of base, so position
    quantity must match the post-fee balance."""
    state = init_state()
    state.record(time=datetime.now())  # initialise PerformanceTracker

    ticket = _buy_limit(state, BTCUSDT.name, Decimal('1'), Decimal('10000'))
    _, order = state.add_order(ticket)
    state.update_order(
        order,
        status=OrderStatus.FILLED,
        updated_at=datetime.now(),
        id='buy-fee-in-base',
        filled_quantity=Decimal('1'),
        quote_quantity=Decimal('10000'),
        commission_asset=BTC,
        commission_quantity=Decimal('0.01'),
    )

    node = state.record(time=datetime.now())
    btc_position = node.positions[BTC]
    # Initial 1 BTC + (1 filled - 0.01 commission) = 1.99 net.
    assert btc_position.quantity == Decimal('1.99')
    # Initial cost 1 * 10000 + this trade's 10000 acc-currency outlay
    # = 20000 total for 1.99 BTC.
    assert btc_position.cost == Decimal('20000')


def test_buy_commission_in_third_asset_folds_cc_into_cost_basis():
    """BUY whose commission is paid in a 3rd asset (BNB-style): the
    fee is real account-currency outlay and must land in the new
    position's cost basis."""
    state = init_state()
    state.set_symbol(Symbol('BNBUSDT', 'BNB', USDT))
    state.set_price('BNBUSDT', Decimal('500'))
    state.set_notional_limit('BNB', Decimal('5000'))
    state.set_balances([
        Balance('BNB', Decimal('10'), Decimal('0'), balance_time()),
    ])
    state.record(time=datetime.now())

    ticket = _buy_limit(state, BTCUSDT.name, Decimal('1'), Decimal('10000'))
    _, order = state.add_order(ticket)
    state.update_order(
        order,
        status=OrderStatus.FILLED,
        updated_at=datetime.now(),
        id='buy-fee-in-bnb',
        filled_quantity=Decimal('1'),
        quote_quantity=Decimal('10000'),
        commission_asset='BNB',
        commission_quantity=Decimal('0.05'),
    )

    node = state.record(time=datetime.now())
    btc_position = node.positions[BTC]
    # Initial 1 BTC + 1 filled (gross, no overlap with BTC) = 2 BTC.
    assert btc_position.quantity == Decimal('2')
    # Initial 10000 + this trade's 10000 + commission 0.05 BNB * 500
    # USDT/BNB = 25 USDT = 20025.
    assert btc_position.cost == Decimal('20025')


def test_sell_commission_in_quote_subtracts_cc_separately():
    """SELL whose commission is paid in the (account-currency) quote
    asset: the FIFO removal stays at gross filled, and `cc` is
    deducted from realized PnL separately — the increase-side
    shrinkage of the (account-tracked) quote has no PnL effect."""
    state = init_state()
    state.record(time=datetime.now())

    # Accumulate 2 BTC at known cost basis: existing 1 BTC @10000 +
    # bought 1 BTC @5000 (no fee) → cost 15000, lots [1@10000, 1@5000].
    buy_ticket = _buy_limit(state, BTCUSDT.name, Decimal('1'), Decimal('5000'))
    _, buy_order = state.add_order(buy_ticket)
    state.update_order(
        buy_order,
        status=OrderStatus.FILLED,
        updated_at=datetime.now(),
        id='buy-no-fee',
        filled_quantity=Decimal('1'),
        quote_quantity=Decimal('5000'),
        commission_asset=None,
        commission_quantity=None,
    )

    # SELL 1 BTC @10000, fee 10 USDT (commission_asset == quote == account).
    sell_ticket = _sell_limit(state, BTCUSDT.name, Decimal('1'), Decimal('10000'))
    _, sell_order = state.add_order(sell_ticket)
    state.update_order(
        sell_order,
        status=OrderStatus.FILLED,
        updated_at=datetime.now(),
        id='sell-fee-in-quote',
        filled_quantity=Decimal('1'),
        quote_quantity=Decimal('10000'),
        commission_asset=USDT,
        commission_quantity=Decimal('10'),
    )

    node = state.record(time=datetime.now())
    # FIFO removes 1 BTC from the initial 10000-cost lot.
    # Realized PnL = proceeds 10000 − cost 10000 − cc 10 = −10.
    assert node.realized_pnl == Decimal('-10')


def test_sell_commission_in_base_widens_fifo_without_double_subtracting():
    """SELL whose commission is paid in base (BTC): the FIFO removal
    must widen by the commission, and realized PnL must not double-
    subtract `cc` since the inventory cost of the consumed commission
    is already in cost_removed."""
    state = init_state()
    state.record(time=datetime.now())

    # First, accumulate inventory at a known cost basis: BUY 1 BTC
    # @ 5000 USDT, no fee. Position then has 1 (initial) + 1 (bought)
    # = 2 BTC at total cost 10000 + 5000 = 15000.
    buy_ticket = _buy_limit(state, BTCUSDT.name, Decimal('1'), Decimal('5000'))
    _, buy_order = state.add_order(buy_ticket)
    state.update_order(
        buy_order,
        status=OrderStatus.FILLED,
        updated_at=datetime.now(),
        id='buy-no-fee',
        filled_quantity=Decimal('1'),
        quote_quantity=Decimal('5000'),
        commission_asset=None,
        commission_quantity=None,
    )

    # SELL 1 BTC @ 10000 USDT, fee 0.01 BTC. The exchange takes
    # 1.01 BTC from balance; FIFO consumes oldest lots first.
    sell_ticket = _sell_limit(state, BTCUSDT.name, Decimal('1'), Decimal('10000'))
    _, sell_order = state.add_order(sell_ticket)
    state.update_order(
        sell_order,
        status=OrderStatus.FILLED,
        updated_at=datetime.now(),
        id='sell-fee-in-base',
        filled_quantity=Decimal('1'),
        quote_quantity=Decimal('10000'),
        commission_asset=BTC,
        commission_quantity=Decimal('0.01'),
    )

    node = state.record(time=datetime.now())
    # FIFO removes 1.01 BTC starting from the initial 1 BTC @ 10000:
    # the whole initial lot (1 BTC, cost 10000) + 0.01 from the new
    # lot (cost 5000/BTC) = 10000 + 50 = 10050.
    # Realized PnL = proceeds 10000 - cost_removed 10050 = -50.
    # The old code subtracted `cc` (0.01 * 10000 = 100) too, giving
    # -150; this asserts we no longer double-count.
    assert node.realized_pnl == Decimal('-50')
    btc_position = node.positions[BTC]
    # 2 BTC pre-sell - 1.01 BTC consumed = 0.99 BTC.
    assert btc_position.quantity == Decimal('0.99')
