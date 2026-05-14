from datetime import datetime
from decimal import Decimal

from trading_state import (
    TradingConfig,
    Balance,
    CashFlow,
    LimitOrderTicket,
    OrderSide,
    OrderStatus,
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
    assert exposure == Decimal('0.4')

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
