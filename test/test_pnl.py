from datetime import datetime
from decimal import Decimal

from trading_state import (
    TradingConfig,
    Balance,
    CashFlow,
    OrderSide,
    OrderStatus,
    # PerformanceSnapshot
)

from trading_state.common import (
    DECIMAL_ZERO
)

from .fixtures import (
    init_state,
    BTC,
    ETH,
    USDT,
    USDC,
    Z,
    X,
    ZUSDT,
    BTCUSDT,
    ETHUSDT,
    DEFAULT_CONFIG_KWARGS
)


def test_pnl():
    config = TradingConfig(
        benchmark_assets=(BTC, ETH),
        **DEFAULT_CONFIG_KWARGS
    )
    state = init_state(config=config)

    state.set_symbol(ZUSDT)

    now = datetime.now()

    # Cash flow before balance is set and before init, skip
    cash_flow_z = CashFlow(Z, Decimal('0.5'), now)
    state.set_cash_flow(cash_flow_z)

    state.set_balances([
        # invalid balance, asset not found
        Balance('invalid', Decimal('1'), Decimal('0'), now),
        # Zero balance, which is actually invalid
        Balance(X, DECIMAL_ZERO, DECIMAL_ZERO, now)
    ])

    # Initial record
    # BTC: $10000
    # ---------------------------------------------------
    node = state.record(time=now)

    # Cash flow before balance is set, skip
    state.set_cash_flow(cash_flow_z)

    state.set_balances([
        Balance(Z, Decimal('1'), Decimal('0'), time=now),
    ])

    # Set the same cash flow before price is ready, skip
    state.set_cash_flow(cash_flow_z)

    assert node.time == now
    assert node.realized_pnl == DECIMAL_ZERO
    assert node.unrealized_pnl == DECIMAL_ZERO

    assert BTC in node.positions

    # Price not ready yet
    assert Z not in node.positions

    # Account currencies
    assert USDT not in node.positions
    assert USDC not in node.positions

    BTC_position = node.positions[BTC]
    assert BTC_position.quantity == Decimal('1')
    assert BTC_position.cost == Decimal('10000')
    assert BTC_position.valuation_price == Decimal('10000')

    assert node.unrealized_pnl == DECIMAL_ZERO

    # Price increased => unrealized PnL increased
    # ---------------------------------------------------
    price = Decimal('20000')
    state.set_price(BTCUSDT.name, price)

    now2 = datetime.now()
    node2 = state.record(time=now2)

    assert node2.positions[BTC].unrealized_pnl == Decimal('10000')
    assert node2.unrealized_pnl == Decimal('10000')

    now3 = datetime.now()

    # Cash Flow of BTC, + $20000
    # ---------------------------------------------------
    state.set_balances([
        Balance(BTC, Decimal('1'), Decimal('0'), time=now3)
    ], delta=True)

    cash_flow = CashFlow(BTC, Decimal('1'), now3)
    state.set_cash_flow(cash_flow)

    # Set the same cash flow multiple times
    state.set_cash_flow(cash_flow)

    assert 'CashFlow(BTC' in repr(cash_flow)

    node3 = state.record(time=now3)

    BTC_position3 = node3.positions[BTC]

    assert BTC_position3.quantity == Decimal('2')
    assert BTC_position3.cost == Decimal('30000')
    assert node3.unrealized_pnl == Decimal('10000')
    assert node3.net_cash_flow == Decimal('20000')

    _, exposure = state.exposure(BTC)

    assert exposure == Decimal('0.4')

    # Expect a buy order
    # ---------------------------------------------------
    state.expect(
        BTCUSDT.name,
        exposure=Decimal('0.5'),
        price=price,
        use_market_order=False
    )

    orders, _ = state.get_orders()
    assert len(orders) == 1

    order = orders.pop()
    ticket = order.ticket
    assert ticket.symbol.name == BTCUSDT.name
    assert ticket.side == OrderSide.BUY
    assert ticket.quantity == Decimal('0.5')
    assert ticket.price == Decimal('20000')

    now3 = datetime.now()

    state.update_order(
        order,
        filled_quantity=Decimal('0.5'),
        # The USDT used is less than the expected quote quantity
        quote_quantity=Decimal('5000'),
        status=OrderStatus.FILLED,
    )

    # But price decreased significantly
    # ---------------------------------------------------
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

    # Set price of Z
    # The balance of Z should be treated as a cash flow
    # ---------------------------------------------------
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
    # ---------------------------------------------------
    state.expect(
        BTCUSDT.name,
        exposure=Decimal('0.025'),
        price=price2,
        use_market_order=False
    )

    orders, _ = state.get_orders()
    assert len(orders) == 1

    order = orders.pop()

    state.update_order(
        order,
        filled_quantity=Decimal('2'),
        # The USDT used is less than the expected quote quantity
        quote_quantity=Decimal('15000'),
        status=OrderStatus.FILLED,
    )

    now7 = datetime.now()
    node7 = state.record(time=now7)

    assert node7.realized_pnl == Decimal('-18000')
    assert node7.unrealized_pnl == Decimal('-2500')

    # print(state._perf._position_tracker._positions._data)

    # Sell more
    # ---------------------------------------------------
    state.expect(
        BTCUSDT.name,
        exposure=Decimal('0.02'),
        price=price2,
        use_market_order=False
    )

    orders, _ = state.get_orders()
    assert len(orders) == 1
    order = orders.pop()

    state.update_order(
        order,
        filled_quantity=Decimal('0.1'),
        # The USDT used is less than the expected quote quantity
        quote_quantity=Decimal('500'),
        status=OrderStatus.FILLED,
    )

    now8 = datetime.now()
    node8 = state.record(time=now8)

    assert node8.realized_pnl == Decimal('-18500')

    # Decrease ETH
    # ---------------------------------------------------
    state.set_cash_flow(
        CashFlow(Z, Decimal('-2'), datetime.now())
    )

    # This should not happen, however we should handle it to prevent dirty data
    state.set_cash_flow(
        CashFlow(Z, Decimal('-1'), datetime.now())
    )

    # Zero cash flow, which is invalid
    state.set_cash_flow(
        CashFlow(Z, Decimal('0'), datetime.now())
    )

    now9 = datetime.now()
    node9 = state.record(time=now9)

    assert node9.positions[Z].quantity == Decimal('0')
