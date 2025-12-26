from datetime import datetime
from decimal import Decimal

from trading_state import (
    TradingConfig,
    Balance,
    # PerformanceNode
)

from trading_state.common import (
    DECIMAL_ZERO
)

from .fixtures import (
    init_state,
    BTC,
    USDT,
    USDC,
    Z,
    ZUSDT,
    BTCUSDT,
    DEFAULT_CONFIG_KWARGS
)


def test_pnl():
    config = TradingConfig(
        benchmark_assets=(BTC,),
        **DEFAULT_CONFIG_KWARGS
    )
    state = init_state(config=config)

    state.set_symbol(ZUSDT)

    now = datetime.now()
    state.set_balances([
        Balance(Z, Decimal('1'), Decimal('0'), time=now),
    ])

    # Initial record
    # ---------------------------------------------------
    node = state.record(time=now)

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
    state.set_price(BTCUSDT.name, Decimal('20000'))

    now2 = datetime.now()
    node2 = state.record(time=now2)

    assert node2.positions[BTC].unrealized_pnl == Decimal('10000')
    assert node2.unrealized_pnl == Decimal('10000')
