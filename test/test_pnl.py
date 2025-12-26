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
    Z,
    ZUSDT,
    # BTCUSDT,
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

    node = state.record(time=now)

    assert node.time == now
    assert node.realized_pnl == DECIMAL_ZERO
    assert node.unrealized_pnl == DECIMAL_ZERO

    print(node)
