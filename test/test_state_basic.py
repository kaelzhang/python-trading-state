import pytest
from decimal import Decimal

from trading_state import (
    TradingConfig,
    Balance,
    Symbol,
)

from trading_state.common import (
    DECIMAL_ZERO,
)

from .fixtures import (
    init_state,
    balance_time,
    USDT,
    USDC,
)


def test_config():
    with pytest.raises(ValueError, match='must not'):
        TradingConfig(
            account_currency=USDT,
            alt_account_currencies=(USDT, USDC),
        )

    config = TradingConfig(
        account_currency=USDT,
        alt_account_currencies=(USDC,),
    )

    assert config.account_currencies == (USDC, USDT)


def test_underlying_assets():
    state = init_state()

    AAPL = 'AAPL'

    state.set_symbol(Symbol(AAPL, AAPL, ''))
    state.set_price(AAPL, Decimal('100'))
    state.set_notional_limit(AAPL, Decimal('10000'))
    state.set_balances([
        Balance(AAPL, Decimal('10'), DECIMAL_ZERO, balance_time()),
    ])

    exc, exp = state.exposure(
        AAPL,
        include_unsettled_inflow=False,
        include_unsettled_outflow=False,
    )
    assert exc is None
    assert exp.ratio == Decimal('0.1')
