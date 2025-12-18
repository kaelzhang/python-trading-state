import pytest
from decimal import Decimal

from trading_state import (
    TradingConfig,
    Balance
)

from trading_state.common import (
    DECIMAL_ZERO
)

from .fixtures import (
    init_state,
    USDT,
    USDC
)


def test_config():
    with pytest.raises(ValueError, match='must not'):
        TradingConfig(
            account_currency=USDT,
            alt_account_currencies=(USDT, USDC)
        )

    config = TradingConfig(
        account_currency=USDT,
        alt_account_currencies=(USDC,)
    )

    assert config.account_currencies == (USDC, USDT)


def test_trading_state_basic():
    state = init_state()

    assert state.config.account_currencies == (USDC, USDT)

    balances = state.get_balances()
    balances[USDT] = Balance(USDT, DECIMAL_ZERO, DECIMAL_ZERO)

    assert state.get_balance(USDT).free == Decimal('200000')

    state.set_balances([
        Balance(USDT, Decimal('100000'), DECIMAL_ZERO)
    ], delta=True)

    assert state.get_balance(USDT).free == Decimal('300000')
