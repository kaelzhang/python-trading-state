from datetime import datetime
from decimal import Decimal

import pytest

from trading_state import (
    Balance,
    TradingConfig,
    TradingState,
    Symbol
)

from .fixtures import BTC


def test_balance():
    balance = Balance(
        asset=BTC,
        free=Decimal('1.0'),
        locked=Decimal('0.0')
    )

    assert repr(balance) == 'Balance(BTC free=1.0, locked=0.0)'


def test_dependency_manager_clear_when_symbols_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    config = TradingConfig(account_currency='USDT')
    state = TradingState(config)

    state.set_symbol(Symbol('XBTC', 'X', 'BTC'))
    state.set_symbol(Symbol('BTCUSDT', 'BTC', 'USDT'))

    state.set_balances([
        Balance('X', Decimal('1'), Decimal('0'))
    ])

    # First snapshot: both XBTC and BTCUSDT prices missing
    state.record(time=datetime(2024, 1, 1))

    # Only XBTC price is ready, dependency shrinks to BTCUSDT
    state.set_price('XBTC', Decimal('0.01'))
    state.record(time=datetime(2024, 1, 2))

    # Ensure monotonic times for cash flow updates
    import trading_state.state as state_module

    times = iter([
        datetime(2024, 1, 3, 0, 0, 0, 0),
        datetime(2024, 1, 3, 0, 0, 0, 1),
    ])

    class FakeDateTime:
        @staticmethod
        def now() -> datetime:
            return next(times)

    monkeypatch.setattr(state_module, 'datetime', FakeDateTime)

    # Now BTCUSDT becomes ready, cash flow is recorded and clear() removes asset symbols
    state.set_price('BTCUSDT', Decimal('30000'))

    dm = state._balances.not_ready_assets
    assert 'X' not in dm._asset_symbols
    assert 'X' in dm._symbol_assets.get('XBTC', set())

    original_clear = dm.clear

    def wrapped_clear(asset: str):
        assert asset not in dm._asset_symbols
        return original_clear(asset)

    monkeypatch.setattr(dm, 'clear', wrapped_clear)

    before_flows = len(state._perf._cash_flows)
    state.set_price('XBTC', Decimal('0.011'))
    assert len(state._perf._cash_flows) == before_flows + 1
