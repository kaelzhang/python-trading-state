from trading_state.types import (
    SymbolInfo,
    Balance
)


def test_float_precision():
    info = SymbolInfo(
        'BTCUSDT',
        'BTC',
        'USDT',

        '0.010000000000',
        '1000000',
        '0.010000000000',

        '0.000001000000',
        '9000',
        '0.000001000000',

        '10.00000000'
    )

    assert info.min_quantity_step_precision == 6


def test_balance():
    balance = Balance('BTCUSDT', 0, 0)

    assert not balance.exists()
