import json
from pathlib import Path
from typing import Dict
from decimal import Decimal

from trading_state import (
    Symbol,
    TradingState,
    TradingConfig,
    Balance
)

from trading_state.binance import (
    generate_symbols_from_exchange_info
)

def load_exchange_info() -> dict:
    with open(Path(__file__).parent / 'bn_exchange_info.json', 'r') as f:
        return json.load(f)


BTCUSDC = 'BTCUSDC'
BTCUSDT = 'BTCUSDT'
BTC = 'BTC'
USDT = 'USDT'


def mock_get_avg_price(symbol_name: str, mins: int) -> Decimal:
    return Decimal('10000')


Symbols = Dict[str, Symbol]
symbols: Symbols = {}

def get_symbols() -> Symbols:
    global symbols

    if symbols:
        return symbols

    exchange_info = load_exchange_info()
    for symbol in generate_symbols_from_exchange_info(exchange_info):
        symbols[symbol.name] = symbol

    return symbols


def init_state() -> TradingState:
    symbols = get_symbols()

    state = TradingState(
        config=TradingConfig(
            numeraire=USDT,
            context={
                'get_avg_price': mock_get_avg_price
            }
        )
    )

    for symbol in symbols.values():
        state.set_symbol(symbol)

    state.set_price(BTCUSDC, Decimal('10000'))
    assert state.set_price(BTCUSDT, Decimal('10000'))

    assert not state.set_price(BTCUSDT, Decimal('10000'))

    # 10 BTC
    state.set_quota(BTC, Decimal('-1'))
    state.set_quota(BTC, None)
    state.set_quota(BTC, Decimal('100000'))

    state.set_balances([
        Balance(BTC, Decimal('1'), Decimal('0'))
    ])

    return state
