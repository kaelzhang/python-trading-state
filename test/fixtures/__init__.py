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
    decode_exchange_info_response
)

def load_exchange_info() -> dict:
    with open(Path(__file__).parent / 'bn_exchange_info.json', 'r') as f:
        return json.load(f)


BTCUSDC = 'BTCUSDC'
BTCUSDT = 'BTCUSDT'
BTC = 'BTC'
USDT = 'USDT'
USDC = 'USDC'


def mock_get_avg_price(symbol_name: str, mins: int) -> Decimal:
    return Decimal('10000')


Symbols = Dict[str, Symbol]
symbols: Symbols = {}

def get_symbols() -> Symbols:
    global symbols

    if symbols:
        return symbols

    exchange_info = load_exchange_info()
    for symbol in decode_exchange_info_response(exchange_info):
        symbols[symbol.name] = symbol

    return symbols


def create_state() -> TradingState:
    return TradingState(
        config=TradingConfig(
            account_currency=USDT,
            alt_account_currencies=[USDC],
            context={
                'get_avg_price': mock_get_avg_price
            }
        )
    )


def init_symbols(state: TradingState) -> None:
    symbols = get_symbols()

    for symbol in symbols.values():
        state.set_symbol(symbol)

        # It is ok to set the symbol multiple times
        state.set_symbol(symbol)


def init_prices(state: TradingState) -> None:
    state.set_price(BTCUSDC, Decimal('10000'))
    assert state.set_price(BTCUSDT, Decimal('10000'))

    assert not state.set_price(BTCUSDT, Decimal('10000'))


def init_notional_limits(state: TradingState) -> None:
    # 10 BTC
    state.set_notional_limit(BTC, Decimal('-1'))
    state.set_notional_limit(BTC, None)
    state.set_notional_limit(BTC, Decimal('100000'))


def init_balances(state: TradingState) -> None:
    state.set_balances([
        Balance(BTC, Decimal('1'), Decimal('0')),
        Balance(USDC, Decimal('200000'), Decimal('0')),
        Balance(USDT, Decimal('200000'), Decimal('0')),
    ])


def init_state() -> TradingState:
    state = create_state()
    init_symbols(state)
    init_prices(state)
    init_notional_limits(state)
    init_balances(state)

    return state
