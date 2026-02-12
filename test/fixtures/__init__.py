from typing import Optional
import json
from pathlib import Path
from typing import Dict
from decimal import Decimal

from trading_state import (
    Symbol,
    TradingState,
    TradingConfig,
    Balance,
    ExecutionStrategy,
    ExecutionStrategyResolver
)

from trading_state.binance import (
    decode_exchange_info_response
)

from stock_pandas import StockDataFrame
import pandas as pd


FIXTURE_ROOT = Path(__file__).parent


def read_fixture(name: str) -> str:
    return



def load_exchange_info() -> dict:
    with open(FIXTURE_ROOT / 'bn_exchange_info.json', 'r') as f:
        return json.load(f)


def get_stock() -> StockDataFrame:
    return StockDataFrame(
        pd.read_csv(FIXTURE_ROOT / 'stock.csv'),
        date_col='open_time'
    )


BTC = 'BTC'
ETH = 'ETH'
USDT = 'USDT'
USDC = 'USDC'
FDUSD = 'FDUSD'
X = 'X'
Y = 'Y'
Z = 'Z'

XY = Symbol(X + Y, X, Y)
ZY = Symbol(Z + Y, Z, Y)
ZUSDT = Symbol(Z + USDT, Z, USDT)
ZUSDC = Symbol(Z + USDC, Z, USDC)
BTCUSDC = Symbol(BTC + USDC, BTC, USDC)
BTCUSDT = Symbol(BTC + USDT, BTC, USDT)
BTCFDUSD = Symbol(BTC + FDUSD, BTC, FDUSD)
ETHUSDT = Symbol(ETH + USDT, ETH, USDT)


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


DEFAULT_CONFIG_KWARGS = dict(
    account_currency=USDT,
    alt_account_currencies=(USDC,),
    max_order_history_size=2,
    context={
        'get_avg_price': mock_get_avg_price
    }
)


def create_state(
    config: Optional[TradingConfig],
    default_execution_strategy: Optional[ExecutionStrategy] = None,
    execution_strategy_resolver: Optional[
        ExecutionStrategyResolver
    ] = None
) -> TradingState:
    if config is None:
        config = TradingConfig(
            **DEFAULT_CONFIG_KWARGS
        )

    state = TradingState(
        config=config,
        default_execution_strategy=default_execution_strategy,
        execution_strategy_resolver=execution_strategy_resolver
    )

    assert state.config == config
    return state


def init_symbols(state: TradingState) -> None:
    symbols = get_symbols()

    for symbol in symbols.values():
        state.set_symbol(symbol)

        # It is ok to set the symbol multiple times
        state.set_symbol(symbol)


def init_prices(state: TradingState) -> None:
    state.set_price(BTCUSDC.name, Decimal('10000'))
    assert state.set_price(BTCUSDT.name, Decimal('10000'))

    assert not state.set_price(BTCUSDT.name, Decimal('10000'))


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


def init_state(
    config: Optional[TradingConfig] = None,
    with_balances: bool = True,
    default_execution_strategy: Optional[ExecutionStrategy] = None,
    execution_strategy_resolver: Optional[
        ExecutionStrategyResolver
    ] = None
) -> TradingState:
    state = create_state(
        config=config,
        default_execution_strategy=default_execution_strategy,
        execution_strategy_resolver=execution_strategy_resolver
    )
    init_symbols(state)
    init_prices(state)
    init_notional_limits(state)

    if with_balances:
        init_balances(state)

    return state
