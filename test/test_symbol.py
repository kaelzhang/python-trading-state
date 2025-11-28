from pytest import fixture
from typing import Dict

from trading_state.symbol import Symbol
from trading_state.enums import FeatureType, OrderType

from .fixtures import (
    load_exchange_info,
    get_symbols
)


@fixture
def test_symbols():
    exchange_info = load_exchange_info()
    return get_symbols(exchange_info)


def test_support_order_type(test_symbols: Dict[str, Symbol]):
    symbol_BTCUSDT = test_symbols['BTCUSDT']

    assert symbol_BTCUSDT.support(FeatureType.ORDER_TYPE, OrderType.LIMIT)
    assert symbol_BTCUSDT.support(FeatureType.ORDER_TYPE, OrderType.MARKET)
    assert symbol_BTCUSDT.support(FeatureType.ORDER_TYPE, OrderType.STOP_LOSS)
    assert symbol_BTCUSDT.support(FeatureType.ORDER_TYPE, OrderType.STOP_LOSS_LIMIT)
    assert symbol_BTCUSDT.support(FeatureType.ORDER_TYPE, OrderType.TAKE_PROFIT)
    assert symbol_BTCUSDT.support(FeatureType.ORDER_TYPE, OrderType.TAKE_PROFIT_LIMIT)

