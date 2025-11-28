from pytest import fixture
from typing import Dict

from trading_state.symbol import Symbol
from trading_state.enums import FeatureType, OrderType

from .fixtures import (
    get_symbols
)


@fixture
def test_symbols():
    return get_symbols()


def test_support_order_type(test_symbols: Dict[str, Symbol]):
    symbol_BTCUSDT = test_symbols['BTCUSDT']

    assert symbol_BTCUSDT.support(FeatureType.ORDER_TYPE, OrderType.LIMIT)
    assert symbol_BTCUSDT.support(FeatureType.ORDER_TYPE, OrderType.MARKET)
    assert symbol_BTCUSDT.support(FeatureType.ORDER_TYPE, OrderType.STOP_LOSS)
    assert symbol_BTCUSDT.support(FeatureType.ORDER_TYPE, OrderType.STOP_LOSS_LIMIT)
    assert symbol_BTCUSDT.support(FeatureType.ORDER_TYPE, OrderType.TAKE_PROFIT)
    assert symbol_BTCUSDT.support(FeatureType.ORDER_TYPE, OrderType.TAKE_PROFIT_LIMIT)

