from pytest import fixture

from trading_state.symbol import Symbol
from trading_state.enums import FeatureType, OrderType

from .fixtures import (
    binance_symbol_loader,
    load_symbol_info,
)


@fixture
def test_symbol():
    symbol_info = load_symbol_info()
    return binance_symbol_loader(symbol_info)


def test_support_order_type(test_symbol: Symbol):
    assert test_symbol.support(FeatureType.ORDER_TYPE, OrderType.LIMIT)
    assert test_symbol.support(FeatureType.ORDER_TYPE, OrderType.MARKET)
    assert test_symbol.support(FeatureType.ORDER_TYPE, OrderType.STOP_LOSS)
    assert test_symbol.support(FeatureType.ORDER_TYPE, OrderType.STOP_LOSS_LIMIT)
    assert test_symbol.support(FeatureType.ORDER_TYPE, OrderType.TAKE_PROFIT)
    assert test_symbol.support(FeatureType.ORDER_TYPE, OrderType.TAKE_PROFIT_LIMIT)

