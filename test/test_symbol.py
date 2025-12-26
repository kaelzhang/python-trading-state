# from pytest import fixture
# from typing import Dict

# from trading_state.symbol import Symbol
from trading_state.enums import FeatureType, OrderType

from .fixtures import (
    get_symbols,
    init_state,
    USDT
)


def test_support_order_type():
    test_symbols = get_symbols()
    symbol_BTCUSDT = test_symbols['BTCUSDT']

    assert symbol_BTCUSDT.support(FeatureType.ORDER_TYPE, OrderType.LIMIT)
    assert symbol_BTCUSDT.support(FeatureType.ORDER_TYPE, OrderType.MARKET)
    assert symbol_BTCUSDT.support(FeatureType.ORDER_TYPE, OrderType.STOP_LOSS)
    assert symbol_BTCUSDT.support(FeatureType.ORDER_TYPE, OrderType.STOP_LOSS_LIMIT)
    assert symbol_BTCUSDT.support(FeatureType.ORDER_TYPE, OrderType.TAKE_PROFIT)
    assert symbol_BTCUSDT.support(FeatureType.ORDER_TYPE, OrderType.TAKE_PROFIT_LIMIT)


# def _test_valuation_path():
#     state = init_state()
#     longest = None
#     no_path = set()
#     symbols = state._symbols

#     for asset in symbols._assets:
#         path = symbols.valuation_path(asset)

#         if path is None:
#             no_path.add((asset, path))
#             continue

#         if longest is None or len(path) > len(longest[1]):
#             longest = (asset, path)

#     print('longest', longest)
#     print('no_path', no_path)

#     for asset, _ in no_path:
#         print(asset, symbols._base_asset_symbols[asset])


def test_valuation_path():
    state = init_state()

    path = state._symbols.valuation_path('LUN')

    step1 = path[0]
    step2 = path[1]

    assert step1.symbol.base_asset == 'LUN'
    assert (
        step1.symbol.quote_asset == 'ETH'
        # Two possible paths
        or step1.symbol.quote_asset == 'BTC'
    )
    assert step2.symbol.base_asset == step1.symbol.quote_asset
    assert step2.symbol.quote_asset == USDT

    assert step1.forward
    assert step2.forward
