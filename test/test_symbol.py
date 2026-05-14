from decimal import Decimal

from trading_state import (
    FeatureType,
    OrderType,
)
from trading_state.symbol import ValuationPathStep

from .fixtures import (
    get_symbols,
    init_state,
    USDT,
    X,
    XY,
    ZY,
    ZUSDC,
    ZUSDT
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


def test_order_type_values_match_binance_spot():
    # Binance Spot uses 'STOP_LOSS' / 'TAKE_PROFIT'; Futures uses
    # 'STOP_MARKET'. Encoder will serialise `.value` straight back to
    # the wire, so a regression here breaks order placement silently.
    assert OrderType.LIMIT.value == 'LIMIT'
    assert OrderType.MARKET.value == 'MARKET'
    assert OrderType.STOP_LOSS.value == 'STOP_LOSS'
    assert OrderType.STOP_LOSS_LIMIT.value == 'STOP_LOSS_LIMIT'
    assert OrderType.TAKE_PROFIT.value == 'TAKE_PROFIT'
    assert OrderType.TAKE_PROFIT_LIMIT.value == 'TAKE_PROFIT_LIMIT'


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


def test_valuation_path_primary_account_currency():
    state = init_state()

    state.set_symbol(XY)
    state.set_symbol(ZY)
    state.set_symbol(ZUSDC)
    state.set_symbol(ZUSDT)

    path = state._symbols.valuation_path(X)
    assert path == [
        ValuationPathStep(XY, True),
        ValuationPathStep(ZY, False),
        ValuationPathStep(ZUSDT, True),
    ]

def test_set_symbol_invalidates_cached_unreachable_path():
    """A previously cached "no path" verdict must be reconsidered once
    a new symbol that completes the path is registered."""
    state = init_state()

    state.set_symbol(XY)
    state.set_symbol(ZY)

    # No symbol yet connects Z to an account currency → no path.
    assert state._symbols.valuation_path(X) is None

    # Register the missing leg; the cache must be reset by set_symbol
    # so the path becomes discoverable without manual cache busting.
    state.set_symbol(ZUSDT)
    path = state._symbols.valuation_path(X)
    assert path == [
        ValuationPathStep(XY, True),
        ValuationPathStep(ZY, False),
        ValuationPathStep(ZUSDT, True),
    ]


def test_valuation_price_info():
    state = init_state()

    state.set_symbol(XY)
    state.set_symbol(ZY)
    state.set_symbol(ZUSDC)

    price, dependencies = state._symbols.valuation_price_info(X)
    assert price is None
    assert dependencies == {XY.name, ZY.name, ZUSDC.name}

    state.set_price(ZY.name, Decimal('2'))

    price, dependencies = state._symbols.valuation_price_info(X)
    assert price is None
    assert dependencies == {XY.name, ZUSDC.name}

    state.set_price(XY.name, Decimal('10000'))
    state.set_price(ZUSDC.name, Decimal('0.5'))

    price, dependencies = state._symbols.valuation_price_info(X)
    assert price == Decimal('2500')
    assert dependencies is None
