import json
from pathlib import Path

from trading_state.symbol import Symbol
from trading_state.filters import (
    PrecisionFilter,
    PriceFilter,
    QuantityFilter,
    IcebergQuantityFilter,
    MarketQuantityFilter,
    TrailingDeltaFilter,
    # PercentPriceBySideFilter,
    NotionalFilter
)
from trading_state.enums import FeatureType, OrderType, STPMode


def load_symbol_info() -> dict:
    with open(Path(__file__).parent / 'bn_symbol_info.json', 'r') as f:
        return json.load(f)


ORDER_TYPE_MAP = {
    'LIMIT': OrderType.LIMIT,
    # 'LIMIT_MAKER': OrderType.LIMIT_MAKER,
    'MARKET': OrderType.MARKET,
    'STOP_LOSS': OrderType.STOP_LOSS,
    'STOP_LOSS_LIMIT': OrderType.STOP_LOSS_LIMIT,
    'TAKE_PROFIT': OrderType.TAKE_PROFIT,
    'TAKE_PROFIT_LIMIT': OrderType.TAKE_PROFIT_LIMIT
}

STP_MODE_MAP = {
    'EXPIRE_TAKER': STPMode.EXPIRE_TAKER,
    'EXPIRE_MAKER': STPMode.EXPIRE_MAKER,
    'EXPIRE_BOTH': STPMode.EXPIRE_BOTH,
    'DECREMENT': STPMode.DECREMENT
}


def binance_symbol_loader(symbol_info: dict) -> Symbol:
    symbol = Symbol(
        name=symbol_info['symbol'],
        base_asset=symbol_info['baseAsset'],
        quote_asset=symbol_info['quoteAsset']
    )

    order_types = symbol_info['orderTypes']
    allow_post_only = 'LIMIT_MAKER' in order_types
    symbol.allow(FeatureType.POST_ONLY, allow_post_only)

    if allow_post_only:
        order_types.remove('LIMIT_MAKER')

    symbol.allow(FeatureType.ORDER_TYPE, [
        ORDER_TYPE_MAP[order_type]
        for order_type in symbol_info['orderTypes']
    ])

    symbol.allow(FeatureType.STP_MODE, [
        STP_MODE_MAP[stp_mode]
        for stp_mode in symbol_info['allowedSelfTradePreventionModes']
    ])

    symbol.add_filter(PrecisionFilter(
        base_asset_precision=symbol_info['baseAssetPrecision'],
        quote_asset_precision=symbol_info['quoteAssetPrecision']
    ))

    for filter in symbol_info['filters']:
        if filter['filterType'] == 'PRICE_FILTER':
            symbol.add_filter(PriceFilter(
                min_price=filter['minPrice'],
                max_price=filter['maxPrice'],
                tick_size=filter['tickSize']
            ))
        elif filter['filterType'] == 'LOT_SIZE':
            symbol.add_filter(QuantityFilter(
                min_quantity=filter['minQty'],
                max_quantity=filter['maxQty'],
                step_size=filter['stepSize']
            ))
        elif filter['filterType'] == 'ICEBERG_PARTS':
            symbol.add_filter(IcebergQuantityFilter(
                limit=filter['limit']
            ))
        elif filter['filterType'] == 'MARKET_LOT_SIZE':
            symbol.add_filter(MarketQuantityFilter(
                min_quantity=filter['minQty'],
                max_quantity=filter['maxQty'],
                step_size=filter['stepSize']
            ))
        elif filter['filterType'] == 'TRAILING_DELTA':
            symbol.add_filter(TrailingDeltaFilter(
                min_trailing_above_delta=filter['minTrailingAboveDelta'],
                max_trailing_above_delta=filter['maxTrailingAboveDelta'],
                min_trailing_below_delta=filter['minTrailingBelowDelta'],
                max_trailing_below_delta=filter['maxTrailingBelowDelta']
            ))
        # elif filter['filterType'] == 'PERCENT_PRICE_BY_SIDE':
        #     symbol.add_filter(PercentPriceBySideFilter(
        #         bid_multiplier_up=filter['bidMultiplierUp'],
        #         bid_multiplier_down=filter['bidMultiplierDown'],
        elif filter['filterType'] == 'NOTIONAL':
            symbol.add_filter(NotionalFilter(
                min_notional=filter['minNotional'],
                max_notional=filter['maxNotional'],
                apply_min_to_market=filter['applyMinToMarket'],
                apply_max_to_market=filter['applyMaxToMarket'],
                avg_price_mins=filter['avgPriceMins']
            ))

    return symbol
