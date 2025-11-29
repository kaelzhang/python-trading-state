from typing import Set
from decimal import Decimal


from trading_state import (
    Symbol,
    PrecisionFilter,
    PriceFilter,
    QuantityFilter,
    IcebergQuantityFilter,
    MarketQuantityFilter,
    TrailingDeltaFilter,
    NotionalFilter,
    FeatureType, OrderType, STPMode
)


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


Symbols = Set[Symbol]


def generate_symbols_from_exchange_info(exchange_info: dict) -> Symbols:
    """
    Generate symbols from the Binance exchange info.

    Args:
        exchange_info: (dict) the exchange info fetch from the API of https://api.binance.com/api/v3/exchangeInfo

    Returns:
        Dict[str, Symbol]
    """

    symbols = set[Symbol]()

    for symbol_info in exchange_info['symbols']:
        symbol = _get_symbol(symbol_info)
        symbols.add(symbol)

    return symbols


def _get_symbol(symbol_info: dict) -> Symbol:
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
        base_asset_precision=int(symbol_info['baseAssetPrecision']),
        quote_asset_precision=int(symbol_info['quoteAssetPrecision'])
    ))

    for filter in symbol_info['filters']:
        if filter['filterType'] == 'PRICE_FILTER':
            symbol.add_filter(PriceFilter(
                min_price=Decimal(filter['minPrice']),
                max_price=Decimal(filter['maxPrice']),
                tick_size=Decimal(filter['tickSize'])
            ))
        elif filter['filterType'] == 'LOT_SIZE':
            symbol.add_filter(QuantityFilter(
                min_quantity=Decimal(filter['minQty']),
                max_quantity=Decimal(filter['maxQty']),
                step_size=Decimal(filter['stepSize'])
            ))
        elif filter['filterType'] == 'ICEBERG_PARTS':
            symbol.add_filter(IcebergQuantityFilter(
                limit=int(filter['limit'])
            ))
        elif filter['filterType'] == 'MARKET_LOT_SIZE':
            symbol.add_filter(MarketQuantityFilter(
                min_quantity=Decimal(filter['minQty']),
                max_quantity=Decimal(filter['maxQty']),
                step_size=Decimal(filter['stepSize'])
            ))
        elif filter['filterType'] == 'TRAILING_DELTA':
            symbol.add_filter(TrailingDeltaFilter(
                min_trailing_above_delta=int(filter['minTrailingAboveDelta']),
                max_trailing_above_delta=int(filter['maxTrailingAboveDelta']),
                min_trailing_below_delta=int(filter['minTrailingBelowDelta']),
                max_trailing_below_delta=int(filter['maxTrailingBelowDelta'])
            ))
        elif filter['filterType'] == 'NOTIONAL':
            symbol.add_filter(NotionalFilter(
                min_notional=Decimal(filter['minNotional']),
                max_notional=Decimal(filter['maxNotional']),
                apply_min_to_market=bool(filter['applyMinToMarket']),
                apply_max_to_market=bool(filter['applyMaxToMarket']),
                avg_price_mins=int(filter['avgPriceMins'])
            ))

    return symbol
