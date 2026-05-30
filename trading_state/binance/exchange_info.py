"""
Binance protocol adapter for exchange info.

`decode_exchange_info_response` returns `ValueOrException[Set[Symbol]]`.
Each symbol entry is built with its filter chain; malformed symbol
entries surface a single InvalidExchangeData.
"""
from typing import Set
from decimal import Decimal, InvalidOperation

from trading_state import (
    Symbol,
    PrecisionFilter,
    PriceFilter,
    QuantityFilter,
    IcebergQuantityFilter,
    MarketQuantityFilter,
    TrailingDeltaFilter,
    NotionalFilter,
    FeatureType, OrderType, STPMode,
    InvalidExchangeData,
)
from trading_state.common import DECIMAL_INF, ValueOrException


# Spot exchangeInfo `filters[]` entries that this adapter explicitly
# recognises but does not enforce — see `_attach_filter`. Verified
# 2026-05-30 against
# https://developers.binance.com/docs/binance-spot-api-docs/filters.
#
# - PERCENT_PRICE / PERCENT_PRICE_BY_SIDE need a market-price reference
#   (lastPrice avg over `avgPriceMins` minutes) that the filter layer
#   does not have access to under the current synchronous, kwarg-free
#   `Symbol.apply_filters` contract. Leaving them as silent-skip is the
#   conservative choice: the exchange still rejects out-of-range orders
#   server-side, so the worst local effect is a `-1013` or `-2010` from
#   the REST layer rather than a state-level corruption.
# - MAX_POSITION needs the symbol's current base-asset position to
#   enforce; state tracks balances per asset, not per symbol, so we do
#   not attempt this client-side.
# - MAX_NUM_ORDERS / MAX_NUM_ALGO_ORDERS / MAX_NUM_ICEBERG_ORDERS /
#   MAX_NUM_ORDER_AMENDS / MAX_NUM_ORDER_LISTS are pure count budgets.
#   trading-state is a passive store; counting open orders / amends /
#   list orders is the caller's responsibility above the state.
_SILENTLY_SKIPPED_FILTER_TYPES = frozenset({
    'PERCENT_PRICE',
    'PERCENT_PRICE_BY_SIDE',
    'MAX_POSITION',
    'MAX_NUM_ORDERS',
    'MAX_NUM_ALGO_ORDERS',
    'MAX_NUM_ICEBERG_ORDERS',
    'MAX_NUM_ORDER_AMENDS',
    'MAX_NUM_ORDER_LISTS',
})


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


def decode_exchange_info_response(
    exchange_info: dict,
) -> ValueOrException[Set[Symbol]]:
    """
    Decode the REST `GET /api/v3/exchangeInfo` response into a set of
    Symbol value objects, complete with their filter chain.

    Returns:
      (exception, None) — payload missing the top-level 'symbols' list
                          OR any per-symbol field could not be decoded.
      (None, Set[Symbol]) — successfully decoded.
    """
    if 'symbols' not in exchange_info:
        return (
            InvalidExchangeData(
                "exchange_info missing required field 'symbols'"
            ),
            None,
        )

    symbols: Set[Symbol] = set()
    for symbol_info in exchange_info['symbols']:
        exc, symbol = _build_symbol(symbol_info)
        if exc is not None:
            return exc, None
        symbols.add(symbol)

    return None, symbols


def _build_symbol(
    symbol_info: dict,
) -> ValueOrException[Symbol]:
    try:
        symbol = Symbol(
            name=symbol_info['symbol'],
            base_asset=symbol_info['baseAsset'],
            quote_asset=symbol_info['quoteAsset'],
        )
    except KeyError as e:
        return (
            InvalidExchangeData(
                f'exchange_info symbol entry missing field: {e}'
            ),
            None,
        )

    order_types = symbol_info.get('orderTypes', [])
    if 'LIMIT_MAKER' in order_types:
        order_types.remove('LIMIT_MAKER')
        symbol.allow(FeatureType.POST_ONLY, True)

    symbol.allow(FeatureType.ORDER_TYPE, [
        ORDER_TYPE_MAP[order_type]
        for order_type in order_types
    ])

    symbol.allow(FeatureType.STP_MODE, [
        STP_MODE_MAP[stp_mode]
        for stp_mode in symbol_info.get('allowedSelfTradePreventionModes', [])
    ])

    if symbol_info.get('icebergAllowed'):
        symbol.allow(FeatureType.ICEBERG, True)
    if symbol_info.get('ocoAllowed'):
        symbol.allow(FeatureType.OCO, True)
    if symbol_info.get('otoAllowed'):
        symbol.allow(FeatureType.OTO, True)
    if symbol_info.get('quoteOrderQtyMarketAllowed'):
        symbol.allow(FeatureType.QUOTE_ORDER_QUANTITY, True)
    if symbol_info.get('allowTrailingStop'):
        symbol.allow(FeatureType.TRAILING_STOP, True)
    if symbol_info.get('cancelReplaceAllowed'):
        symbol.allow(FeatureType.CANCEL_REPLACE, True)
    if symbol_info.get('amendAllowed'):
        symbol.allow(FeatureType.AMEND, True)
    if symbol_info.get('pegInstructionsAllowed'):
        symbol.allow(FeatureType.PEG_INSTRUCTIONS, True)
    if symbol_info.get('isSpotTradingAllowed'):
        symbol.allow(FeatureType.SPOT, True)
    if symbol_info.get('isMarginTradingAllowed'):
        symbol.allow(FeatureType.MARGIN, True)

    try:
        symbol.add_filter(PrecisionFilter(
            base_asset_precision=int(symbol_info['baseAssetPrecision']),
            quote_asset_precision=int(symbol_info['quoteAssetPrecision']),
        ))
    except (KeyError, ValueError, TypeError) as e:
        return (
            InvalidExchangeData(
                f'exchange_info precision fields invalid: {e}'
            ),
            None,
        )

    for filter_info in symbol_info.get('filters', []):
        exc = _attach_filter(symbol, filter_info)
        if exc is not None:
            return exc, None

    return None, symbol


def _attach_filter(symbol: Symbol, filter_info: dict):
    """Helper that builds the appropriate filter from a `filters[]`
    entry and attaches it to `symbol`. Returns None on success or
    an InvalidExchangeData on parse failure."""
    filter_type = filter_info.get('filterType')

    try:
        if filter_type == 'PRICE_FILTER':
            symbol.add_filter(PriceFilter(
                min_price=Decimal(filter_info['minPrice']),
                max_price=Decimal(filter_info['maxPrice']),
                tick_size=Decimal(filter_info['tickSize']),
            ))
        elif filter_type == 'LOT_SIZE':
            symbol.add_filter(QuantityFilter(
                min_quantity=Decimal(filter_info['minQty']),
                max_quantity=Decimal(filter_info['maxQty']),
                step_size=Decimal(filter_info['stepSize']),
            ))
        elif filter_type == 'ICEBERG_PARTS':
            symbol.add_filter(IcebergQuantityFilter(
                limit=int(filter_info['limit']),
            ))
        elif filter_type == 'MARKET_LOT_SIZE':
            symbol.add_filter(MarketQuantityFilter(
                min_quantity=Decimal(filter_info['minQty']),
                max_quantity=Decimal(filter_info['maxQty']),
                step_size=Decimal(filter_info['stepSize']),
            ))
        elif filter_type == 'TRAILING_DELTA':
            symbol.add_filter(TrailingDeltaFilter(
                min_trailing_above_delta=int(
                    filter_info['minTrailingAboveDelta']
                ),
                max_trailing_above_delta=int(
                    filter_info['maxTrailingAboveDelta']
                ),
                min_trailing_below_delta=int(
                    filter_info['minTrailingBelowDelta']
                ),
                max_trailing_below_delta=int(
                    filter_info['maxTrailingBelowDelta']
                ),
            ))
        elif filter_type == 'NOTIONAL':
            symbol.add_filter(NotionalFilter(
                min_notional=Decimal(filter_info['minNotional']),
                max_notional=Decimal(filter_info['maxNotional']),
                apply_min_to_market=bool(filter_info['applyMinToMarket']),
                apply_max_to_market=bool(filter_info['applyMaxToMarket']),
                avg_price_mins=int(filter_info['avgPriceMins']),
            ))
        elif filter_type == 'MIN_NOTIONAL':
            # Legacy single-cap form (predates 2022-04-04 NOTIONAL).
            # Some older symbols still emit it instead of NOTIONAL. We
            # alias it onto NotionalFilter with the max set to +Inf so
            # the lower bound is enforced identically and the upper
            # bound is a no-op.
            symbol.add_filter(NotionalFilter(
                min_notional=Decimal(filter_info['minNotional']),
                max_notional=DECIMAL_INF,
                apply_min_to_market=bool(
                    filter_info.get('applyToMarket', False)
                ),
                apply_max_to_market=False,
                avg_price_mins=int(filter_info.get('avgPriceMins', 0)),
            ))
        elif filter_type in _SILENTLY_SKIPPED_FILTER_TYPES:
            # Recognised but not enforced — see
            # _SILENTLY_SKIPPED_FILTER_TYPES for the per-type rationale.
            pass
        # Unknown filter types are skipped silently: Binance adds new
        # filter types over time and we'd rather no-op than fail.
    except (KeyError, InvalidOperation, ValueError, TypeError) as e:
        return InvalidExchangeData(
            f'exchange_info filter {filter_type!r} invalid field: {e}'
        )
    return None
