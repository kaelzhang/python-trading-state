from datetime import datetime
from decimal import Decimal

import pytest

from trading_state import (
    FeatureType,
    InvalidExchangeData,
    OrderSide,
    OrderStatus,
    OrderType,
    STPMode,
    TimeInForce,
)
from trading_state.binance.balance import (
    decode_account_info_response,
    decode_account_update_event,
    decode_balance_update_event,
)
from trading_state.binance.common import timestamp_to_datetime
from trading_state.binance.exchange_info import decode_exchange_info_response
from trading_state.binance.order import (
    _decode_order_status,
    decode_order_create_response,
    decode_order_update_event,
    encode_order_request,
)
from trading_state.filters import (
    IcebergQuantityFilter,
    MarketQuantityFilter,
    NotionalFilter,
    PrecisionFilter,
    PriceFilter,
    QuantityFilter,
    TrailingDeltaFilter,
)
from trading_state.order_ticket import (
    LimitOrderTicket,
    MarketOrderTicket,
    StopLossOrderTicket,
)
from trading_state.symbol import Symbol
from trading_state.enums import MarketQuantityType


def test_timestamp_to_datetime_ms_epoch():
    timestamp_ms = 1499827319559
    expected = datetime.fromtimestamp(timestamp_ms / 1000)
    assert timestamp_to_datetime(timestamp_ms) == expected


def test_decode_account_update_event():
    payload = {
        "u": 1564034571073,
        "B": [
            {"a": "ETH", "f": "10000.000000", "l": "0.000000"},
            {"a": "BTC", "f": "1.23456789", "l": "0.000000"},
        ],
    }
    exc, balances = decode_account_update_event(payload)
    assert exc is None
    assert len(balances) == 2
    assets = {balance.asset for balance in balances}
    assert assets == {"ETH", "BTC"}
    for balance in balances:
        assert balance.time == timestamp_to_datetime(payload["u"])


def test_decode_account_update_event_missing_field():
    exc, value = decode_account_update_event({"B": []})
    assert value is None
    assert isinstance(exc, InvalidExchangeData)
    assert 'event_time' in str(exc)


def test_decode_account_update_event_negative_free():
    exc, value = decode_account_update_event({
        "u": 1,
        "B": [{"a": "ETH", "f": "-1.0", "l": "0"}],
    })
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


def test_decode_account_update_event_invalid_decimal():
    exc, value = decode_account_update_event({
        "u": 1,
        "B": [{"a": "ETH", "f": "not-a-number", "l": "0"}],
    })
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


@pytest.mark.parametrize('drop_key', ['B', 'a', 'f', 'l'])
def test_decode_account_update_event_missing_inner_field(drop_key):
    base = {
        "u": 1,
        "B": [{"a": "ETH", "f": "1", "l": "0"}],
    }
    if drop_key == 'B':
        del base['B']
    else:
        del base['B'][0][drop_key]
    exc, value = decode_account_update_event(base)
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


def test_decode_account_update_event_invalid_locked():
    exc, value = decode_account_update_event({
        "u": 1,
        "B": [{"a": "ETH", "f": "1", "l": "-2"}],
    })
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


def test_decode_balance_update_event():
    payload = {"a": "BTC", "d": "100.00000000", "T": 1573200697068}
    exc, cash_flow = decode_balance_update_event(payload)
    assert exc is None
    assert cash_flow.asset == "BTC"
    assert cash_flow.quantity == Decimal("100.00000000")
    assert cash_flow.time == timestamp_to_datetime(payload["T"])


def test_decode_balance_update_event_allows_negative_delta():
    """d is a signed delta — negative is legal (withdrawal)."""
    exc, cash_flow = decode_balance_update_event(
        {"a": "BTC", "d": "-5.5", "T": 1573200697068},
    )
    assert exc is None
    assert cash_flow.quantity == Decimal("-5.5")


@pytest.mark.parametrize('drop_key', ['a', 'd', 'T'])
def test_decode_balance_update_event_missing_field(drop_key):
    payload = {"a": "BTC", "d": "1", "T": 1}
    del payload[drop_key]
    exc, value = decode_balance_update_event(payload)
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


def test_decode_balance_update_event_invalid_decimal():
    exc, value = decode_balance_update_event({
        "a": "BTC", "d": "not-a-decimal", "T": 1,
    })
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


def test_decode_account_info_response():
    account_info = {
        "updateTime": 123456789,
        "balances": [
            {"asset": "BTC", "free": "4723846.89208129", "locked": "0.00000000"},
            {"asset": "LTC", "free": "4763368.68006011", "locked": "0.00000000"},
        ],
    }
    exc, balances = decode_account_info_response(account_info)
    assert exc is None
    assets = {balance.asset for balance in balances}
    assert assets == {"BTC", "LTC"}
    for balance in balances:
        assert balance.time == timestamp_to_datetime(account_info["updateTime"])


def test_decode_account_info_response_missing_field():
    exc, value = decode_account_info_response({"balances": []})
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


def test_decode_account_info_response_negative_locked():
    exc, value = decode_account_info_response({
        "updateTime": 1,
        "balances": [{"asset": "BTC", "free": "1", "locked": "-1"}],
    })
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


@pytest.mark.parametrize('drop_key', ['balances', 'asset', 'free', 'locked'])
def test_decode_account_info_response_missing_inner_field(drop_key):
    base = {
        "updateTime": 1,
        "balances": [{"asset": "BTC", "free": "1", "locked": "0"}],
    }
    if drop_key == 'balances':
        del base['balances']
    else:
        del base['balances'][0][drop_key]
    exc, value = decode_account_info_response(base)
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


def test_decode_account_info_response_negative_free():
    exc, value = decode_account_info_response({
        "updateTime": 1,
        "balances": [{"asset": "BTC", "free": "-1", "locked": "0"}],
    })
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


def test_decode_exchange_info_response():
    exchange_info = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "baseAssetPrecision": 8,
                "quoteAssetPrecision": 8,
                "orderTypes": [
                    "LIMIT",
                    "LIMIT_MAKER",
                    "MARKET",
                    "STOP_LOSS",
                    "STOP_LOSS_LIMIT",
                    "TAKE_PROFIT",
                    "TAKE_PROFIT_LIMIT",
                ],
                "allowedSelfTradePreventionModes": [
                    "EXPIRE_TAKER",
                    "EXPIRE_MAKER",
                    "EXPIRE_BOTH",
                    "DECREMENT",
                ],
                "icebergAllowed": True,
                "ocoAllowed": True,
                "otoAllowed": True,
                "quoteOrderQtyMarketAllowed": True,
                "allowTrailingStop": True,
                "cancelReplaceAllowed": True,
                "amendAllowed": True,
                "pegInstructionsAllowed": True,
                "isSpotTradingAllowed": True,
                "isMarginTradingAllowed": True,
                "filters": [
                    {
                        "filterType": "PRICE_FILTER",
                        "minPrice": "0.00000100",
                        "maxPrice": "100000.00000000",
                        "tickSize": "0.00000100",
                    },
                    {
                        "filterType": "LOT_SIZE",
                        "minQty": "0.00100000",
                        "maxQty": "100000.00000000",
                        "stepSize": "0.00100000",
                    },
                    {"filterType": "ICEBERG_PARTS", "limit": 10},
                    {
                        "filterType": "MARKET_LOT_SIZE",
                        "minQty": "0.00100000",
                        "maxQty": "100000.00000000",
                        "stepSize": "0.00100000",
                    },
                    {
                        "filterType": "TRAILING_DELTA",
                        "minTrailingAboveDelta": 10,
                        "maxTrailingAboveDelta": 2000,
                        "minTrailingBelowDelta": 10,
                        "maxTrailingBelowDelta": 2000,
                    },
                    {
                        "filterType": "NOTIONAL",
                        "minNotional": "10.00000000",
                        "maxNotional": "10000.00000000",
                        "applyMinToMarket": False,
                        "applyMaxToMarket": False,
                        "avgPriceMins": 5,
                    },
                ],
            }
        ]
    }

    exc, symbols = decode_exchange_info_response(exchange_info)
    assert exc is None
    assert len(symbols) == 1
    symbol = next(iter(symbols))
    assert symbol.name == "BTCUSDT"
    assert symbol.base_asset == "BTC"
    assert symbol.quote_asset == "USDT"

    assert symbol.support(FeatureType.POST_ONLY)
    assert symbol.support(FeatureType.ORDER_TYPE, OrderType.LIMIT)
    assert symbol.support(FeatureType.ORDER_TYPE, OrderType.MARKET)
    assert symbol.support(FeatureType.ORDER_TYPE, OrderType.STOP_LOSS)
    assert symbol.support(FeatureType.ORDER_TYPE, OrderType.STOP_LOSS_LIMIT)
    assert symbol.support(FeatureType.ORDER_TYPE, OrderType.TAKE_PROFIT)
    assert symbol.support(FeatureType.ORDER_TYPE, OrderType.TAKE_PROFIT_LIMIT)
    assert symbol.support(FeatureType.STP_MODE, STPMode.EXPIRE_TAKER)
    assert symbol.support(FeatureType.STP_MODE, STPMode.EXPIRE_MAKER)
    assert symbol.support(FeatureType.STP_MODE, STPMode.EXPIRE_BOTH)
    assert symbol.support(FeatureType.STP_MODE, STPMode.DECREMENT)

    assert symbol.support(FeatureType.ICEBERG)
    assert symbol.support(FeatureType.OCO)
    assert symbol.support(FeatureType.OTO)
    assert symbol.support(FeatureType.QUOTE_ORDER_QUANTITY)
    assert symbol.support(FeatureType.TRAILING_STOP)
    assert symbol.support(FeatureType.CANCEL_REPLACE)
    assert symbol.support(FeatureType.AMEND)
    assert symbol.support(FeatureType.PEG_INSTRUCTIONS)
    assert symbol.support(FeatureType.SPOT)
    assert symbol.support(FeatureType.MARGIN)

    filter_types = {type(filter) for filter in symbol._filters}
    assert PrecisionFilter in filter_types
    assert PriceFilter in filter_types
    assert QuantityFilter in filter_types
    assert IcebergQuantityFilter in filter_types
    assert MarketQuantityFilter in filter_types
    assert TrailingDeltaFilter in filter_types
    assert NotionalFilter in filter_types


def test_decode_exchange_info_response_missing_symbols():
    exc, value = decode_exchange_info_response({})
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


def test_decode_exchange_info_response_symbol_missing_field():
    exc, value = decode_exchange_info_response({"symbols": [{"symbol": "X"}]})
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


def test_decode_exchange_info_response_precision_invalid():
    exc, value = decode_exchange_info_response({
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "baseAssetPrecision": "not-an-int",
                "quoteAssetPrecision": 8,
            },
        ],
    })
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


def test_decode_exchange_info_response_filter_invalid():
    exc, value = decode_exchange_info_response({
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "baseAssetPrecision": 8,
                "quoteAssetPrecision": 8,
                "filters": [
                    {"filterType": "PRICE_FILTER"},  # missing fields
                ],
            },
        ],
    })
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


def test_encode_order_request_limit():
    symbol = Symbol("BTCUSDT", "BTC", "USDT")
    ticket = LimitOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal("0.5"),
        price=Decimal("100"),
        time_in_force=TimeInForce.GTC,
    )
    exc, payload = encode_order_request(ticket)
    assert exc is None
    assert payload["symbol"] == "BTCUSDT"
    assert payload["side"] == OrderSide.BUY
    assert payload["type"] == OrderType.LIMIT
    assert payload["timeInForce"] == TimeInForce.GTC
    assert payload["quantity"] == Decimal("0.5")
    assert payload["price"] == "100"
    assert payload["newOrderRespType"] == "FULL"


def test_encode_order_request_market_base_and_quote():
    symbol = Symbol("BTCUSDT", "BTC", "USDT")
    base_ticket = MarketOrderTicket(
        symbol=symbol,
        side=OrderSide.SELL,
        quantity=Decimal("1"),
        quantity_type=MarketQuantityType.BASE,
        estimated_price=Decimal("10000"),
    )
    exc, base_payload = encode_order_request(base_ticket)
    assert exc is None
    assert "quantity" in base_payload
    assert "quoteOrderQty" not in base_payload

    quote_ticket = MarketOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal("1000"),
        quantity_type=MarketQuantityType.QUOTE,
        estimated_price=Decimal("10000"),
    )
    exc, quote_payload = encode_order_request(quote_ticket)
    assert exc is None
    assert "quoteOrderQty" in quote_payload
    assert "quantity" not in quote_payload


def test_encode_order_request_unsupported():
    symbol = Symbol("BTCUSDT", "BTC", "USDT")
    ticket = StopLossOrderTicket(
        symbol=symbol,
        side=OrderSide.SELL,
        quantity=Decimal("1"),
        stop_price=Decimal("9000"),
    )
    exc, payload = encode_order_request(ticket)
    assert payload is None
    assert isinstance(exc, InvalidExchangeData)
    assert 'unsupported' in str(exc).lower()


def test_decode_order_status():
    assert _decode_order_status("FILLED")[1] == OrderStatus.FILLED
    assert _decode_order_status("CANCELED")[1] == OrderStatus.CANCELLED
    assert _decode_order_status("PARTIALLY_FILLED")[1] == OrderStatus.CREATED
    assert _decode_order_status("NEW")[1] == OrderStatus.CREATED
    assert _decode_order_status("EXPIRED")[1] == OrderStatus.CANCELLED
    assert _decode_order_status("REJECTED")[1] == OrderStatus.REJECTED

    exc, value = _decode_order_status("UNKNOWN")
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


def test_decode_order_create_response_without_fills():
    response = {
        "status": "FILLED",
        "clientOrderId": "6gCrw2kRUAF9CvJDGP16IP",
        "transactTime": 1507725176595,
        "executedQty": "10.00000000",
        "cummulativeQuoteQty": "10.00000000",
        "fills": [],
    }
    exc, updates = decode_order_create_response(response)
    assert exc is None
    assert updates["status"] == OrderStatus.FILLED
    assert updates["id"] == response["clientOrderId"]
    assert updates["created_at"] == datetime.fromtimestamp(
        response["transactTime"] / 1000
    )
    assert updates["filled_quantity"] == Decimal("10.00000000")
    assert updates["quote_quantity"] == Decimal("10.00000000")
    assert "commission_asset" not in updates
    assert "commission_quantity" not in updates


def test_decode_order_create_response_with_fills():
    response = {
        "status": "FILLED",
        "clientOrderId": "abc123",
        "transactTime": 1507725176595,
        "executedQty": "2.00000000",
        "cummulativeQuoteQty": "10.00000000",
        "fills": [
            {"commission": "0.10000000", "commissionAsset": "USDT"},
            {"commission": "0.20000000", "commissionAsset": "USDT"},
        ],
    }
    exc, updates = decode_order_create_response(response)
    assert exc is None
    assert updates["commission_asset"] == "USDT"
    assert updates["commission_quantity"] == Decimal("0.30000000")


def test_decode_order_create_response_missing_field():
    exc, value = decode_order_create_response({"status": "FILLED"})
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


def test_decode_order_create_response_unknown_status():
    exc, value = decode_order_create_response({
        "status": "WTF",
        "clientOrderId": "x",
        "transactTime": 1,
        "executedQty": "0",
        "cummulativeQuoteQty": "0",
    })
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


def test_decode_order_create_response_commission_without_asset():
    exc, value = decode_order_create_response({
        "status": "FILLED",
        "clientOrderId": "x",
        "transactTime": 1,
        "executedQty": "1",
        "cummulativeQuoteQty": "10",
        "fills": [
            {"commission": "0.5"},  # no commissionAsset
        ],
    })
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


def test_decode_order_create_response_negative_qty():
    exc, value = decode_order_create_response({
        "status": "FILLED",
        "clientOrderId": "x",
        "transactTime": 1,
        "executedQty": "-1",
        "cummulativeQuoteQty": "10",
    })
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


def test_decode_order_create_response_invalid_decimal():
    exc, value = decode_order_create_response({
        "status": "FILLED",
        "clientOrderId": "x",
        "transactTime": 1,
        "executedQty": "1",
        "cummulativeQuoteQty": "not-a-decimal",
    })
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


@pytest.mark.parametrize(
    "status_raw,expected_status",
    [
        ("CANCELED", OrderStatus.CANCELLED),
        ("FILLED", OrderStatus.FILLED),
        ("EXPIRED", OrderStatus.CANCELLED),
        ("REJECTED", OrderStatus.REJECTED),
        ("NEW", OrderStatus.CREATED),
        ("PARTIALLY_FILLED", OrderStatus.CREATED),
    ],
)
def test_decode_order_update_event(status_raw, expected_status):
    payload = {
        "X": status_raw,
        "c": "client-1",
        "z": "1.00000000",
        "Z": "100.00000000",
        "N": None,
        "n": "0.00000000",
        "T": 1499405658657,
    }
    exc, decoded = decode_order_update_event(payload)
    assert exc is None
    client_id, updates = decoded
    assert client_id == "client-1"
    assert updates["filled_quantity"] == Decimal("1.00000000")
    assert updates["quote_quantity"] == Decimal("100.00000000")
    assert updates["commission_asset"] is None
    assert updates["commission_quantity"] == Decimal("0.00000000")
    assert updates["updated_at"] == timestamp_to_datetime(payload["T"])
    assert updates["status"] is expected_status


def test_decode_order_update_event_missing_field():
    exc, value = decode_order_update_event({"X": "FILLED"})
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


def test_decode_order_update_event_unknown_status():
    exc, value = decode_order_update_event({
        "c": "c",
        "X": "UNKNOWN",
        "z": "0",
        "Z": "0",
        "n": "0",
        "T": 1,
    })
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


def test_decode_order_update_event_commission_without_asset():
    exc, value = decode_order_update_event({
        "c": "c",
        "X": "FILLED",
        "z": "1",
        "Z": "10",
        "N": None,
        "n": "0.5",
        "T": 1,
    })
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


@pytest.mark.parametrize('drop_key', ['c', 'X', 'z', 'Z', 'n', 'T'])
def test_decode_order_update_event_missing_each_field(drop_key):
    payload = {
        "c": "c",
        "X": "FILLED",
        "z": "1",
        "Z": "10",
        "N": "USDT",
        "n": "0",
        "T": 1,
    }
    del payload[drop_key]
    exc, value = decode_order_update_event(payload)
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


@pytest.mark.parametrize('field_key,new_value', [
    ('z', 'not-a-decimal'),
    ('Z', 'not-a-decimal'),
    ('n', '-0.5'),
])
def test_decode_order_update_event_invalid_decimals(field_key, new_value):
    payload = {
        "c": "c",
        "X": "FILLED",
        "z": "1",
        "Z": "10",
        "N": "USDT",
        "n": "0",
        "T": 1,
    }
    payload[field_key] = new_value
    exc, value = decode_order_update_event(payload)
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


@pytest.mark.parametrize('drop_key', [
    'status', 'clientOrderId', 'transactTime',
    'executedQty', 'cummulativeQuoteQty',
])
def test_decode_order_create_response_missing_each_field(drop_key):
    payload = {
        "status": "FILLED",
        "clientOrderId": "x",
        "transactTime": 1,
        "executedQty": "1",
        "cummulativeQuoteQty": "10",
    }
    del payload[drop_key]
    exc, value = decode_order_create_response(payload)
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


def test_decode_order_create_response_invalid_fill_commission():
    exc, value = decode_order_create_response({
        "status": "FILLED",
        "clientOrderId": "x",
        "transactTime": 1,
        "executedQty": "1",
        "cummulativeQuoteQty": "10",
        "fills": [
            {"commission": "not-a-decimal", "commissionAsset": "USDT"},
        ],
    })
    assert value is None
    assert isinstance(exc, InvalidExchangeData)


def test_decode_order_create_response_negative_quote():
    exc, value = decode_order_create_response({
        "status": "FILLED",
        "clientOrderId": "x",
        "transactTime": 1,
        "executedQty": "1",
        "cummulativeQuoteQty": "-10",
    })
    assert value is None
    assert isinstance(exc, InvalidExchangeData)
