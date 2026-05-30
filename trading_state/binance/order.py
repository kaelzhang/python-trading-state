"""
Binance protocol adapters for orders.

Every public function returns `ValueOrException[T]`. Validation lives
inside the function body — callers never need a separate `validate_*`
step. The library never raises business exceptions; protocol-level
errors come back through the return value so the caller has to handle
them explicitly.
"""
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)
from decimal import Decimal, InvalidOperation
from datetime import datetime

from trading_state import (
    LimitOrderTicket,
    MarketOrderTicket,
    MarketQuantityType,
    Order,
    OrderSide,
    OrderStatus,
    OrderTicket,
    StopLossLimitOrderTicket,
    StopLossOrderTicket,
    TakeProfitLimitOrderTicket,
    TakeProfitOrderTicket,
    TimeInForce,
    InvalidExchangeData,
)
from trading_state.common import DECIMAL_ZERO, ValueOrException
from trading_state.symbol import Symbol

from .common import timestamp_to_datetime


class UnsupportedOrderTypeError(Exception):
    """
    Raised by `decode_order_snapshot` when the wire payload references
    an order type that this package does not model on the ticket side
    (e.g. Binance returns a snapshot of a MARKET order, which is
    terminal at placement and never appears in /openOrders, or a type
    we have not added a ticket class for yet).

    Surfaced through `ValueOrException` so callers decide how to react
    (log + skip the row, raise, retry under a different schema).
    """
    def __init__(self, order_type: str) -> None:
        super().__init__(
            f"order type '{order_type}' is not supported by "
            f'trading_state.binance.decode_order_snapshot'
        )
        self.order_type = order_type


def _format_decimal(value: Decimal) -> str:
    # Binance rejects scientific notation in numeric fields; format 'f'
    # always emits fixed-point and preserves the Decimal's precision
    # (so a quantity already quantised by PrecisionFilter renders with
    # the right number of trailing zeros).
    return format(value, 'f')


# Ref:
# https://developers.binance.com/docs/binance-spot-api-docs/rest-api/trading-endpoints#new-order-trade
def encode_order_request(
    ticket: OrderTicket,
) -> ValueOrException[dict]:
    """
    Encode an OrderTicket to the Binance REST `POST /api/v3/order`
    request body. All numeric and enum values are serialised to the
    plain strings Binance expects on the wire — the returned dict is
    directly json-serialisable.

    Returns:
        (None, dict)  — encoded payload ready for the HTTP layer.
        (exc, None)   — when the ticket type is not yet supported.
    """
    match ticket:
        case LimitOrderTicket() if ticket.post_only:
            # Binance Spot's LIMIT_MAKER variant: type changes to
            # 'LIMIT_MAKER' and timeInForce must NOT be sent (the
            # order-side validation enforces time_in_force is None).
            kwargs = dict(
                symbol=ticket.symbol.name,
                side=str(ticket.side),
                type='LIMIT_MAKER',
                quantity=_format_decimal(ticket.quantity),
                price=_format_decimal(ticket.price),
            )
        case LimitOrderTicket():
            kwargs = dict(
                symbol=ticket.symbol.name,
                side=str(ticket.side),
                type=str(ticket.type),
                timeInForce=str(ticket.time_in_force),
                quantity=_format_decimal(ticket.quantity),
                price=_format_decimal(ticket.price),
            )
        case MarketOrderTicket():
            kwargs = dict(
                symbol=ticket.symbol.name,
                side=str(ticket.side),
                type=str(ticket.type),
            )

            if ticket.quantity_type == MarketQuantityType.BASE:
                kwargs['quantity'] = _format_decimal(ticket.quantity)
            else:
                kwargs['quoteOrderQty'] = _format_decimal(ticket.quantity)
        case _:
            # Stop-loss / take-profit families aren't wired into the
            # encoder yet. Surface this as a protocol-side error rather
            # than a runtime crash deep in the request layer.
            return (
                InvalidExchangeData(
                    f'unsupported ticket type for encode_order_request: '
                    f'{type(ticket).__name__}'
                ),
                None,
            )

    # Get the full response of the order creation
    kwargs['newOrderRespType'] = 'FULL'

    return None, kwargs


# Maps the documented Spot OrderStatus surface
# (developers.binance.com/docs/binance-spot-api-docs/enums, verified
# 2026-05-30) into trading-state's deliberately smaller, exchange-
# agnostic OrderStatus enum. trading-state owns its OrderStatus
# vocabulary; this table is the binance <-> trading-state bridge.
#
# Mapping rationale:
#   NEW              -> CREATED          (order accepted by exchange, on the book)
#   PENDING_NEW      -> CREATED          (accepted but not yet on the book; state
#                                         does not distinguish this sub-state)
#   PARTIALLY_FILLED -> PARTIALLY_FILLED (trading-state has carried a first-class
#                                         PARTIALLY_FILLED member in OrderStatus
#                                         since v0.0.2 / commit e70822a
#                                         2025-11-24; activating it lets
#                                         query_orders(status=PARTIALLY_FILLED)
#                                         match partially-filled orders directly)
#   FILLED           -> FILLED           (terminal, filled in full)
#   CANCELED         -> CANCELLED        (terminal, user cancel)
#   PENDING_CANCEL   -> CANCELLING       (cancel request acknowledged but not
#                                         yet final — matches our intermediate)
#   EXPIRED          -> CANCELLED        (terminal, TIF expiry)
#   EXPIRED_IN_MATCH -> CANCELLED        (terminal, STP-triggered expiry)
#   REJECTED         -> REJECTED         (terminal, rejected pre-book)
_BINANCE_ORDER_STATUS_MAP = {
    'NEW': OrderStatus.CREATED,
    'PENDING_NEW': OrderStatus.CREATED,
    'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
    'FILLED': OrderStatus.FILLED,
    'CANCELED': OrderStatus.CANCELLED,
    'PENDING_CANCEL': OrderStatus.CANCELLING,
    'EXPIRED': OrderStatus.CANCELLED,
    'EXPIRED_IN_MATCH': OrderStatus.CANCELLED,
    'REJECTED': OrderStatus.REJECTED,
}


def _decode_order_status(
    status_str: str,
) -> ValueOrException[OrderStatus]:
    status = _BINANCE_ORDER_STATUS_MAP.get(status_str)
    if status is None:
        return (
            InvalidExchangeData(
                f'unknown order status from exchange: {status_str!r}'
            ),
            None,
        )
    return None, status


def _decode_decimal(
    raw,
    field_name: str,
    *,
    allow_negative: bool,
) -> ValueOrException[Decimal]:
    try:
        value = Decimal(raw)
    except (InvalidOperation, TypeError, ValueError):
        return (
            InvalidExchangeData(
                f'{field_name} is not a valid decimal: {raw!r}'
            ),
            None,
        )
    if not allow_negative and value < DECIMAL_ZERO:
        return (
            InvalidExchangeData(
                f'{field_name} must not be negative: {value}'
            ),
            None,
        )
    return None, value


def _require(payload: dict, key: str, field_name: str) -> ValueOrException:
    if key not in payload:
        return (
            InvalidExchangeData(
                f'missing required field {field_name!r} (payload key '
                f'{key!r})'
            ),
            None,
        )
    return None, payload[key]


# Ref:
# https://github.com/binance/binance-spot-api-docs/blob/master/user-data-stream.md#order-create
def decode_order_create_response(
    response: dict,
) -> ValueOrException[dict]:
    """
    Decode the REST `POST /api/v3/order` response into the kwargs for
    `state.update_order(...)`.

    Validates:
      - required fields present (status, clientOrderId, transactTime,
        executedQty, cummulativeQuoteQty)
      - status is one of the recognised Binance order statuses
      - executedQty / cummulativeQuoteQty are non-negative decimals
      - if fills are present, the commission components are coherent
        (commissionAsset present when commission > 0)
    """
    exc, status_raw = _require(response, 'status', 'status')
    if exc is not None:
        return exc, None
    exc, status = _decode_order_status(status_raw)
    if exc is not None:
        return exc, None

    exc, client_order_id = _require(
        response, 'clientOrderId', 'client_order_id'
    )
    if exc is not None:
        return exc, None

    exc, transact_time = _require(
        response, 'transactTime', 'transact_time'
    )
    if exc is not None:
        return exc, None

    exc, executed_qty_raw = _require(
        response, 'executedQty', 'filled_quantity'
    )
    if exc is not None:
        return exc, None
    exc, filled_quantity = _decode_decimal(
        executed_qty_raw, 'filled_quantity', allow_negative=False
    )
    if exc is not None:
        return exc, None

    exc, quote_qty_raw = _require(
        response, 'cummulativeQuoteQty', 'quote_quantity'
    )
    if exc is not None:
        return exc, None
    exc, quote_quantity = _decode_decimal(
        quote_qty_raw, 'quote_quantity', allow_negative=False
    )
    if exc is not None:
        return exc, None

    commission_asset = None
    commission_quantity: Decimal = DECIMAL_ZERO

    fills = response.get('fills')
    if fills:
        for fill in fills:
            exc, fill_commission = _decode_decimal(
                fill.get('commission', '0'),
                'commission', allow_negative=False,
            )
            if exc is not None:
                return exc, None
            commission_quantity += fill_commission
            fill_asset = fill.get('commissionAsset') or None
            if fill_asset is not None:
                commission_asset = fill_asset

        if commission_quantity > DECIMAL_ZERO and commission_asset is None:
            return (
                InvalidExchangeData(
                    'commission > 0 but no commissionAsset present in '
                    'any fill'
                ),
                None,
            )
    else:
        # No fills section: leave commission as None / 0 so caller can
        # tell update_order "not updated this round".
        commission_quantity = None  # type: ignore[assignment]

    # The full kwarg set state.update_order requires; emit `None` for
    # anything not present on the wire so `state.update_order(order,
    # **updates)` is a complete call regardless of execution context.
    return None, dict(
        status=status,
        updated_at=datetime.fromtimestamp(transact_time / 1000),
        id=client_order_id,
        filled_quantity=filled_quantity,
        quote_quantity=quote_quantity,
        commission_asset=commission_asset,
        commission_quantity=commission_quantity,
    )


# Ref
# https://github.com/binance/binance-spot-api-docs/blob/master/user-data-stream.md#order-update
def decode_order_update_event(
    payload: dict,
) -> ValueOrException[Tuple[str, dict]]:
    """
    Decode a WebSocket `executionReport` event into the (client order
    id, update kwargs) pair consumed by `state.update_order(...)`.

    Validates:
      - all required event fields present (c, X, z, Z, n, T)
      - status string is recognised
      - filled_quantity / quote_quantity / commission_quantity are
        non-negative decimals
      - commission_asset is present when commission_quantity > 0
    """
    exc, client_order_id = _require(payload, 'c', 'client_order_id')
    if exc is not None:
        return exc, None

    exc, order_status_raw = _require(payload, 'X', 'order_status')
    if exc is not None:
        return exc, None

    exc, filled_raw = _require(payload, 'z', 'filled_quantity')
    if exc is not None:
        return exc, None
    exc, filled_quantity = _decode_decimal(
        filled_raw, 'filled_quantity', allow_negative=False
    )
    if exc is not None:
        return exc, None

    exc, quote_raw = _require(payload, 'Z', 'quote_quantity')
    if exc is not None:
        return exc, None
    exc, quote_quantity = _decode_decimal(
        quote_raw, 'quote_quantity', allow_negative=False
    )
    if exc is not None:
        return exc, None

    exc, commission_raw = _require(payload, 'n', 'commission_quantity')
    if exc is not None:
        return exc, None
    exc, commission_quantity = _decode_decimal(
        commission_raw, 'commission_quantity', allow_negative=False
    )
    if exc is not None:
        return exc, None

    # `N` (commissionAsset) is allowed to be absent / null when no
    # commission was charged on this update; falsy values are
    # normalised to None to keep the downstream branch simple.
    commission_asset = payload.get('N') or None
    if commission_quantity > DECIMAL_ZERO and commission_asset is None:
        return (
            InvalidExchangeData(
                'commission_quantity > 0 but commission_asset is absent'
            ),
            None,
        )

    exc, transact_time = _require(payload, 'T', 'transact_time')
    if exc is not None:
        return exc, None
    updated_at = timestamp_to_datetime(transact_time)

    exc, status = _decode_order_status(order_status_raw)
    if exc is not None:
        return exc, None

    # The full kwarg set state.update_order requires; `id` is unset by
    # executionReport because the order is keyed by clientOrderId at
    # the caller (state.get_order_by_id), so `id=None` here means "do
    # not overwrite the existing Order.id".
    update_kwargs = {
        'status': status,
        'updated_at': updated_at,
        'id': None,
        'filled_quantity': filled_quantity,
        'quote_quantity': quote_quantity,
        'commission_asset': commission_asset,
        'commission_quantity': commission_quantity,
    }

    return None, (client_order_id, update_kwargs)


# Recovery: REST snapshot decoders -------------------------------------
#
# `/api/v3/order` (single GET), `/api/v3/openOrders` (list), and
# `/api/v3/allOrders` (list) all return the same per-order schema. The
# two decoders below consume that schema for different downstream
# consumers:
#
#   decode_order_query_response -> (id, update_kwargs)
#       Caller already has the Order in state (via get_order_by_id) and
#       wants to push the refreshed fields via state.update_order.
#       Symmetric with decode_order_update_event for WS executionReport.
#
#   decode_order_snapshot -> Order
#       Caller has determined the Order is NOT in state (e.g. cold
#       startup, mid-session import for an order placed by another
#       client / process during a WS gap) and wants to feed
#       state.import_order. Constructs the Order with id, status, and
#       all cumulative fields populated from the snapshot.
#
# trading_state.binance does not orchestrate the recovery flow; callers
# choose the right decoder per item based on the in-state existence
# check. See README "Recovery" for the canonical caller patterns.


def _decode_order_side(raw) -> ValueOrException[OrderSide]:
    try:
        return None, OrderSide(raw)
    except ValueError:
        return (
            InvalidExchangeData(f'unknown order side: {raw!r}'),
            None,
        )


def _decode_time_in_force(raw) -> ValueOrException[TimeInForce]:
    try:
        return None, TimeInForce(raw)
    except ValueError:
        return (
            InvalidExchangeData(f'unknown timeInForce: {raw!r}'),
            None,
        )


def decode_order_query_response(
    payload: dict,
) -> ValueOrException[Tuple[str, dict]]:
    """
    Decode a `GET /api/v3/order` response (or a single item from
    `GET /api/v3/openOrders` / `GET /api/v3/allOrders`, same schema)
    into the `(client_order_id, update_kwargs)` pair consumed by
    `state.update_order(...)`.

    Use this when the caller has confirmed the Order is already in
    state (via `state.get_order_by_id(id)`) and wants to refresh its
    fields from a REST snapshot. For Orders that are NOT in state,
    use `decode_order_snapshot` + `state.import_order` instead.

    Validates:
      - required fields present
      - status is recognised
      - executedQty / cummulativeQuoteQty are non-negative decimals

    Commission fields are absent from REST snapshots (only WS
    executionReport carries per-update commission); both are emitted
    as None so that `update_order` treats them as "not updated this
    round".
    """
    exc, client_order_id = _require(
        payload, 'clientOrderId', 'client_order_id'
    )
    if exc is not None:
        return exc, None

    exc, status_raw = _require(payload, 'status', 'status')
    if exc is not None:
        return exc, None
    exc, status = _decode_order_status(status_raw)
    if exc is not None:
        return exc, None

    exc, update_time = _require(payload, 'updateTime', 'updated_at')
    if exc is not None:
        return exc, None
    updated_at = timestamp_to_datetime(update_time)

    exc, executed_qty_raw = _require(
        payload, 'executedQty', 'filled_quantity'
    )
    if exc is not None:
        return exc, None
    exc, filled_quantity = _decode_decimal(
        executed_qty_raw, 'filled_quantity', allow_negative=False
    )
    if exc is not None:
        return exc, None

    exc, quote_qty_raw = _require(
        payload, 'cummulativeQuoteQty', 'quote_quantity'
    )
    if exc is not None:
        return exc, None
    exc, quote_quantity = _decode_decimal(
        quote_qty_raw, 'quote_quantity', allow_negative=False
    )
    if exc is not None:
        return exc, None

    update_kwargs = {
        'status': status,
        'updated_at': updated_at,
        'id': None,
        'filled_quantity': filled_quantity,
        'quote_quantity': quote_quantity,
        'commission_asset': None,
        'commission_quantity': None,
    }
    return None, (client_order_id, update_kwargs)


def _build_snapshot_ticket(
    item: dict,
    *,
    symbol: Symbol,
    side: OrderSide,
    quantity: Decimal,
) -> ValueOrException[OrderTicket]:
    """
    Dispatch on `item['type']` and construct the appropriate ticket.
    Returns `UnsupportedOrderTypeError` for types that have no
    trading_state ticket-side representation (currently MARKET — the
    Spot REST schema only ever surfaces MARKET orders as terminal, and
    a terminal market order has no ticket-level price to round-trip
    through filters).
    """
    type_raw = item.get('type')
    if type_raw is None:
        return InvalidExchangeData("missing required field 'type'"), None

    if type_raw in ('LIMIT', 'LIMIT_MAKER', 'STOP_LOSS_LIMIT',
                    'TAKE_PROFIT_LIMIT'):
        exc, price_raw = _require(item, 'price', 'price')
        if exc is not None:
            return exc, None
        exc, price = _decode_decimal(
            price_raw, 'price', allow_negative=False
        )
        if exc is not None:
            return exc, None
    else:
        price = None

    if type_raw in ('STOP_LOSS', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT',
                    'TAKE_PROFIT_LIMIT'):
        exc, stop_price_raw = _require(item, 'stopPrice', 'stop_price')
        if exc is not None:
            return exc, None
        exc, stop_price = _decode_decimal(
            stop_price_raw, 'stop_price', allow_negative=False
        )
        if exc is not None:
            return exc, None
    else:
        stop_price = None

    if type_raw in ('LIMIT', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT_LIMIT'):
        exc, tif_raw = _require(item, 'timeInForce', 'time_in_force')
        if exc is not None:
            return exc, None
        exc, tif = _decode_time_in_force(tif_raw)
        if exc is not None:
            return exc, None
    else:
        tif = None

    if type_raw == 'LIMIT':
        return None, LimitOrderTicket(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            time_in_force=tif,
        )
    if type_raw == 'LIMIT_MAKER':
        return None, LimitOrderTicket(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            post_only=True,
        )
    if type_raw == 'STOP_LOSS':
        return None, StopLossOrderTicket(
            symbol=symbol,
            side=side,
            quantity=quantity,
            stop_price=stop_price,
        )
    if type_raw == 'STOP_LOSS_LIMIT':
        return None, StopLossLimitOrderTicket(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=tif,
        )
    if type_raw == 'TAKE_PROFIT':
        return None, TakeProfitOrderTicket(
            symbol=symbol,
            side=side,
            quantity=quantity,
            stop_price=stop_price,
        )
    if type_raw == 'TAKE_PROFIT_LIMIT':
        return None, TakeProfitLimitOrderTicket(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=tif,
        )

    return UnsupportedOrderTypeError(str(type_raw)), None


def decode_order_snapshot(
    item: dict,
    *,
    symbol: Symbol,
    data: Optional[Dict[str, Any]] = None,
) -> ValueOrException[Order]:
    """
    Decode a Binance Spot order snapshot item (single item from
    `GET /api/v3/openOrders`, `GET /api/v3/allOrders`, or the body of
    `GET /api/v3/order`) into a fully-populated `Order` ready to feed
    `state.import_order`.

    Used on the recovery path when the caller has confirmed the order
    is NOT already in state (cold startup, mid-session reconnect
    discovery of orders placed by another client / process during a
    WS gap, cold pull of /allOrders for accounting completeness).

    Synthesizes a ticket from `item['type']` via
    `_build_snapshot_ticket` and populates `id`, `status`,
    `filled_quantity`, `quote_quantity`, `created_at`, and `updated_at`
    from the snapshot fields. Commission fields are left at their
    defaults — REST snapshots do not carry per-update commission.

    Accepts any status the wire returns (open or terminal). Callers
    that want to gate by status (e.g. skip terminal rows from
    /allOrders) should filter before calling.

    Returns `UnsupportedOrderTypeError` for types that trading_state
    does not model on the ticket side (currently MARKET).
    """
    exc, client_order_id = _require(
        item, 'clientOrderId', 'client_order_id'
    )
    if exc is not None:
        return exc, None

    exc, side_raw = _require(item, 'side', 'side')
    if exc is not None:
        return exc, None
    exc, side = _decode_order_side(side_raw)
    if exc is not None:
        return exc, None

    exc, status_raw = _require(item, 'status', 'status')
    if exc is not None:
        return exc, None
    exc, status = _decode_order_status(status_raw)
    if exc is not None:
        return exc, None

    exc, orig_qty_raw = _require(item, 'origQty', 'quantity')
    if exc is not None:
        return exc, None
    exc, quantity = _decode_decimal(
        orig_qty_raw, 'quantity', allow_negative=False
    )
    if exc is not None:
        return exc, None

    exc, executed_qty_raw = _require(
        item, 'executedQty', 'filled_quantity'
    )
    if exc is not None:
        return exc, None
    exc, filled_quantity = _decode_decimal(
        executed_qty_raw, 'filled_quantity', allow_negative=False
    )
    if exc is not None:
        return exc, None

    exc, quote_qty_raw = _require(
        item, 'cummulativeQuoteQty', 'quote_quantity'
    )
    if exc is not None:
        return exc, None
    exc, quote_quantity = _decode_decimal(
        quote_qty_raw, 'quote_quantity', allow_negative=False
    )
    if exc is not None:
        return exc, None

    exc, time_raw = _require(item, 'time', 'created_at')
    if exc is not None:
        return exc, None
    created_at = timestamp_to_datetime(time_raw)

    exc, update_time_raw = _require(item, 'updateTime', 'updated_at')
    if exc is not None:
        return exc, None
    updated_at = timestamp_to_datetime(update_time_raw)

    exc, ticket = _build_snapshot_ticket(
        item, symbol=symbol, side=side, quantity=quantity,
    )
    if exc is not None:
        return exc, None

    order = Order(
        ticket=ticket,
        data=data,
        id=client_order_id,
        status=status,
        filled_quantity=filled_quantity,
        quote_quantity=quote_quantity,
        commission_asset=None,
        commission_quantity=DECIMAL_ZERO,
        created_at=created_at,
        updated_at=updated_at,
    )
    return None, order
