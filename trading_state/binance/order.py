"""
Binance protocol adapters for orders.

Every public function returns `ValueOrException[T]`. Validation lives
inside the function body — callers never need a separate `validate_*`
step. The library never raises business exceptions; protocol-level
errors come back through the return value so the caller has to handle
them explicitly.
"""
from typing import (
    Tuple,
)
from decimal import Decimal, InvalidOperation
from datetime import datetime

from trading_state import (
    OrderTicket,
    OrderStatus,
    LimitOrderTicket,
    MarketOrderTicket,
    MarketQuantityType,
    InvalidExchangeData,
)
from trading_state.common import DECIMAL_ZERO, ValueOrException

from .common import timestamp_to_datetime


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
#   NEW              -> CREATED      (order accepted by exchange, on the book)
#   PENDING_NEW      -> CREATED      (accepted but not yet on the book; state
#                                     does not distinguish this sub-state)
#   PARTIALLY_FILLED -> CREATED      (state treats partial fill as a quality
#                                     of CREATED — caller derives "partially
#                                     filled" from order.filled_quantity
#                                     against order.ticket.quantity)
#   FILLED           -> FILLED       (terminal, filled in full)
#   CANCELED         -> CANCELLED    (terminal, user cancel)
#   PENDING_CANCEL   -> CANCELLING   (cancel request acknowledged but not
#                                     yet final — matches our intermediate)
#   EXPIRED          -> CANCELLED    (terminal, TIF expiry)
#   EXPIRED_IN_MATCH -> CANCELLED    (terminal, STP-triggered expiry)
#   REJECTED         -> REJECTED     (terminal, rejected pre-book)
_BINANCE_ORDER_STATUS_MAP = {
    'NEW': OrderStatus.CREATED,
    'PENDING_NEW': OrderStatus.CREATED,
    'PARTIALLY_FILLED': OrderStatus.CREATED,
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
