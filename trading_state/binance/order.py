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


# Covers every Spot OrderStatus documented at
# developers.binance.com/docs/binance-spot-api-docs/enums.
# Verified 2026-05-30 against the Spot enums page.
#
# Mapping rationale:
#   NEW            -> CREATED        (order accepted by exchange, on the book)
#   PENDING_NEW    -> CREATED        (accepted but not yet on the book; state
#                                     does not distinguish this sub-state)
#   PARTIALLY_FILLED -> PARTIALLY_FILLED
#                                    (state has a first-class status for this)
#   FILLED         -> FILLED         (terminal, filled in full)
#   CANCELED       -> CANCELLED      (terminal, user cancel)
#   PENDING_CANCEL -> CANCELLING     (cancel request acknowledged but not
#                                     yet final — matches our intermediate)
#   EXPIRED        -> CANCELLED      (terminal, TIF expiry)
#   EXPIRED_IN_MATCH -> CANCELLED    (terminal, STP-triggered expiry; behaves
#                                     like a non-trading-prevented cancel from
#                                     state's perspective)
#   REJECTED       -> REJECTED       (terminal, rejected pre-book)
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
# https://developers.binance.com/docs/binance-spot-api-docs/rest-api/trading-endpoints#new-order-trade
# Verified against the Spot REST trading endpoints documentation
# 2026-05-30.
def decode_order_create_response(
    response: dict,
) -> ValueOrException[dict]:
    """
    Decode the REST `POST /api/v3/order` response (newOrderRespType =
    `FULL`) into the kwargs for `state.update_order(...)`.

    The `FULL` response is the union of the `ACK` and `RESULT` shapes
    plus a `fills` list (one entry per per-trade leg of the initial
    fill). Consumed keys:

      status              — order status (mapped via
                            `_BINANCE_ORDER_STATUS_MAP`)
      clientOrderId       — primary key; set as Order.id
      transactTime        — used as `created_at`
      executedQty         — cumulative filled quantity
      cummulativeQuoteQty — cumulative quote quantity
      fills[]             — summed per-fill commission + commissionAsset

    Intentionally not surfaced (consistent with `decode_order_update_event`):
      symbol / orderId / orderListId / origClientOrderId / price /
      origQty / type / side / timeInForce / workingTime /
      selfTradePreventionMode / preventedMatchId / preventedQuantity /
      stopPrice / strategyId / strategyType / trailingDelta /
      trailingTime / icebergQty / fills[].price / fills[].qty /
      fills[].tradeId / fills[].allocId. Same rationale as the
      executionReport docstring above.

    Validates:
      - required fields present (status, clientOrderId, transactTime,
        executedQty, cummulativeQuoteQty)
      - status is one of the documented Spot enums
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

    updates = dict(
        status=status,
        id=client_order_id,
        created_at=datetime.fromtimestamp(transact_time / 1000),
        filled_quantity=filled_quantity,
        quote_quantity=quote_quantity,
    )

    fills = response.get('fills')
    if fills:
        commission_quantity = DECIMAL_ZERO
        commission_asset = None
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

        updates['commission_asset'] = commission_asset
        updates['commission_quantity'] = commission_quantity

    return None, updates


# Ref
# https://developers.binance.com/docs/binance-spot-api-docs/user-data-stream
# (current page; the older github mirror is no longer authoritative.)
def decode_order_update_event(
    payload: dict,
) -> ValueOrException[Tuple[str, dict]]:
    """
    Decode a WebSocket `executionReport` event into the (client order
    id, update kwargs) pair consumed by `state.update_order(...)`.

    Verified against the Spot User Data Stream documentation
    2026-05-30. The Binance payload is the full ~50-key executionReport
    surface; this decoder is deliberately narrower than the SDK's
    pass-through column map because `trading_state.state` exposes a
    much smaller state-mutating contract.

    Consumed keys:
      c  client_order_id  — primary key into `state.get_order_by_id`
      X  order_status     — mapped via `_BINANCE_ORDER_STATUS_MAP`
      z  filled_quantity  — cumulative base filled
      Z  quote_quantity   — cumulative quote transacted
      n  commission_qty   — cumulative commission charged on this event
      N  commission_asset — asset the commission was charged in
      T  transact_time    — used as the `updated_at` timestamp

    Intentionally not surfaced (with rationale):
      e, E         event type / event time — bookkeeping; transact_time
                                              already carries the
                                              order-level moment.
      s            symbol               — the order is already keyed by id.
      S, o, f, q,  side / type / TIF /  — the originating ticket already
       p, P, F,    quantities / prices    carries these; resending them
       Q, V                               is redundant.
      g, C         OCO list id / orig    — OCO lists are not modeled in
                   client_order_id        state today; the original
                                          client_order_id is the
                                          same value as `c` for non-OCO
                                          flow.
      x            execution_type        — finer-grained than `X`; state
                                          collapses NEW / TRADE / REPLACED
                                          / EXPIRED / REJECTED / CALCULATED
                                          into the `X` status transitions.
      r            reject_reason         — captured implicitly by the
                                          REJECTED status; surfacing the
                                          reason string requires adding
                                          a diagnostic surface on Order,
                                          tracked as a follow-up.
      i, t, I      exchange order_id /   — state currently keys orders
                   trade_id / exec_id     by client_order_id; exchange
                                          ids are queryable on the SDK
                                          side, not on state.
      l, L, Y      last filled qty /     — state derives per-fill Trade
                   price / quote qty      records from the cumulative
                                          (z, Z, n) deltas, so the
                                          per-event last-trade fields
                                          are not needed.
      w, m, M      is_on_book / maker /  — diagnostic-only; state has no
                   ignore                 representation for them.
      O            order_creation_time   — `created_at` is set from the
                                          first `updated_at` transition
                                          into CREATED; `O` would be
                                          marginally more accurate but
                                          not state-correctness-relevant.
      d, D, j, J,  trailing_delta /      — STP / SOR / OCO / peg / trailing
       v, A, B, u,  trailing_time /       are not first-class concepts in
       U, Cs, pl,   strategy / prevented   state today (they ride on
       pL, pY, b,   match / counter /     OrderTicket subclasses that
       a, k, uS,    prevented exec /      `state` treats as opaque). They
       gP, gOT,     match_type /           do not transition the order
       gOV, gp,     allocation /           state machine.
       eR, W        working_floor /
                    used_sor / peg
                    fields / expiry_reason
                    / working_time
      subscriptionId — WS-API routing tag; the caller already knows
                       which subscription it owns.

    Validates:
      - all required event fields present (c, X, z, Z, n, T)
      - status string is one of the documented Spot enums
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

    update_kwargs = {
        'status': status,
        'filled_quantity': filled_quantity,
        'quote_quantity': quote_quantity,
        'updated_at': updated_at,
        'commission_asset': commission_asset,
        'commission_quantity': commission_quantity,
    }

    return None, (client_order_id, update_kwargs)
