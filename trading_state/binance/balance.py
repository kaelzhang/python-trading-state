"""
Binance protocol adapters for balances and balance-side events.

Every public function returns `ValueOrException[T]`. Validation is
embedded; callers do not call a separate `validate_*` step.
"""
from typing import Set
from decimal import Decimal, InvalidOperation

from trading_state import (
    Balance,
    CashFlow,
    InvalidExchangeData,
)
from trading_state.common import ValueOrException, DECIMAL_ZERO

from .common import timestamp_to_datetime


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


# Ref
# https://github.com/binance/binance-spot-api-docs/blob/master/user-data-stream.md#balance-update
def decode_account_update_event(
    payload: dict,
) -> ValueOrException[Set[Balance]]:
    """
    Decode a WS `outboundAccountPosition` event into a set of
    Balance value objects.

    Validates:
      - required event fields ('u' and 'B') present
      - each balance entry has 'a' / 'f' / 'l'
      - free / locked are non-negative decimals
    """
    exc, raw_time = _require(payload, 'u', 'event_time')
    if exc is not None:
        return exc, None
    time = timestamp_to_datetime(raw_time)

    exc, raw_balances = _require(payload, 'B', 'balances')
    if exc is not None:
        return exc, None

    balances: Set[Balance] = set()
    for entry in raw_balances:
        exc, asset = _require(entry, 'a', 'asset')
        if exc is not None:
            return exc, None
        exc, free_raw = _require(entry, 'f', 'free')
        if exc is not None:
            return exc, None
        exc, locked_raw = _require(entry, 'l', 'locked')
        if exc is not None:
            return exc, None
        exc, free = _decode_decimal(free_raw, 'free', allow_negative=False)
        if exc is not None:
            return exc, None
        exc, locked = _decode_decimal(
            locked_raw, 'locked', allow_negative=False
        )
        if exc is not None:
            return exc, None

        balances.add(Balance(asset, free, locked, time))

    return None, balances


def decode_balance_update_event(
    payload: dict,
) -> ValueOrException[CashFlow]:
    """
    Decode a WS `balanceUpdate` event into a CashFlow value object.

    `d` is a signed delta — it can legitimately be negative, so we
    validate decimal parsing but allow either sign.
    """
    exc, asset = _require(payload, 'a', 'asset')
    if exc is not None:
        return exc, None

    exc, delta_raw = _require(payload, 'd', 'delta')
    if exc is not None:
        return exc, None
    exc, delta = _decode_decimal(delta_raw, 'delta', allow_negative=True)
    if exc is not None:
        return exc, None

    exc, raw_time = _require(payload, 'T', 'clear_time')
    if exc is not None:
        return exc, None
    clear_time = timestamp_to_datetime(raw_time)

    return None, CashFlow(asset, delta, clear_time)


def decode_account_info_response(
    account_info: dict,
) -> ValueOrException[Set[Balance]]:
    """
    Decode the REST `GET /api/v3/account` response into a set of
    Balance value objects.

    Validates required fields and non-negative free / locked.
    """
    exc, raw_time = _require(account_info, 'updateTime', 'update_time')
    if exc is not None:
        return exc, None
    time = timestamp_to_datetime(raw_time)

    exc, raw_balances = _require(account_info, 'balances', 'balances')
    if exc is not None:
        return exc, None

    balances: Set[Balance] = set()
    for entry in raw_balances:
        exc, asset = _require(entry, 'asset', 'asset')
        if exc is not None:
            return exc, None
        exc, free_raw = _require(entry, 'free', 'free')
        if exc is not None:
            return exc, None
        exc, locked_raw = _require(entry, 'locked', 'locked')
        if exc is not None:
            return exc, None
        exc, free = _decode_decimal(free_raw, 'free', allow_negative=False)
        if exc is not None:
            return exc, None
        exc, locked = _decode_decimal(
            locked_raw, 'locked', allow_negative=False
        )
        if exc is not None:
            return exc, None

        balances.add(Balance(asset, free, locked, time))

    return None, balances
