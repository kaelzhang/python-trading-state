"""
Tests for the recovery surface:

- `state.import_order(order)` — fail-fast preconditions, ORDER_CREATED
  emission, INIT phase skip, terminal-status import.
- `trading_state.binance.decode_order_snapshot` — per-type ticket
  synthesis, status passthrough including terminals, error surfacing.
- `trading_state.binance.decode_order_query_response` — (id, kwargs)
  pair for refresh-an-existing-order path.
- End-to-end flows: cold startup, periodic check, mid-session
  reconnect with gap-fill.
"""
from datetime import datetime
from decimal import Decimal

import pytest

from trading_state import (
    DuplicateOrderIdError,
    InvalidOrderForImportError,
    LimitOrderTicket,
    Order,
    OrderSide,
    OrderStatus,
    StopLossLimitOrderTicket,
    StopLossOrderTicket,
    TakeProfitLimitOrderTicket,
    TakeProfitOrderTicket,
    TimeInForce,
    TradingStateEvent,
)
from trading_state.binance import (
    UnsupportedOrderTypeError,
    decode_order_query_response,
    decode_order_snapshot,
)
from trading_state.common import DECIMAL_ZERO

from .fixtures import (
    BTCUSDT,
    init_state,
)


BTCUSDT_NAME = BTCUSDT.name


def _snapshot_item(**overrides):
    """Wire-shape /api/v3/openOrders item, with defaults for a NEW
    LIMIT BUY BTCUSDT that the caller can override per test."""
    item = dict(
        symbol='BTCUSDT',
        orderId=12345,
        clientOrderId='ord-1',
        price='30000',
        origQty='0.5',
        executedQty='0',
        cummulativeQuoteQty='0',
        status='NEW',
        timeInForce='GTC',
        type='LIMIT',
        side='BUY',
        stopPrice='0',
        time=1_700_000_000_000,
        updateTime=1_700_000_000_000,
    )
    item.update(overrides)
    return item


# import_order ----------------------------------------------------------

def test_import_order_happy_path_open_status_emits_order_created():
    state = init_state()
    btcusdt = state.get_symbol(BTCUSDT_NAME)
    captured: list[Order] = []
    state.on(
        TradingStateEvent.ORDER_CREATED,
        lambda o: captured.append(o),
    )

    order = Order(
        ticket=LimitOrderTicket(
            symbol=btcusdt,
            side=OrderSide.BUY,
            quantity=Decimal('0.5'),
            price=Decimal('30000'),
            time_in_force=TimeInForce.GTC,
        ),
        id='external-1',
        status=OrderStatus.CREATED,
        created_at=datetime(2024, 6, 1),
        updated_at=datetime(2023, 11, 1),
    )

    exc, returned = state.import_order(order)
    assert exc is None
    assert returned is order
    assert state.get_order_by_id('external-1') is order
    assert order in state.get_open_orders()
    assert captured == [order]


def test_import_order_terminal_status_goes_to_history_not_open():
    """Imported terminal orders must populate history + id-index but
    not _open_orders (they're already done)."""
    state = init_state()
    btcusdt = state.get_symbol(BTCUSDT_NAME)
    order = Order(
        ticket=LimitOrderTicket(
            symbol=btcusdt,
            side=OrderSide.BUY,
            quantity=Decimal('0.5'),
            price=Decimal('30000'),
            time_in_force=TimeInForce.GTC,
        ),
        id='hist-1',
        status=OrderStatus.FILLED,
        filled_quantity=Decimal('0.5'),
        quote_quantity=Decimal('15000'),
        created_at=datetime(2024, 6, 1),
        updated_at=datetime(2024, 6, 1, 0, 1),
    )

    exc, _ = state.import_order(order)
    assert exc is None
    assert state.get_order_by_id('hist-1') is order
    # Terminal: NOT in get_open_orders, but in query_orders history.
    assert order not in state.get_open_orders()
    history = list(state.query_orders(id=lambda v, _: v == 'hist-1'))
    assert history == [order]


def test_import_order_skips_init_phase_in_event_order():
    """Imported orders never emit a synthetic INIT -> CREATED
    transition; subscribers that listen only to ORDER_STATUS_UPDATED
    see no spurious events at import time. ORDER_CREATED is the only
    creation-time signal."""
    state = init_state()
    btcusdt = state.get_symbol(BTCUSDT_NAME)

    status_events: list = []
    state.on(
        TradingStateEvent.ORDER_STATUS_UPDATED,
        lambda order, status: status_events.append(status),
    )

    order = Order(
        ticket=LimitOrderTicket(
            symbol=btcusdt,
            side=OrderSide.BUY,
            quantity=Decimal('0.5'),
            price=Decimal('30000'),
            time_in_force=TimeInForce.GTC,
        ),
        id='no-init-1',
        status=OrderStatus.PARTIALLY_FILLED,
        filled_quantity=Decimal('0.2'),
        quote_quantity=Decimal('6000'),
        created_at=datetime(2024, 6, 1),
        updated_at=datetime(2024, 6, 1, 0, 1),
    )

    exc, _ = state.import_order(order)
    assert exc is None
    # The Order's status remains PARTIALLY_FILLED; no synthetic events
    # were emitted between INIT and PARTIALLY_FILLED.
    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert status_events == []


def test_import_order_missing_id_fail_fast():
    state = init_state()
    btcusdt = state.get_symbol(BTCUSDT_NAME)
    order = Order(
        ticket=LimitOrderTicket(
            symbol=btcusdt,
            side=OrderSide.BUY,
            quantity=Decimal('0.5'),
            price=Decimal('30000'),
            time_in_force=TimeInForce.GTC,
        ),
        # id omitted on purpose
        status=OrderStatus.CREATED,
    )
    exc, returned = state.import_order(order)
    assert returned is None
    assert isinstance(exc, InvalidOrderForImportError)


def test_import_order_duplicate_id_fail_fast():
    state = init_state()
    btcusdt = state.get_symbol(BTCUSDT_NAME)

    def build_order():
        return Order(
            ticket=LimitOrderTicket(
                symbol=btcusdt,
                side=OrderSide.BUY,
                quantity=Decimal('0.5'),
                price=Decimal('30000'),
                time_in_force=TimeInForce.GTC,
            ),
            id='dup-1',
            status=OrderStatus.CREATED,
            created_at=datetime(2024, 6, 1),
            updated_at=datetime(2023, 11, 1),
        )

    exc1, _ = state.import_order(build_order())
    assert exc1 is None

    exc2, returned = state.import_order(build_order())
    assert returned is None
    assert isinstance(exc2, DuplicateOrderIdError)
    assert exc2.order_id == 'dup-1'


def test_import_order_then_update_order_works_normally():
    """An imported order goes through subsequent update_order calls
    exactly like a natively allocated order: stale-update drop, fill
    events, terminal purge."""
    state = init_state()
    btcusdt = state.get_symbol(BTCUSDT_NAME)
    order = Order(
        ticket=LimitOrderTicket(
            symbol=btcusdt,
            side=OrderSide.BUY,
            quantity=Decimal('0.5'),
            price=Decimal('30000'),
            time_in_force=TimeInForce.GTC,
        ),
        id='ext-1',
        status=OrderStatus.PARTIALLY_FILLED,
        filled_quantity=Decimal('0.1'),
        quote_quantity=Decimal('3000'),
        created_at=datetime(2024, 6, 1),
        updated_at=datetime(2024, 6, 1, 0, 1),
    )
    state.import_order(order)

    # Now push the fill completion.
    state.update_order(
        order,
        status=OrderStatus.FILLED,
        updated_at=datetime(2024, 6, 1, 0, 2),
        id=None,
        filled_quantity=Decimal('0.5'),
        quote_quantity=Decimal('15000'),
        commission_asset=None,
        commission_quantity=None,
    )
    assert order.status is OrderStatus.FILLED
    assert order not in state.get_open_orders()


# decode_order_snapshot -------------------------------------------------

@pytest.mark.parametrize(
    'type_raw,extra,expected_cls',
    [
        ('LIMIT', {}, LimitOrderTicket),
        ('LIMIT_MAKER', {'timeInForce': 'GTC'}, LimitOrderTicket),
        ('STOP_LOSS', {'stopPrice': '25000'}, StopLossOrderTicket),
        (
            'STOP_LOSS_LIMIT',
            {'stopPrice': '25000', 'timeInForce': 'GTC'},
            StopLossLimitOrderTicket,
        ),
        ('TAKE_PROFIT', {'stopPrice': '35000'}, TakeProfitOrderTicket),
        (
            'TAKE_PROFIT_LIMIT',
            {'stopPrice': '35000', 'timeInForce': 'GTC'},
            TakeProfitLimitOrderTicket,
        ),
    ],
)
def test_decode_order_snapshot_per_type_ticket_synthesis(
    type_raw, extra, expected_cls,
):
    state = init_state()
    btcusdt = state.get_symbol(BTCUSDT_NAME)
    item = _snapshot_item(type=type_raw, **extra)

    exc, order = decode_order_snapshot(item, symbol=btcusdt)
    assert exc is None
    assert isinstance(order.ticket, expected_cls)
    assert order.id == 'ord-1'
    assert order.ticket.side is OrderSide.BUY


def test_decode_order_snapshot_market_type_unsupported():
    state = init_state()
    btcusdt = state.get_symbol(BTCUSDT_NAME)
    item = _snapshot_item(type='MARKET')
    exc, order = decode_order_snapshot(item, symbol=btcusdt)
    assert order is None
    assert isinstance(exc, UnsupportedOrderTypeError)
    assert exc.order_type == 'MARKET'


def test_decode_order_snapshot_unknown_type_unsupported():
    state = init_state()
    btcusdt = state.get_symbol(BTCUSDT_NAME)
    item = _snapshot_item(type='OCO')
    exc, order = decode_order_snapshot(item, symbol=btcusdt)
    assert order is None
    assert isinstance(exc, UnsupportedOrderTypeError)
    assert exc.order_type == 'OCO'


def test_decode_order_snapshot_accepts_terminal_status():
    """Snapshots from /allOrders or /api/v3/order may legitimately
    carry terminal status; the decoder must accept that and let
    state.import_order decide what to do."""
    state = init_state()
    btcusdt = state.get_symbol(BTCUSDT_NAME)
    item = _snapshot_item(
        status='FILLED',
        executedQty='0.5',
        cummulativeQuoteQty='15000',
    )
    exc, order = decode_order_snapshot(item, symbol=btcusdt)
    assert exc is None
    assert order.status is OrderStatus.FILLED
    assert order.filled_quantity == Decimal('0.5')


def test_decode_order_snapshot_propagates_data_param():
    state = init_state()
    btcusdt = state.get_symbol(BTCUSDT_NAME)
    item = _snapshot_item()
    exc, order = decode_order_snapshot(
        item, symbol=btcusdt, data={'strategy': 'recover'},
    )
    assert exc is None
    assert order.data == {'strategy': 'recover'}


def test_decode_order_snapshot_missing_required_field():
    state = init_state()
    btcusdt = state.get_symbol(BTCUSDT_NAME)
    item = _snapshot_item()
    item.pop('clientOrderId')
    exc, order = decode_order_snapshot(item, symbol=btcusdt)
    assert order is None
    assert exc is not None


# decode_order_query_response ------------------------------------------

def test_decode_order_query_response_happy_path():
    item = _snapshot_item(
        status='PARTIALLY_FILLED',
        executedQty='0.2',
        cummulativeQuoteQty='6000',
        updateTime=1_700_000_000_500,
    )
    exc, decoded = decode_order_query_response(item)
    assert exc is None
    cid, kwargs = decoded
    assert cid == 'ord-1'
    assert kwargs['status'] is OrderStatus.PARTIALLY_FILLED
    assert kwargs['filled_quantity'] == Decimal('0.2')
    assert kwargs['quote_quantity'] == Decimal('6000')
    # id is None: caller already has the Order looked up by id.
    assert kwargs['id'] is None
    # Commission absent in REST snapshot — emitted as None so
    # update_order treats them as "not updated this round".
    assert kwargs['commission_asset'] is None
    assert kwargs['commission_quantity'] is None


def test_decode_order_query_response_kwargs_match_update_order_signature():
    """The returned kwargs must be feedable directly to
    state.update_order without missing or extraneous keys."""
    state = init_state()
    btcusdt = state.get_symbol(BTCUSDT_NAME)
    # Build a live order via allocate
    _, [order] = state.allocate(
        LimitOrderTicket(
            symbol=btcusdt, side=OrderSide.BUY,
            quantity=Decimal('0.1'), price=Decimal('30000'),
            time_in_force=TimeInForce.GTC,
        )
    )
    state.update_order(
        order,
        status=OrderStatus.CREATED,
        updated_at=datetime(2023, 11, 1),
        id='ord-2',
        filled_quantity=DECIMAL_ZERO,
        quote_quantity=DECIMAL_ZERO,
        commission_asset=None,
        commission_quantity=None,
    )
    # Refresh via REST snapshot
    item = _snapshot_item(
        clientOrderId='ord-2',
        status='FILLED',
        executedQty='0.1',
        cummulativeQuoteQty='3000',
        updateTime=1_700_000_001_000,
    )
    exc, decoded = decode_order_query_response(item)
    assert exc is None
    cid, kwargs = decoded
    # This must not TypeError on a missing required kwarg.
    state.update_order(order, **kwargs)
    assert order.status is OrderStatus.FILLED


# End-to-end recovery flows -------------------------------------------

def test_cold_startup_imports_open_orders_from_snapshot():
    state = init_state()
    api_payload = [
        _snapshot_item(clientOrderId='cold-1', origQty='0.1'),
        _snapshot_item(
            clientOrderId='cold-2',
            origQty='0.2',
            status='PARTIALLY_FILLED',
            executedQty='0.05',
            cummulativeQuoteQty='1500',
        ),
    ]
    for item in api_payload:
        sym = state.get_symbol(item['symbol'])
        assert sym is not None
        exc, order = decode_order_snapshot(item, symbol=sym)
        assert exc is None
        exc, _ = state.import_order(order)
        assert exc is None

    open_ids = {o.id for o in state.get_open_orders()}
    assert open_ids == {'cold-1', 'cold-2'}


def test_periodic_check_refreshes_known_order_via_update_order():
    """A long-open order is refreshed by `/api/v3/order` once and the
    resulting (id, kwargs) is fed directly to update_order."""
    state = init_state()
    btcusdt = state.get_symbol(BTCUSDT_NAME)
    _, [order] = state.allocate(
        LimitOrderTicket(
            symbol=btcusdt, side=OrderSide.BUY,
            quantity=Decimal('0.1'), price=Decimal('30000'),
            time_in_force=TimeInForce.GTC,
        )
    )
    state.update_order(
        order,
        status=OrderStatus.CREATED,
        updated_at=datetime(2023, 11, 1),
        id='periodic-1',
        filled_quantity=DECIMAL_ZERO,
        quote_quantity=DECIMAL_ZERO,
        commission_asset=None,
        commission_quantity=None,
    )

    # ... time passes; caller polls /api/v3/order and finds FILLED.
    rest_payload = _snapshot_item(
        clientOrderId='periodic-1',
        status='FILLED',
        executedQty='0.1',
        cummulativeQuoteQty='3000',
        updateTime=1_700_000_500_000,
    )
    exc, decoded = decode_order_query_response(rest_payload)
    assert exc is None
    cid, kwargs = decoded
    refreshed = state.get_order_by_id(cid)
    assert refreshed is order
    state.update_order(refreshed, **kwargs)
    assert refreshed.status is OrderStatus.FILLED
    assert refreshed not in state.get_open_orders()


def test_mid_session_reconnect_imports_new_and_refreshes_existing():
    """Reconnect after a WS gap: openOrders payload contains both an
    order the local state already knows (refresh path -> update_order)
    and an order the local state has never seen (import path ->
    import_order). Local-open orders not present in the payload need a
    separate gap-fill via /api/v3/order."""
    state = init_state()
    btcusdt = state.get_symbol(BTCUSDT_NAME)

    # Pre-existing local order.
    _, [known] = state.allocate(
        LimitOrderTicket(
            symbol=btcusdt, side=OrderSide.BUY,
            quantity=Decimal('0.1'), price=Decimal('30000'),
            time_in_force=TimeInForce.GTC,
        )
    )
    state.update_order(
        known,
        status=OrderStatus.CREATED,
        updated_at=datetime(2023, 11, 1),
        id='local-known',
        filled_quantity=DECIMAL_ZERO,
        quote_quantity=DECIMAL_ZERO,
        commission_asset=None,
        commission_quantity=None,
    )

    # Another pre-existing local order that the gap-fill will discover
    # has terminated.
    _, [gap_closed] = state.allocate(
        LimitOrderTicket(
            symbol=btcusdt, side=OrderSide.BUY,
            quantity=Decimal('0.05'), price=Decimal('30000'),
            time_in_force=TimeInForce.GTC,
        )
    )
    state.update_order(
        gap_closed,
        status=OrderStatus.CREATED,
        updated_at=datetime(2023, 11, 1),
        id='gap-closed',
        filled_quantity=DECIMAL_ZERO,
        quote_quantity=DECIMAL_ZERO,
        commission_asset=None,
        commission_quantity=None,
    )

    # WS dropped, reconnect — openOrders shows local-known still open
    # plus a brand-new order placed during the gap, but no longer shows
    # gap-closed (it terminated during the gap).
    open_orders_payload = [
        _snapshot_item(
            clientOrderId='local-known',
            status='PARTIALLY_FILLED',
            executedQty='0.03',
            cummulativeQuoteQty='900',
            updateTime=1_700_000_500_000,
        ),
        _snapshot_item(
            clientOrderId='gap-discovered-1',
            status='PARTIALLY_FILLED',
            origQty='0.2',
            executedQty='0.05',
            cummulativeQuoteQty='1500',
            time=1_700_000_400_000,
            updateTime=1_700_000_500_000,
        ),
    ]

    exchange_open_ids = set()
    for item in open_orders_payload:
        sym = state.get_symbol(item['symbol'])
        assert sym is not None
        cid = item['clientOrderId']
        exchange_open_ids.add(cid)
        existing = state.get_order_by_id(cid)
        if existing is None:
            exc, order = decode_order_snapshot(item, symbol=sym)
            assert exc is None
            state.import_order(order)
        else:
            exc, decoded = decode_order_query_response(item)
            assert exc is None
            _, kwargs = decoded
            state.update_order(existing, **kwargs)

    # local-known should now be PARTIALLY_FILLED.
    assert known.status is OrderStatus.PARTIALLY_FILLED
    assert known.filled_quantity == Decimal('0.03')

    # gap-discovered-1 should now be in state.
    discovered = state.get_order_by_id('gap-discovered-1')
    assert discovered is not None
    assert discovered.status is OrderStatus.PARTIALLY_FILLED

    # Gap-fill: a local open order not in exchange_open_ids must be
    # queried individually; here we simulate the /api/v3/order
    # response saying it FILLED.
    locally_open = state.get_open_orders()
    closed_during_gap = [
        o for o in locally_open
        if o.id not in exchange_open_ids
    ]
    assert closed_during_gap == [gap_closed]
    fill_payload = _snapshot_item(
        clientOrderId='gap-closed',
        status='FILLED',
        executedQty='0.05',
        cummulativeQuoteQty='1500',
        updateTime=1_700_000_450_000,
    )
    exc, decoded = decode_order_query_response(fill_payload)
    assert exc is None
    _, kwargs = decoded
    state.update_order(gap_closed, **kwargs)
    assert gap_closed.status is OrderStatus.FILLED
    assert gap_closed not in state.get_open_orders()
