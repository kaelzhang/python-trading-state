[![](https://github.com/kaelzhang/python-trading-state/actions/workflows/python.yml/badge.svg)](https://github.com/kaelzhang/python-trading-state/actions/workflows/python.yml)
[![](https://codecov.io/gh/kaelzhang/python-trading-state/branch/main/graph/badge.svg)](https://codecov.io/gh/kaelzhang/python-trading-state)
[![](https://img.shields.io/pypi/v/trading-state.svg)](https://pypi.org/project/trading-state/)
<!-- [![Conda version](https://img.shields.io/conda/vn/conda-forge/trading-state)](https://anaconda.org/conda-forge/trading-state) -->
<!-- [![](https://img.shields.io/pypi/l/trading-state.svg)](https://github.com/kaelzhang/python-trading-state) -->

# trading-state

`trading-state` is a small, focused Python library that models the dynamic state of a trading account — balances, positions, open orders, fills, unsettled flow, exposure, and PnL — under realistic exchange-like constraints.

It is **passive**: the library never schedules, polls, or talks to an exchange. Every state change is the consequence of a caller-driven write. It is **synchronous**: all reads return immediately. It separates **state** from **strategy** — the library owns the truth about what is held, ordered, filled, settled, and unsettled, while the caller owns the question "what should we do next?".

---

## Project positioning & design principles

These principles are load-bearing. Reading them once will save you from arguing with the API later.

### 1. Single entry point for orders: `state.allocate(...)`

There is **one** way to create an Order: `state.allocate(ticket)`. There is no `add_order`, no `submit`, no shortcut. `allocate` runs:

1. Setup checks (fail-fast on missing config — see below).
2. Cross-currency split across configured account currencies (when the ticket type supports it).
3. Per-bucket filter normalization against the symbol's `exchangeInfo` filters.
4. Per-bucket pre-flight (BUY notional, free balance).
5. Materializes each surviving sub-ticket into a registered `Order` and emits `ORDER_CREATED`.

The caller drives every subsequent state transition through `state.update_order(order, ...)`.

### 2. Recovery is a separate path: `state.import_order(order)`

When a process restarts, reconnects after a WS gap, or polls for an order placed by another client, the order may already exist at the exchange but be missing from local state. `state.import_order(order)` is the recovery entry point. The Binance decoders (`decode_order_snapshot`, `decode_order_query_response`) produce the right inputs.

Recovery and allocation are kept separate so neither path needs to know about the other's preconditions.

### 3. All parameters are passed explicitly

Required parameters are keyword-only. There are very few defaults, and they are reserved for opaque metadata bags (`Order.data`, `allocate(data=...)`). This is deliberate: real-money trading is unforgiving of "I forgot to pass that argument and the library used a default I didn't know about".

`state.update_order(...)` is the strictest example — every field is a required keyword. Pass `None` if you do not want to update a field; the library will not guess.

### 4. Initialization is exhaustive

Before `allocate` will succeed, the caller must have set:

- `set_symbol(symbol)` for every symbol the strategy touches
- `set_price(symbol_name, price)` for every symbol used in valuation paths
- `set_notional_limit(asset, limit)` for every base asset the strategy takes exposure on
- `set_alt_currency_weights(((BUY_weights,), (SELL_weights,)))` — mandatory, even if every alt weight is zero
- `set_balances([Balance(asset, free, locked, time)])` — `Balance.time` is required, not optional

A missing setup step is a **fail-fast** error: `AllocationWeightsNotSetError`, `SymbolNotDefinedError`, `NotionalLimitNotSetError`, etc. The library will not silently work around missing config.

### 5. Fail-fast on setup, best-effort on business outcomes

- **Fail-fast** = the caller's setup or call signature is wrong. Returns `(exc, None)` so the caller cannot accidentally drop the error.
- **Best-effort** = a per-bucket business outcome (filter rejection, insufficient balance, aggregate notional cap reached). Returns a shorter result list — possibly empty. The caller decides whether the partial result is good enough.

The split keeps "did the caller break the contract?" cleanly separate from "did the market refuse to honour this allocation?".

### 6. State never raises a business exception

`state.update_order(...)` silently drops stale data (status / time / filled regression) and emits a diagnostic `STALE_UPDATE` event. Protocol-side errors (a malformed wire payload) surface through `ValueOrException` returns from `trading_state.binance.*` decoders, never as raises from state.

### 7. `trading_state` is exchange-agnostic

The core package knows nothing about Binance. All wire ↔ domain mapping lives in `trading_state.binance`. If you add a second exchange, mirror the pattern in a sibling subpackage.

### 8. `Decimal` arithmetic everywhere

All internal money math uses `Decimal`. Tickets are frozen value objects; filters produce normalized copies via `dataclasses.replace`.

### 9. Caller chooses what counts as "exposure"

`state.exposure(asset, include_unsettled_inflow=..., include_unsettled_outflow=...)` requires both flags at every call site. There is no default. Trading decisions depend on which components of the holding you consider "real" right now; the library refuses to guess.

---

## Install

```sh
$ pip install trading-state
```

## Usage

### 1) Initialize state, market data, and weights

```py
from datetime import datetime
from decimal import Decimal

from trading_state import (
    Balance,
    Symbol,
    TradingConfig,
    TradingState,
)

config = TradingConfig(
    account_currency='USDT',
    alt_account_currencies=('USDC',),
    benchmark_assets=('BTC',),
)
state = TradingState(config)

state.set_symbol(Symbol('BTCUSDT', 'BTC', 'USDT'))
state.set_symbol(Symbol('BTCUSDC', 'BTC', 'USDC'))
state.set_price('BTCUSDT', Decimal('30000'))
state.set_price('BTCUSDC', Decimal('30000'))
state.set_notional_limit('BTC', Decimal('100000'))

# Mandatory: weights for the alternative account currencies on BUY
# and SELL. The primary account currency has implicit weight 1. Use
# zero across the alts if you do not want cross-currency splitting.
state.set_alt_currency_weights((
    (Decimal('0.5'),),   # BUY weights vs alt_account_currencies
    (Decimal('0'),),     # SELL weights
))

state.set_balances([
    Balance('USDT', Decimal('10000'), Decimal('0'), datetime.now()),
    Balance('USDC', Decimal('10000'), Decimal('0'), datetime.now()),
])
```

`Balance.time` is required: it drives the reconciliation between order fills (from the order channel) and balance snapshots (from the balance channel).

### 2) Allocate Orders from a canonical ticket

```py
from trading_state import (
    LimitOrderTicket,
    OrderSide,
    OrderStatus,
    TimeInForce,
)

btcusdt = state.get_symbol('BTCUSDT')

ticket = LimitOrderTicket(
    symbol=btcusdt,
    side=OrderSide.BUY,
    quantity=Decimal('0.2'),
    price=Decimal('30000'),
    time_in_force=TimeInForce.GTC,
)

# allocate returns ValueOrException[List[Order]]. Each Order is
# already registered with the order manager, the reconciliation
# manager, and has had `ORDER_CREATED` emitted once. The caller
# drives each Order's state machine via update_order.
exc, orders = state.allocate(ticket, data={'strategy': 'momentum'})
assert exc is None

for order in orders:
    # The caller drives the state machine explicitly. update_order
    # requires every keyword — pass None for fields you aren't touching.
    state.update_order(
        order,
        status=OrderStatus.SUBMITTING,
        updated_at=None,
        id=None,
        filled_quantity=None,
        quote_quantity=None,
        commission_asset=None,
        commission_quantity=None,
    )

    state.update_order(
        order,
        status=OrderStatus.CREATED,
        updated_at=datetime.now(),
        id='order-1',
        filled_quantity=Decimal('0.05'),
        quote_quantity=Decimal('1500'),
        commission_asset=None,
        commission_quantity=None,
    )
```

`exc is None` and `orders == []` is a legal outcome: every per-bucket pre-flight (filter / balance / aggregate notional) was best-effort skipped. The library does NOT signal "I rejected your ticket"; the caller inspects the empty list and decides whether to retry with smaller size, different weights, or to wait.

Failure modes:
- `(AllocationWeightsNotSetError(), None)` — `set_alt_currency_weights` was never called.
- `(SymbolNotDefinedError(name), None)` — `ticket.symbol` was never registered via `set_symbol`.
- `(NotionalLimitNotSetError(asset), None)` — the BUY pre-flight tried to query exposure on an asset without a notional limit.
- `(None, [orders...])` — happy path; may be a shorter list than expected, or empty.

### 3) Query exposure and unsettled flow

```py
exc, exposure_now = state.exposure(
    'BTC',
    include_unsettled_inflow=True,    # count fills the exchange has confirmed but balance has not yet caught up to
    include_unsettled_outflow=False,  # do not deduct unsettled outflows here
)

exc, flow = state.unsettled('BTC')    # diagnostic only — do not drive trading decisions from this
```

Each `include_unsettled_*` flag is required: callers must state at every call site which components of the holding they want.

### 4) Cross-currency allocation across alt account currencies

```py
state.set_alt_currency_weights((
    (Decimal('0.5'),),   # BUY weights against `alt_account_currencies`
    (Decimal('0'),),     # SELL weights
))

# A canonical ticket on the primary symbol is fanned out across every
# weighted account-currency bucket. allocate produces one Order per
# eligible bucket, in `config.account_currencies` declaration order.
exc, orders = state.allocate(ticket)
for order in orders:
    print(order.ticket.symbol.name, order.ticket.quantity)
```

Per-bucket failures (filter rejection, insufficient balance) silently skip that bucket; surviving buckets still produce Orders.

### 5) Subscribe to events and diagnostics

```py
from trading_state import StaleUpdate, TradingStateEvent

state.on(
    TradingStateEvent.ORDER_CREATED,
    lambda order: print('order created', order.id, order.ticket.symbol.name),
)

state.on(
    TradingStateEvent.PERFORMANCE_SNAPSHOT_RECORDED,
    lambda snapshot: ...,
)

state.on(
    TradingStateEvent.STALE_UPDATE,
    lambda event: print('dropped stale update', event.kind, event),
)
```

Every Order in state was preceded by an `ORDER_CREATED` event — whether it came from `allocate` or from `import_order` (recovery). Downstream subscribers do not need to differentiate.

### 6) Record snapshots and analyze performance

```py
from trading_state import CashFlow

state.set_cash_flow(
    CashFlow('USDT', Decimal('1000'), datetime.now()),
)

snapshot = state.record()
```

```py
from trading_state import TradingStateEvent
from trading_state.analyzer import AnalyzerType, PerformanceAnalyzer

analyzer = PerformanceAnalyzer([
    AnalyzerType.TOTAL_RETURN,
    AnalyzerType.SHARPE_RATIO.params(trading_days=365),
    AnalyzerType.MAX_DRAWDOWN,
])

state.on(
    TradingStateEvent.PERFORMANCE_SNAPSHOT_RECORDED,
    analyzer.add_snapshots,
)

results = analyzer.analyze()
total_return = results[AnalyzerType.TOTAL_RETURN].value
```

### 7) Bridging to Binance (or any exchange) via decoders

WS executionReport → `state.update_order`:

```py
from trading_state.binance import decode_order_update_event

exc, decoded = decode_order_update_event(ws_payload)
if exc is not None:
    log.error('bad executionReport', err=exc)
    return
client_id, updates = decoded

order = state.get_order_by_id(client_id)
if order is None:
    return  # unknown order; consider recovery (Section 8)
state.update_order(order, **updates)
```

REST `POST /api/v3/order` response → `state.update_order`:

```py
from trading_state.binance import decode_order_create_response

exc, updates = decode_order_create_response(rest_response)
if exc is None:
    state.update_order(order, **updates)
```

All decoders in `trading_state.binance` return `ValueOrException[T]`; validation is embedded so callers cannot accidentally feed malformed data into state.

### 8) Recovery

`trading_state.binance` provides two recovery-oriented decoders on top of the same wire schema. The caller chooses which one based on whether the Order is already in state.

```py
from trading_state.binance import (
    decode_order_query_response,
    decode_order_snapshot,
)
```

| Decoder | Returns | Pair with |
|---|---|---|
| `decode_order_snapshot(item, *, symbol, data=None)` | `Order` (fully populated) | `state.import_order(order)` |
| `decode_order_query_response(payload)` | `(client_order_id, update_kwargs)` | `state.update_order(existing, **kwargs)` |

Both consume the per-order schema returned by `GET /api/v3/openOrders`, `GET /api/v3/allOrders`, and the body of `GET /api/v3/order` (single-order query).

#### 8a) Cold startup

State is empty. Pull `openOrders`, import each:

```py
account = await binance.get_account()
state.set_balances(decode_account_info_response(account))

for item in await binance.get_open_orders():
    sym = state.get_symbol(item['symbol'])
    if sym is None:
        continue
    exc, order = decode_order_snapshot(item, symbol=sym)
    if exc is not None:
        log.error('snapshot decode failed', err=exc)
        continue
    state.import_order(order)
```

#### 8b) Periodic check of one specific order

Caller is tracking a long-lived order and wants a status refresh. Order is known to be in state, so use `update_order`:

```py
order = state.get_order_by_id('order-1')
item = await binance.get_order(
    orig_client_order_id='order-1',
    symbol=order.ticket.symbol.name,
)
exc, decoded = decode_order_query_response(item)
if exc is None:
    _, kwargs = decoded
    state.update_order(order, **kwargs)
```

The `update_order` path naturally handles terminal status (FILLED / CANCELED / REJECTED): it just transitions the existing Order through `update_order`'s standard logic, including the stale-update silent drop.

#### 8c) Mid-session reconnect after a WS gap

WS has no replay. After reconnect, pull `openOrders` and dispatch per item on existing-in-state, then gap-fill local-open orders that are no longer in the exchange's open set:

```py
api_open = await binance.get_open_orders()
exchange_open_ids = set()

# (1) For each open at exchange: import if new, refresh if known.
for item in api_open:
    sym = state.get_symbol(item['symbol'])
    if sym is None:
        continue
    cid = item['clientOrderId']
    exchange_open_ids.add(cid)
    existing = state.get_order_by_id(cid)
    if existing is None:
        exc, order = decode_order_snapshot(item, symbol=sym)
        if exc is None:
            state.import_order(order)
    else:
        exc, decoded = decode_order_query_response(item)
        if exc is None:
            _, kwargs = decoded
            state.update_order(existing, **kwargs)

# (2) Locally open but not in the exchange's open set must have
#     terminated during the gap; query each one and push the
#     terminal status via update_order.
for order in state.get_open_orders():
    if order.id in exchange_open_ids:
        continue
    item = await binance.get_order(
        orig_client_order_id=order.id,
        symbol=order.ticket.symbol.name,
    )
    exc, decoded = decode_order_query_response(item)
    if exc is None:
        _, kwargs = decoded
        state.update_order(order, **kwargs)

# (3) Refresh balances.
account = await binance.get_account()
state.set_balances(decode_account_info_response(account))
```

`state.get_open_orders()` returns the locally-known orders the exchange has acknowledged (status `CREATED`, `PARTIALLY_FILLED`, `CANCELLING`). It does **not** include orders in `INIT` or `SUBMITTING` — those are still caller-side in-flight; the caller holds the reference returned by `allocate` and is responsible for driving them forward.

#### Allocation pre-flight semantics

`allocate`'s passthrough flow (`MarketOrderTicket(QUOTE)`, stop-loss / take-profit variants) always applies filter normalization but only runs pre-flight on amounts that are precisely computable. Anything that depends on the trigger-time market price is silently allowed through — the exchange does the authoritative check at trigger time.

| Ticket | notional pre-flight | free-balance pre-flight |
|---|---|---|
| `MarketOrderTicket(QUOTE)` BUY | yes (notional = quote_quantity) | yes (quote currency) |
| `MarketOrderTicket(QUOTE)` SELL | N/A | skipped (base quantity unknown) |
| `StopLossLimit / TakeProfitLimit` BUY | yes (notional = price × quantity) | yes (quote currency) |
| `StopLoss / TakeProfit` (no limit) BUY | skipped (depends on trigger price) | skipped |
| `StopLoss / TakeProfit / StopLossLimit / TakeProfitLimit` SELL | N/A | yes (base quantity known) |
