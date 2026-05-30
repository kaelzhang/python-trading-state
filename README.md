[![](https://github.com/kaelzhang/python-trading-state/actions/workflows/python.yml/badge.svg)](https://github.com/kaelzhang/python-trading-state/actions/workflows/python.yml)
[![](https://codecov.io/gh/kaelzhang/python-trading-state/branch/main/graph/badge.svg)](https://codecov.io/gh/kaelzhang/python-trading-state)
[![](https://img.shields.io/pypi/v/trading-state.svg)](https://pypi.org/project/trading-state/)
<!-- [![Conda version](https://img.shields.io/conda/vn/conda-forge/trading-state)](https://anaconda.org/conda-forge/trading-state) -->
<!-- [![](https://img.shields.io/pypi/l/trading-state.svg)](https://github.com/kaelzhang/python-trading-state) -->

# trading-state

`trading-state` is a small, focused Python library that models the dynamic state of a trading account — balances, positions, open orders, fills, unsettled flow, exposure, and PnL — under realistic exchange-like constraints.

The library is **passive**: it never schedules, polls, or talks to an exchange. The caller owns the network — when an exchange responds to a request, or pushes an event over a WebSocket, the caller hands that data to `trading_state`. All reads are **synchronous** and return immediately.

## Install

```sh
$ pip install trading-state
```

## Usage

### 1) Set up the state

Before placing any order, configure the account, register every Symbol you will trade, set notional limits, set cross-currency allocation weights, then seed initial balances. The library does not fall back to defaults for any of these — a missing setup step makes `state.allocate(...)` fail fast with a specific exception, by design.

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
    # Alt account currencies should be stablecoins pegged to the
    # primary account currency (USDC, FDUSD, DAI, etc.). They are the
    # other quote currencies you can route into when you want to split
    # a trade for depth — the splitting math assumes USDT≈USDC≈primary
    # in value. Non-stablecoin alts will systematically bias the split.
    alt_account_currencies=('USDC',),
    benchmark_assets=('BTC',),
)
state = TradingState(config)

# Register every symbol the strategy will trade.
state.set_symbol(Symbol('BTCUSDT', 'BTC', 'USDT'))
state.set_symbol(Symbol('BTCUSDC', 'BTC', 'USDC'))

# Seed prices used by the valuation paths.
state.set_price('BTCUSDT', Decimal('30000'))
state.set_price('BTCUSDC', Decimal('30000'))

# Hard cap on how much BTC notional the strategy is allowed to hold.
# Required for every base asset the strategy takes exposure on.
state.set_notional_limit('BTC', Decimal('100000'))

# Initial balances from the exchange's account snapshot.
# Balance.time is required: it lets the library tell unsettled flow
# (fills not yet reflected in a balance update) from settled balance.
state.set_balances([
    Balance('USDT', Decimal('10000'), Decimal('0'), datetime.now()),
    Balance('USDC', Decimal('10000'), Decimal('0'), datetime.now()),
])
```

### 2) Place a trade and keep state in sync with the exchange

Suppose your strategy decided to buy 0.2 BTC at 30000 USDT. Encode that decision as an `OrderTicket` and hand it to `state.create_order(...)`. The library registers the resulting `Order` (or `Order`s, if you asked it to split — see below) and emits `ORDER_CREATED` for each. From there on, the caller drives every transition through `state.update_order(...)`.

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

# `allocate` is a required keyword: pass `None` to skip cross-currency
# splitting and create a single Order on the ticket's symbol. The
# library runs filter normalization and BUY pre-flight (notional cap
# + free balance) regardless of which path you take.
exc, orders = state.create_order(
    ticket,
    data={'strategy': 'momentum'},
    allocate=None,
)
if exc is not None:
    # Setup error (symbol not registered, notional limit not set,
    # allocate vector malformed when splitting, etc.).
    ...
elif not orders:
    # Best-effort outcome: filter or pre-flight knocked the order
    # out; nothing landed in state.
    ...
```

For each returned `Order`, push it through the exchange's order-create endpoint, then push the exchange's response back into state. The library does not perform any network I/O — that is the caller's exchange client.

```py
from trading_state.binance import (
    decode_order_create_response,
    encode_order_request,
)

for order in orders:
    # 1. Mark the Order as being submitted to the exchange. This
    #    transitions it from INIT to SUBMITTING. update_order requires
    #    every keyword — pass None for fields you are not touching.
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

    # 2. Encode the ticket for the exchange's POST body.
    exc, body = encode_order_request(order.ticket)
    if exc is not None:
        continue

    # 3. Send to the exchange. trading_state is passive; this network
    #    call is owned by your HTTP client.
    response = exchange_client.post('/api/v3/order', body=body)

    # 4. Decode the response into update_order's keyword set and push
    #    it back into state. The id, status, and fill cumulatives all
    #    come from the exchange — never hand-write them.
    exc, kwargs = decode_order_create_response(response)
    if exc is None:
        state.update_order(order, **kwargs)
```

After the create response lands, subsequent state changes for this Order — partial fills, full fill, cancellation — typically arrive over the user-data WebSocket as `executionReport` events. When one arrives, decode it, look up the Order by its client-order-id, and feed the result to `update_order`.

```py
from trading_state.binance import decode_order_update_event

# When an `executionReport` arrives over the user-data WS (your WS
# infrastructure is out of scope; trading_state never opens sockets):
exc, decoded = decode_order_update_event(ws_payload)
if exc is None:
    client_order_id, kwargs = decoded
    order = state.get_order_by_id(client_order_id)
    if order is not None:
        state.update_order(order, **kwargs)
    # If order is None, the Order is not in local state — see §4 Recovery.
```

`state.update_order` silently drops stale writes (status / time / filled-quantity regressions) and emits a diagnostic `STALE_UPDATE` event. It never raises a business exception.

#### Splitting a single trade across alt account currencies

When the strategy wants to distribute a trade across the configured alt account currencies (to source depth from multiple books and reduce slippage), pass the per-call weights via `allocate=`. The vector is aligned with `config.alt_account_currencies`; the primary account currency has implicit weight 1. Weights are intentionally per-call because in production they are recomputed every decision from live inputs (book depth, stablecoin balances, basis, inventory skew) — none of which the library holds.

Two patterns work, pick whichever fits your routing logic:

**Delegate the split to trading_state.** Provide a weights vector; the library splits the canonical ticket across alt buckets, filter-normalizes each, runs per-bucket pre-flight, and registers one Order per surviving bucket.

```py
weights = my_router.compute_buy_weights(live_book_depth, free_balances)
exc, orders = state.create_order(ticket, allocate=weights)
# orders has one entry per surviving account-currency bucket, in
# `config.account_currencies` declaration order. The loop above
# (SUBMITTING -> encode -> POST -> decode -> update_order) works
# unchanged.
```

**Split caller-side, then submit each sub-ticket without further splitting.** Useful when your router does something the library's split math doesn't model (e.g., venue-aware sizing, latency-weighted shares).

```py
sub_tickets = my_router.split(ticket, live_book_depth)
for sub_ticket in sub_tickets:
    exc, orders = state.create_order(sub_ticket, allocate=None)
    # ... drive each order through the SUBMITTING -> POST -> ... loop
```

#### Split reference and pre-flight per ticket type

When `allocate=` triggers the split flow, each ticket type contributes a different price field as the split reference; the pre-flight checks below use the same reference:

| Ticket | split reference | aggregate BUY notional check | per-bucket BUY quote balance | aggregate SELL base balance |
|---|---|---|---|---|
| `LimitOrderTicket` (incl. post_only) | `price` | precise | yes | yes |
| `MarketOrderTicket(BASE)` | `estimated_price` | precise | yes | yes |
| `MarketOrderTicket(QUOTE)` | `estimated_price` | precise (notional = quantity) | yes | yes (uses `quantity / estimated_price`) |
| `StopLossLimit` / `TakeProfitLimit` | `price` (limit) | precise | yes | yes |
| `StopLoss` / `TakeProfit` (no limit) | `stop_price` (or symbol's last `set_price` if only `trailing_delta` is set) | **estimate** — actual fill price at trigger may differ | estimate | yes |

For bare stop / take-profit BUYs, the notional pre-flight is an estimate (real fill happens at trigger-time market price). An estimated overshoot results in an empty result list, accepted as the trade-off for catching oversize orders before they hit the exchange.

Stop / take-profit splits inherit a basis-asymmetry risk: in a fast move, one alt symbol may trigger before another, leaving the protective intent partially honoured. The library does not model OCO (one-cancels-other); using split stops means accepting this trade-off in exchange for distributed depth. If that is unacceptable for your strategy, pass `allocate=None` so the stop stays on a single market.

### 3) Check exposure when sizing the next trade

Before deciding on the next order size, the strategy typically needs to know how much room is left under the asset's notional cap, accounting for what is held now and for any fills that have not yet been reflected in a balance snapshot.

`state.exposure(asset, ...)` returns an `Exposure` value object. Both `include_unsettled_*` flags are required at every call site — there is no library default, because what counts as "real" depends on whether you are sizing a new BUY (conservative: include unsettled inflows so you don't double-spend the room), a SELL (include outflows), or a diagnostic read (your call).

```py
exc, exposure = state.exposure(
    'BTC',
    include_unsettled_inflow=True,
    include_unsettled_outflow=False,
)
if exc is None:
    room_left = exposure.notional_limit - exposure.notional_value
```

`state.unsettled(asset)` returns just the unsettled inflow / outflow magnitudes — it is a diagnostic read only. Do not drive trading decisions from it; go through `state.exposure(...)` with explicit flags.

### 4) Recover after restart or a WebSocket gap

The user-data WebSocket does not replay missed events. After a process restart or a WS drop, local state can drift from the exchange. `trading_state.binance` provides two recovery decoders that consume the same per-order schema returned by `GET /api/v3/openOrders`, `GET /api/v3/allOrders`, and `GET /api/v3/order`; the caller chooses the right one based on whether the Order is already in state.

| Decoder | Returns | Pair with |
|---|---|---|
| `decode_order_snapshot(item, *, symbol, data=None)` | a fully-populated `Order` | `state.import_order(order)` |
| `decode_order_query_response(payload)` | `(client_order_id, update_kwargs)` | `state.update_order(existing, **kwargs)` |

#### 4a) Cold startup

The process just launched; state is empty. Pull every open order from the exchange and import each one.

```py
from trading_state.binance import (
    decode_account_info_response,
    decode_order_snapshot,
)

# Seed balances from the account snapshot.
account = exchange_client.get_account()
exc, balances = decode_account_info_response(account)
if exc is None:
    state.set_balances(balances)

# Import every open order.
for item in exchange_client.get_open_orders():
    sym = state.get_symbol(item['symbol'])
    if sym is None:
        continue
    exc, order = decode_order_snapshot(item, symbol=sym)
    if exc is None:
        state.import_order(order)
```

#### 4b) Periodic refresh of one known order

The strategy is tracking a long-lived order (the local state knows it) and wants a status refresh. The Order is already in state, so route through `update_order`.

```py
from trading_state.binance import decode_order_query_response

order = state.get_order_by_id('order-1')
item = exchange_client.get_order(
    orig_client_order_id='order-1',
    symbol=order.ticket.symbol.name,
)
exc, decoded = decode_order_query_response(item)
if exc is None:
    _, kwargs = decoded
    state.update_order(order, **kwargs)
```

`update_order` handles terminal status naturally: if the refresh returns FILLED, it transitions the Order through the same path it would have taken via WS, including the stale-update silent drop.

#### 4c) Mid-session reconnect after a WS gap

WS reconnected; some local Orders may have advanced or terminated during the gap, and orders placed by another client / process during the gap need discovering. Pull `openOrders` and dispatch per item on existing-in-state, then gap-fill local-open orders that are no longer in the exchange's open set.

```py
api_open = exchange_client.get_open_orders()
exchange_open_ids = set()

# (1) For each open order at the exchange: import if new, refresh if known.
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

# (2) Local Orders that the exchange no longer lists as open must
#     have terminated during the gap; query each one and push the
#     terminal status via update_order.
for order in state.get_open_orders():
    if order.id in exchange_open_ids:
        continue
    item = exchange_client.get_order(
        orig_client_order_id=order.id,
        symbol=order.ticket.symbol.name,
    )
    exc, decoded = decode_order_query_response(item)
    if exc is None:
        _, kwargs = decoded
        state.update_order(order, **kwargs)

# (3) Refresh balances.
account = exchange_client.get_account()
exc, balances = decode_account_info_response(account)
if exc is None:
    state.set_balances(balances)
```

`state.get_open_orders()` returns Orders the exchange has acknowledged (`CREATED` / `PARTIALLY_FILLED` / `CANCELLING`). It does **not** include `INIT` / `SUBMITTING` — those Orders are still caller-side in-flight; the caller holds the reference returned by `allocate` and is responsible for driving them forward.

### 5) Observe state changes via events

For dashboards, audit logs, or downstream pipelines that should not sit on the trading hot path, subscribe to `TradingStateEvent`s.

```py
from trading_state import TradingStateEvent

state.on(
    TradingStateEvent.ORDER_CREATED,
    lambda order: ...,
)
state.on(
    TradingStateEvent.ORDER_STATUS_UPDATED,
    lambda order, status: ...,
)
state.on(
    TradingStateEvent.ORDER_FILLED_QUANTITY_UPDATED,
    lambda order, filled_quantity: ...,
)
state.on(
    TradingStateEvent.STALE_UPDATE,
    lambda event: ...,
)
```

Every Order in state — whether created via `allocate` or injected via `import_order` (§4) — was preceded by an `ORDER_CREATED`, so a single subscription covers both paths.

`STALE_UPDATE` fires whenever state silently drops a write that would regress a monotonic invariant (out-of-order status, filled-quantity going backwards, balance snapshot older than current). It is the main observability hook for "is my exchange feed misbehaving?".

### 6) Record snapshots and analyze performance

For backtesting or live PnL tracking, periodically take a performance snapshot — typically at the close of each decision window — and feed the snapshot stream to a `PerformanceAnalyzer`.

```py
from trading_state import CashFlow

# Account-level external cash flow (deposit / withdrawal).
state.set_cash_flow(
    CashFlow('USDT', Decimal('1000'), datetime.now()),
)

# Snapshot the current account value, holdings, and metrics.
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

# Push every recorded snapshot into the analyzer as soon as state
# emits it.
state.on(
    TradingStateEvent.PERFORMANCE_SNAPSHOT_RECORDED,
    analyzer.add_snapshots,
)

results = analyzer.analyze()
total_return = results[AnalyzerType.TOTAL_RETURN].value
```

### 7) Binance decoders reference

Every wire ↔ domain mapping for Binance Spot lives in `trading_state.binance`. Each channel pairs with a `state` method as follows. All decoders return `ValueOrException[T]`; validation is embedded so malformed wire data cannot accidentally land in state.

| Direction | Channel | Function | Output | Pair with |
|---|---|---|---|---|
| → exchange | `POST /api/v3/order` body | `encode_order_request(ticket)` | `dict` | (your HTTP client) |
| ← exchange | `POST /api/v3/order` response | `decode_order_create_response(response)` | `update_kwargs` | `state.update_order(order, **kwargs)` |
| ← exchange | `GET /api/v3/order` response (also a single item from `/openOrders` / `/allOrders`) | `decode_order_query_response(payload)` | `(client_order_id, update_kwargs)` | `state.update_order(existing, **kwargs)` |
| ← exchange | `/openOrders` / `/allOrders` item (recovery — order not in state) | `decode_order_snapshot(item, *, symbol, data=None)` | `Order` | `state.import_order(order)` |
| ← exchange | WS `executionReport` | `decode_order_update_event(payload)` | `(client_order_id, update_kwargs)` | `state.update_order(order, **kwargs)` |
| ← exchange | WS `outboundAccountPosition` | `decode_account_update_event(payload)` | `Set[Balance]` | `state.set_balances(balances)` |
| ← exchange | WS `balanceUpdate` | `decode_balance_update_event(payload)` | `CashFlow` | `state.set_cash_flow(cash_flow)` |
| ← exchange | `GET /api/v3/account` response | `decode_account_info_response(response)` | `Set[Balance]` | `state.set_balances(balances)` |
| ← exchange | `GET /api/v3/exchangeInfo` response | `decode_exchange_info_response(response)` | `Set[Symbol]` | `state.set_symbol(symbol)` per item |
