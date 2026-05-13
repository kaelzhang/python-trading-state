[![](https://github.com/kaelzhang/python-trading-state/actions/workflows/python.yml/badge.svg)](https://github.com/kaelzhang/python-trading-state/actions/workflows/python.yml)
[![](https://codecov.io/gh/kaelzhang/python-trading-state/branch/main/graph/badge.svg)](https://codecov.io/gh/kaelzhang/python-trading-state)
[![](https://img.shields.io/pypi/v/trading-state.svg)](https://pypi.org/project/trading-state/)
<!-- [![Conda version](https://img.shields.io/conda/vn/conda-forge/trading-state)](https://anaconda.org/conda-forge/trading-state) -->
<!-- [![](https://img.shields.io/pypi/l/trading-state.svg)](https://github.com/kaelzhang/python-trading-state) -->

# trading-state

`trading-state` is a small, focused Python library that models the dynamic state of a trading account: balances, positions, open orders, and PnL — under realistic exchange-like constraints.

It is **passive**: it never schedules, polls, or talks to an exchange. Every state change is a caller-driven write. It is **synchronous**: all reads return immediately. It separates **state** from **strategy** — the library owns the truth about what is held, ordered, filled, settled, and unsettled, while the caller owns the question "what should we do next?".

Highlights:
- All internal arithmetic uses `Decimal`.
- `OrderTicket` value objects are frozen; filters produce normalized copies via `dataclasses.replace`.
- Exchange / local async is reconciled by an internal `ReconciliationManager`; callers query `state.exposure(asset, include_unsettled_inflow=..., include_unsettled_outflow=...)` and `state.unsettled(asset)` to decide what they want to count.
- State never raises a business-level exception. Stale updates (status / time / filled regression) are silently dropped and emit a diagnostic `STALE_UPDATE` event. Protocol-side errors surface through `ValueOrException` returns from the `trading_state.binance.*` adapters.

## Install

```sh
$ pip install trading-state
```

## Usage

### 1) Initialize state and market data

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
state.set_price('BTCUSDT', Decimal('30000'))
state.set_notional_limit('BTC', Decimal('100000'))

state.set_balances([
    Balance('USDT', Decimal('10000'), Decimal('0'), datetime.now()),
])
```

`Balance.time` is required: it drives the reconciliation between order
fills (from the order channel) and balance snapshots (from the balance
channel).

### 2) Build a ticket, register it, and drive the lifecycle

```py
from trading_state import (
    LimitOrderTicket,
    OrderSide,
    OrderStatus,
    TimeInForce,
)

btcusdt = state._symbols.get_symbol('BTCUSDT')   # the Symbol registered above

ticket = LimitOrderTicket(
    symbol=btcusdt,
    side=OrderSide.BUY,
    quantity=Decimal('0.2'),
    price=Decimal('30000'),
    time_in_force=TimeInForce.GTC,
)

# add_order returns (exc, Order). On filter rejection you get the
# exception back via the value — state is never raised at.
exc, order = state.add_order(ticket, data={'strategy': 'momentum'})
assert exc is None

# The caller drives the state machine explicitly. update_order requires
# every keyword (pass None for fields you aren't touching).
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
    filled_quantity=Decimal('0.1'),
    quote_quantity=Decimal('3000'),
    commission_asset=None,
    commission_quantity=None,
)

state.set_balances([
    Balance('BTC', Decimal('0.1'), Decimal('0'), datetime.now()),
])

state.update_order(
    order,
    status=OrderStatus.FILLED,
    updated_at=datetime.now(),
    id=None,
    filled_quantity=Decimal('0.1'),
    quote_quantity=Decimal('3000'),
    commission_asset=None,
    commission_quantity=None,
)
```

### 3) Query exposure and unsettled flow

```py
exc, exposure_now = state.exposure(
    'BTC',
    include_unsettled_inflow=True,    # count fills the exchange has confirmed but balance has not yet caught up to
    include_unsettled_outflow=False,  # do not deduct unsettled outflows here
)

exc, flow = state.unsettled('BTC')    # diagnostic only — do not drive trading decisions from this
```

Each `include_unsettled_*` flag is required: callers must state at every
call site which components of the holding they want.

### 4) Best-effort allocation across alt account currencies

```py
state.set_alt_currency_weights((
    (Decimal('0.5'),),   # BUY weights against `alt_account_currencies`
    (Decimal('0'),),     # SELL weights
))

# allocate splits a canonical ticket across the configured account
# currencies and returns filter-applied sub-tickets. When it can't
# split (weights unset, no eligible bucket, unsupported ticket kind,
# etc.) it returns [ticket] so the caller has nothing to special-case.
sub_tickets = state.allocate(ticket)
for t in sub_tickets:
    exc, order = state.add_order(t)
```

### 5) Subscribe to events and diagnostics

```py
from trading_state import StaleUpdate, TradingStateEvent

state.on(
    TradingStateEvent.PERFORMANCE_SNAPSHOT_RECORDED,
    lambda snapshot: ...,
)

state.on(
    TradingStateEvent.STALE_UPDATE,
    lambda event: print('dropped stale update', event.kind, event),
)
```

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

```py
from trading_state import InvalidExchangeData
from trading_state.binance import (
    decode_order_update_event,
)

exc, decoded = decode_order_update_event(payload)
if exc is not None:
    log.error('bad executionReport', err=exc)   # caller decides:
    return                                       # raise / log / retry
client_id, updates = decoded

order = state.get_order_by_id(client_id)
if order is None:
    return                                       # unknown order
state.update_order(order, **updates)             # state silently
                                                 # drops stale data
```

All decoders in `trading_state.binance` return `ValueOrException[T]`;
validation is embedded so callers cannot accidentally feed malformed
data into state.
