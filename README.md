[![](https://github.com/kaelzhang/python-trading-state/actions/workflows/python.yml/badge.svg)](https://github.com/kaelzhang/python-trading-state/actions/workflows/python.yml)
[![](https://codecov.io/gh/kaelzhang/python-trading-state/branch/main/graph/badge.svg)](https://codecov.io/gh/kaelzhang/python-trading-state)
[![](https://img.shields.io/pypi/v/trading-state.svg)](https://pypi.org/project/trading-state/)
<!-- [![Conda version](https://img.shields.io/conda/vn/conda-forge/trading-state)](https://anaconda.org/conda-forge/trading-state) -->
<!-- [![](https://img.shields.io/pypi/l/trading-state.svg)](https://github.com/kaelzhang/python-trading-state) -->

# trading-state

`trading-state` is a small, focused Python library that models the dynamic state of a trading account, including balances, positions, open orders, and PnL, under realistic exchange-like constraints.

It provides configurable rules for price limits, notional and quantity limits, order validation, and order lifecycle transitions (new, partially filled, filled, canceled, rejected), making it easy to plug into backtesting frameworks, execution simulators, or custom quant research tools.

By separating “trading state” from strategy logic, `trading-state` helps you build cleaner, more testable quantitative trading systems.

- Use `Decimal` to do all internal calculations

## Install

```sh
$ pip install trading-state
```

## Usage

Below are the most common workflows. For full parameter details, see
the docstrings in each class.

### 1) Initialize state and market data

```py
from decimal import Decimal

from trading_state import (
    TradingConfig,
    TradingState,
    Symbol,
    Balance
)

config = TradingConfig(
    account_currency='USDT',
    alt_account_currencies=('USDC',),
    benchmark_assets=('BTC',)
)
state = TradingState(config)

state.set_symbol(Symbol('BTCUSDT', 'BTC', 'USDT'))
state.set_price('BTCUSDT', Decimal('30000'))
state.set_notional_limit('BTC', Decimal('100000'))

state.set_balances([
    Balance('USDT', Decimal('10000'), Decimal('0'))
])
```

### 2) Create target exposure and manage orders

```py
from trading_state import OrderStatus

exception, updated = state.expect(
    'BTCUSDT',
    exposure=Decimal('0.2'),
    price=Decimal('30000'),
    use_market_order=False
)
assert exception is None

orders, _ = state.get_orders()
order = next(iter(orders))

# Apply updates from the exchange
state.update_order(
    order,
    status=OrderStatus.CREATED,
    id='order-1',
    filled_quantity=Decimal('0.1'),
    quote_quantity=Decimal('3000')
)

# Update balances after fills
# according to websocket incomming messages (maybe)
state.set_balances([
    Balance('BTC', Decimal('0.1'), Decimal('0'))
])

state.update_order(order, status=OrderStatus.FILLED)
```

### 3) Record performance snapshots

```py
from trading_state import TradingStateEvent, CashFlow
from datetime import datetime

def on_snapshot(snapshot):
    print(snapshot.time, snapshot.account_value, snapshot.realized_pnl)

state.on(TradingStateEvent.PERFORMANCE_SNAPSHOT_RECORDED, on_snapshot)

# Manual snapshot (e.g. end of day)
snapshot = state.record()

# External cash flow (deposit / withdrawal)
state.set_cash_flow(
    CashFlow('USDT', Decimal('1000'), datetime.utcnow())
)
```

### 4) Analyze performance

```py
from trading_state.analyzer import PerformanceAnalyzer, AnalyzerType
from trading_state import TradingStateEvent

analyzer = PerformanceAnalyzer([
    AnalyzerType.TOTAL_RETURN,
    AnalyzerType.SHARPE_RATIO.params(trading_days=365),
    AnalyzerType.MAX_DRAWDOWN
])

state.on(
    TradingStateEvent.PERFORMANCE_SNAPSHOT_RECORDED,
    analyzer.add_snapshots
)

results = analyzer.analyze()
total_return = results[AnalyzerType.TOTAL_RETURN].value
```
