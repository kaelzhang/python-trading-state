[![](https://github.com/kaelzhang/python-trading-state/actions/workflows/python.yml/badge.svg)](https://github.com/kaelzhang/python-trading-state/actions/workflows/python.yml)
[![](https://codecov.io/gh/kaelzhang/python-trading-state/branch/master/graph/badge.svg)](https://codecov.io/gh/kaelzhang/python-trading-state)
[![](https://img.shields.io/pypi/v/trading-state.svg)](https://pypi.org/project/trading-state/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/trading-state)](https://anaconda.org/conda-forge/trading-state)
[![](https://img.shields.io/pypi/l/trading-state.svg)](https://github.com/kaelzhang/python-trading-state)

# trading-state

`trading-state` is a small, focused Python library that models the dynamic state of a trading account, including balances, positions, open orders, and PnL, under realistic exchange-like constraints.

It provides configurable rules for price limits, notional and quantity limits, order validation, and order lifecycle transitions (new, partially filled, filled, canceled, rejected), making it easy to plug into backtesting frameworks, execution simulators, or custom quant research tools.

By separating “trading state” from strategy logic, `trading-state` helps you build cleaner, more testable quantitative trading systems.

## Install

```sh
$ pip install trading-state
```

## Usage

```py
from trading_state import TradingState
```

## License

[MIT](LICENSE)
