from datetime import datetime
from decimal import Decimal

from stock_pandas import StockDataFrame

from trading_state import (
    LimitOrderTicket,
    TradingStateEvent,
    TradingState,
    TradingConfig,
    Balance,
    OrderSide,
    OrderStatus,
    OrderType,
    CashFlow,
    TimeInForce,
)
from trading_state.analyzer import (
    AnalyzerType,
    PerformanceAnalyzer,
)
from trading_state.common import (
    DECIMAL_ZERO,
    DECIMAL_ONE,
)

from .fixtures import (
    get_stock,
    init_state,
    DEFAULT_CONFIG_KWARGS,
    BTC,
    USDT,
    BTCUSDT,
)


def test_analyzer_type():
    availables = AnalyzerType.availables()
    print(availables)


class Trader:
    _stock: StockDataFrame
    _time: datetime
    _cash_flowed: bool = False

    def __init__(
        self,
        state: TradingState,
        analyzer: PerformanceAnalyzer,
        stock: StockDataFrame,
        init_base_currency: Decimal,
        fee: Decimal,
    ):
        self._state = state
        self._analyzer = analyzer
        self._time = datetime.now()
        self._fee_loss = DECIMAL_ONE - fee
        self._stock = stock

        self._state.on(
            TradingStateEvent.PERFORMANCE_SNAPSHOT_RECORDED,
            self._analyzer.add_snapshots,
        )

        # The backtester models a pure trading account where notional
        # is not actively capped; Decimal('Infinity') satisfies the
        # must-set invariant without imposing a real ceiling.
        self._state.set_notional_limit(BTC, Decimal('Infinity'))

        # Initial balances — time must be set on every Balance.
        self._state.set_balances([
            Balance(USDT, init_base_currency, DECIMAL_ZERO, self._time),
            Balance(BTC, DECIMAL_ZERO, DECIMAL_ZERO, self._time),
        ])

        self._state.set_price(
            BTCUSDT.name,
            Decimal(str(self._stock['close'].iloc[0])),
        )

    def _cash_flow(self) -> None:
        if self._cash_flowed:
            return
        self._cash_flowed = True
        self._state.set_cash_flow(
            CashFlow(USDT, Decimal('-3000'), time=self._time),
        )

    def _buy(self, price: Decimal) -> None:
        usdt = self._state._balances.get_balance(USDT).free
        if usdt <= DECIMAL_ZERO:
            return
        quantity = usdt / price
        sym = self._state.get_symbol(BTCUSDT.name)
        ticket = LimitOrderTicket(
            symbol=sym,
            side=OrderSide.BUY,
            quantity=quantity,
            price=price,
            time_in_force=TimeInForce.GTC,
        )
        exc, orders = self._state.allocate(ticket)
        if exc is not None or not orders:
            return
        order = orders[0]

        assert order.ticket.is_a(OrderType.LIMIT, side=OrderSide.BUY)
        assert not order.ticket.is_a(
            OrderType.LIMIT, side=OrderSide.SELL
        )
        assert not order.ticket.is_a(
            OrderType.LIMIT, stop_price=DECIMAL_ONE
        )

        quantity_gain = order.ticket.quantity * self._fee_loss
        quantity_used = order.ticket.quantity * order.ticket.price

        self._state.update_order(
            order,
            status=OrderStatus.FILLED,
            updated_at=self._time,
            id=None,
            filled_quantity=quantity_gain,
            quote_quantity=quantity_used,
            commission_asset=None,
            commission_quantity=None,
        )

        self._state.set_balances([
            Balance(BTC, quantity_gain, DECIMAL_ZERO, self._time),
            Balance(USDT, -quantity_used, DECIMAL_ZERO, self._time),
        ], delta=True)

        self._state.record(time=self._time)

    def _sell(self, price: Decimal) -> None:
        btc = self._state._balances.get_balance(BTC).free
        if btc <= DECIMAL_ZERO:
            return
        sym = self._state.get_symbol(BTCUSDT.name)
        ticket = LimitOrderTicket(
            symbol=sym,
            side=OrderSide.SELL,
            quantity=btc,
            price=price,
            time_in_force=TimeInForce.GTC,
        )
        exc, orders = self._state.allocate(ticket)
        if exc is not None or not orders:
            return
        order = orders[0]

        quantity_gain = (
            order.ticket.quantity * order.ticket.price * self._fee_loss
        )
        quantity_used = order.ticket.quantity

        self._state.update_order(
            order,
            status=OrderStatus.FILLED,
            updated_at=self._time,
            id=None,
            filled_quantity=quantity_used,
            quote_quantity=quantity_gain,
            commission_asset=None,
            commission_quantity=None,
        )

        self._state.set_balances([
            Balance(USDT, quantity_gain, DECIMAL_ZERO, self._time),
            Balance(BTC, -quantity_used, DECIMAL_ZERO, self._time),
        ], delta=True)

        self._cash_flow()
        self._state.record(time=self._time)

    def go(self) -> None:
        gold_cross = StockDataFrame.directive_stringify(
            'macd // macd.signal'
        )
        dead_cross = StockDataFrame.directive_stringify(
            'macd \\ macd.signal'
        )

        self._stock[gold_cross]
        self._stock[dead_cross]

        for _, row in self._stock.iterrows():
            price = Decimal(str(row['close']))
            self._state.set_price(BTCUSDT.name, price)

            buy = row[gold_cross]
            sell = row[dead_cross]

            self._time = row.name

            if buy:
                self._buy(price)
            elif sell:
                self._sell(price)
            else:
                self._state.record(time=row.name)


def test_analyzer():
    state = init_state(
        config=TradingConfig(**{
            **DEFAULT_CONFIG_KWARGS,
            'benchmark_assets': (BTC,),
        }),
        with_balances=False,
    )

    stock = get_stock()
    init_base_currency = Decimal('10000')
    analyzer = PerformanceAnalyzer(AnalyzerType.all())
    trader = Trader(
        state,
        analyzer,
        stock,
        init_base_currency,
        Decimal('0.0075'),
    )

    trader.go()

    last_snapshot = analyzer._snapshots[-1]

    calculated_account_value = (
        init_base_currency
        + last_snapshot.realized_pnl
        + last_snapshot.unrealized_pnl
        # no cash flow
    )

    assert abs(
        last_snapshot.account_value
        - calculated_account_value
    ) < Decimal('0.000001')

    result = analyzer.analyze()

    analyzer2 = PerformanceAnalyzer([
        AnalyzerType.TOTAL_RETURN,
    ])

    analyzer2.add_snapshots(*state.performance())

    assert (
        analyzer2.analyze()[AnalyzerType.TOTAL_RETURN].value
        == result[AnalyzerType.TOTAL_RETURN].value
    )
