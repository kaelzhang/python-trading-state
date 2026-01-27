from datetime import datetime
from decimal import Decimal

from stock_pandas import StockDataFrame

from trading_state import (
    TradingStateEvent,
    TradingState,
    TradingConfig,
    Balance,
    # Symbol,
    OrderSide,
    OrderStatus
)
from trading_state.analyzer import (
    AnalyzerType,
    PerformanceAnalyzer
)
from trading_state.common import (
    DECIMAL_ZERO,
    DECIMAL_ONE
)

from .fixtures import (
    get_stock,
    init_state,
    DEFAULT_CONFIG_KWARGS,
    BTC,
    USDT,
    BTCUSDT
)


def test_analyzer_type():
    return
    availables = AnalyzerType.availables()

    print(availables)


class Trader:
    _stock: StockDataFrame
    _time: datetime

    def __init__(
        self,
        state: TradingState,
        analyzer: PerformanceAnalyzer,
        stock: StockDataFrame,
        init_base_currency: Decimal,
        fee: Decimal
    ):
        self._state = state
        self._analyzer = analyzer
        self._time = datetime.now()
        self._fee_loss = DECIMAL_ONE - fee
        self._stock = stock

        self._state.on(
            TradingStateEvent.PERFORMANCE_SNAPSHOT_RECORDED,
            self._analyzer.add_snapshots
        )

        self._state.on(
            TradingStateEvent.POSITION_TARGET_UPDATED,
            self._handle_orders
        )

        self._state.set_notional_limit(BTC, None)

        self._state.set_balances([
            Balance(
                USDT,
                init_base_currency,
                DECIMAL_ZERO
            ),
            Balance(
                BTC,
                DECIMAL_ZERO,
                DECIMAL_ZERO
            )
        ])

        self._state.set_price(
            BTCUSDT.name,
            Decimal(str(self._stock['close'].iloc[0]))
        )

    def _handle_orders(self) -> None:
        orders, _ = self._state.get_orders()

        # In replayer mode, no orders need to be canceled

        if len(orders) == 0:
            return

        # In replayer, for now, we only support one symbol at a time
        order = orders.pop()

        time = self._time
        ticket = order.ticket
        symbol = ticket.symbol

        if ticket.side is OrderSide.BUY:
            asset_gain, asset_used = symbol.base_asset, symbol.quote_asset
            quantity_gain = ticket.quantity * self._fee_loss
            quantity_used = ticket.quantity * ticket.price

            self._state.update_order(
                order,
                status=OrderStatus.FILLED,
                created_at=time,
                filled_quantity=quantity_gain,
                quote_quantity=quantity_used,
            )
        else:
            asset_gain, asset_used = symbol.quote_asset, symbol.base_asset
            quantity_gain = ticket.quantity * ticket.price * self._fee_loss
            quantity_used = ticket.quantity

            self._state.update_order(
                order,
                status=OrderStatus.FILLED,
                created_at=time,
                filled_quantity=quantity_used,
                quote_quantity=quantity_gain,
            )

        self._state.set_balances([
            Balance(
                asset_gain,
                quantity_gain,
                DECIMAL_ZERO,
                time
            ),
            Balance(
                asset_used,
                - quantity_used,
                DECIMAL_ZERO,
                time
            )
        ], delta=True)

        # print(
        #     f'{order.ticket.side} {order.ticket.price}',
        #     'btc', self._state._balances.get_balance(BTC).free,
        #     'usdt', self._state._balances.get_balance(USDT).free,
        # )

        self._state.record(time=time)

    def go(self) -> None:
        gold_cross = StockDataFrame.directive_stringify('macd // macd.signal')
        dead_cross = StockDataFrame.directive_stringify('macd \\ macd.signal')

        self._stock[gold_cross]
        self._stock[dead_cross]

        for _, row in self._stock.iterrows():
            price = Decimal(str(row['close']))
            self._state.set_price(BTCUSDT.name, price)

            buy = row[gold_cross]
            sell = row[dead_cross]

            self._time = row.name

            if not buy and not sell:
                self._state.record(time=row.name)
                continue

            exception, _ = self._state.expect(
                BTCUSDT.name,
                exposure=DECIMAL_ONE if buy else DECIMAL_ZERO,
                price=price,
                use_market_order=False
            )

            if exception is not None:
                raise exception


def test_analyzer():
    state = init_state(
        config=TradingConfig(**{
            **DEFAULT_CONFIG_KWARGS,
            'benchmark_assets': (BTC,)
        }),
        with_balances=False
    )

    stock = get_stock()
    init_base_currency = Decimal('10000')
    analyzer = PerformanceAnalyzer(AnalyzerType.all())
    trader = Trader(
        state,
        analyzer,
        stock,
        init_base_currency,
        Decimal('0.0075')
    )

    trader.go()

    last_snapshot = analyzer._snapshots[-1]

    # print('last snapshot:', last_snapshot)

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

    # print(analyzer.analyze()[AnalyzerType.TOTAL_RETURN])
