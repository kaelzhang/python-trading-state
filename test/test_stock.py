from trading_state import (
    TradingState,
    TradingConfig,
    Symbol
)


def test_stock():
    state = TradingState(
        config=TradingConfig(
            account_currency='',
        )
    )

    state.set_symbol(
        Symbol(
            name='AAPL',
            base_asset='AAPL',
            quote_asset=''
        )
    )


