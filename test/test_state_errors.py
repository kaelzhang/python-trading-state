from pytest import fixture
from decimal import Decimal
import pytest

from trading_state import (
    TradingState,
    TradingConfig,
    Balance,
    AssetNotDefinedError,
    NotionalLimitNotSetError,
    BalanceNotReadyError,
    ValuationPriceNotReadyError,
    SymbolPriceNotReadyError,
    SymbolNotDefinedError
)

from .fixtures import (
    get_symbols,
    Symbols,
    BTCUSDC,
    BTCUSDT,
    BTC,
    USDT,
    USDC
)


@fixture
def test_symbols() -> Symbols:
    return get_symbols()


def test_trading_state_errors(test_symbols: Symbols):
    state = TradingState(
        config=TradingConfig(
            account_currency=USDT,
            alt_account_currencies=[USDC]
        )
    )

    with pytest.raises(ValueError, match='must be equal to'):
        state.set_alt_currency_weights((
            (Decimal('0.5'), Decimal('0.2')),
            (Decimal('0.5'), Decimal('0'))
        ))

    with pytest.raises(ValueError, match='less than 0'):
        state.set_alt_currency_weights(
            (
                (Decimal('-1'),),
                (Decimal('0'),)
            )
        )

    price = Decimal('10000')

    # with pytest.raises(SymbolNotDefinedError):
    exception, _ = state.expect(
        BTCUSDC,
        exposure=1,
        price=price,
        use_market_order=True
    )

    assert isinstance(exception, SymbolNotDefinedError)

    exception, _ = state.exposure(BTC)

    assert isinstance(exception, AssetNotDefinedError)

    state.set_symbol(test_symbols[BTCUSDC])

    exception, _ = state.expect(
        BTCUSDC,
        exposure=1,
        price=price,
        use_market_order=True
    )

    assert isinstance(exception, SymbolPriceNotReadyError)

    state.set_price(BTCUSDC, Decimal('10000'))

    exception, _ = state.expect(
        BTCUSDC,
        exposure=1,
        price=price,
        use_market_order=True
    )

    assert isinstance(exception, NotionalLimitNotSetError)

    state.set_notional_limit(BTC, Decimal('10000'))

    exception, _ = state.expect(
        BTCUSDC,
        exposure=1,
        price=price,
        use_market_order=True
    )

    assert isinstance(exception, ValuationPriceNotReadyError)

    state.set_price(BTCUSDT, Decimal('10000'))

    exception, _ = state.expect(
        BTCUSDC,
        exposure=1,
        price=price,
        use_market_order=True
    )

    assert isinstance(exception, BalanceNotReadyError)

    state.set_balances([
        Balance(BTC, Decimal('1'), Decimal('0'))
    ])

    exception, _ = state.expect(
        BTCUSDC,
        exposure=1,
        price=price,
        use_market_order=False
    )

    assert isinstance(exception, BalanceNotReadyError)

    state.set_balances([
        Balance(USDC, Decimal('100000'), Decimal('0'))
    ])
