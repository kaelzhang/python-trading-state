from pytest import fixture
from decimal import Decimal

from trading_state import (
    TradingState,
    TradingConfig,
    Balance,
    SymbolPriceNotReadyError,
    AssetNotDefinedError,
    NotionalLimitNotSetError,
    BalanceNotReadyError,
    ExpectWithoutPriceError,
    ValuationPriceNotReadyError,
    SymbolNotDefinedError
)

from .fixtures import (
    get_symbols,
    Symbols,
    BTCUSDC,
    BTCUSDT,
    BTC,
    USDT
)


@fixture
def test_symbols() -> Symbols:
    return get_symbols()


def test_trading_state_errors(test_symbols: Symbols):
    state = TradingState(
        config=TradingConfig(
            account_currency=USDT
        )
    )

    # with pytest.raises(SymbolNotDefinedError):
    exception, _ = state.expect(
        BTCUSDC,
        utilization=1,
        price=None,
        immediate=True
    )

    assert isinstance(exception, SymbolNotDefinedError)

    exception, _ = state.utilization(BTC)

    assert isinstance(exception, AssetNotDefinedError)

    state.set_symbol(test_symbols[BTCUSDC])

    exception, _ = state.expect(
        BTCUSDC,
        utilization=1,
        price=None,
        immediate=True
    )

    assert isinstance(exception, SymbolPriceNotReadyError)

    state.set_price(BTCUSDC, Decimal('10000'))

    exception, _ = state.expect(
        BTCUSDC,
        utilization=1,
        price=None,
        immediate=True
    )

    assert isinstance(exception, NotionalLimitNotSetError)

    state.set_notional_limit(BTC, Decimal('10000'))

    exception, _ = state.expect(
        BTCUSDC,
        utilization=1,
        price=None,
        immediate=True
    )

    assert isinstance(exception, ValuationPriceNotReadyError)

    state.set_price(BTCUSDT, Decimal('10000'))

    exception, _ = state.expect(
        BTCUSDC,
        utilization=1,
        price=None,
        immediate=True
    )

    assert isinstance(exception, BalanceNotReadyError)

    state.set_balances([
        Balance(BTC, Decimal('1'), Decimal('0'))
    ])

    exception, _ = state.expect(
        BTCUSDC,
        utilization=1,
        price=None,
        immediate=False
    )

    assert isinstance(exception, ExpectWithoutPriceError)
