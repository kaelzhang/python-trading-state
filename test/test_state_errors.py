from pytest import fixture
from decimal import Decimal

from trading_state import (
    TradingState,
    TradingConfig,
    Balance,
    SymbolPriceNotReadyError,
    AssetNotDefinedError,
    QuotaNotSetError,
    BalanceNotReadyError,
    ExpectWithoutPriceError,
    NumerairePriceNotReadyError,
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
            numeraire=USDT
        )
    )

    # with pytest.raises(SymbolNotDefinedError):
    exception, _ = state.expect(
        BTCUSDC,
        position=1,
        price=None,
        asap=True
    )

    assert isinstance(exception, SymbolNotDefinedError)

    exception, _ = state.position(BTC)

    assert isinstance(exception, AssetNotDefinedError)

    state.set_symbol(test_symbols[BTCUSDC])

    exception, _ = state.expect(
        BTCUSDC,
        position=1,
        price=None,
        asap=True
    )

    assert isinstance(exception, SymbolPriceNotReadyError)

    state.set_price(BTCUSDC, Decimal('10000'))

    exception, _ = state.expect(
        BTCUSDC,
        position=1,
        price=None,
        asap=True
    )

    assert isinstance(exception, QuotaNotSetError)

    state.set_quota(BTC, Decimal('10000'))

    exception, _ = state.expect(
        BTCUSDC,
        position=1,
        price=None,
        asap=True
    )

    assert isinstance(exception, NumerairePriceNotReadyError)

    state.set_price(BTCUSDT, Decimal('10000'))

    exception, _ = state.expect(
        BTCUSDC,
        position=1,
        price=None,
        asap=True
    )

    assert isinstance(exception, BalanceNotReadyError)

    state.set_balances([
        Balance(BTC, Decimal('1'), Decimal('0'))
    ])

    exception, _ = state.expect(
        BTCUSDC,
        position=1,
        price=None,
        asap=False
    )

    assert isinstance(exception, ExpectWithoutPriceError)
