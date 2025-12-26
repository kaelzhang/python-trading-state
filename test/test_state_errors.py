from pytest import fixture
from decimal import Decimal
import pytest

from trading_state import (
    TradingState,
    TradingConfig,
    TradingStateEvent,
    Balance,
    AssetNotDefinedError,
    NotionalLimitNotSetError,
    BalanceNotReadyError,
    ValuationPriceNotReadyError,
    SymbolPriceNotReadyError,
    SymbolNotDefinedError,
    FeatureNotAllowedError,
    FeatureType,
    ValuationNotAvailableError
)
from trading_state.symbol import ValuationPathStep

from .fixtures import (
    init_state,
    get_symbols,
    Symbols,
    BTCUSDC,
    BTCUSDT,
    BTC,
    USDT,
    USDC,
    X,
    XY,
    ZY,
    ZUSDT
)


BTCUSDC = BTCUSDC.name
BTCUSDT = BTCUSDT.name


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
    state.set_symbol(test_symbols[BTCUSDT])

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


def test_feature_not_allowed_error(test_symbols: Symbols):
    state = init_state()

    # The one that does not support quote order quantity
    SYMBOL = 'BTCUPUSDT'
    ASSET = 'BTCUP'

    state.set_notional_limit(ASSET, Decimal('100000'))
    state.set_price(SYMBOL, Decimal('10000'))
    state.set_balances([
        Balance(ASSET, Decimal('1'), Decimal('0'))
    ])

    exception, _ = state.expect(
        SYMBOL,
        exposure=0.2,
        price=Decimal('10000'),
        use_market_order=True
    )

    assert exception is None

    error = False

    def handler(exception: Exception):
        nonlocal error
        error = True

        symbol = exception.symbol

        assert isinstance(exception, FeatureNotAllowedError)
        assert SYMBOL in str(exception)
        assert symbol.name == SYMBOL
        assert exception.feature == FeatureType.QUOTE_ORDER_QUANTITY

        with pytest.raises(
            ValueError,
            match='but got None'
        ):
            symbol.support(FeatureType.ORDER_TYPE)

        with pytest.raises(
            ValueError,
            match='but got 1'
        ):
            symbol.support(FeatureType.QUOTE_ORDER_QUANTITY, 1)

        assert not symbol.support(FeatureType.QUOTE_ORDER_QUANTITY)

    state.on(TradingStateEvent.TICKET_CREATE_FAILED, handler)

    orders, _ = state.get_orders()
    assert not orders

    assert error


def test_valuation_path_not_available(test_symbols: Symbols):
    state = init_state()

    state.set_symbol(XY)
    state.set_symbol(ZY)

    state.set_notional_limit(X, None)

    exception = state._balances.check_asset_ready(X)
    assert isinstance(exception, ValuationNotAvailableError)
    assert exception.asset == X

    state.set_symbol(ZUSDT)

    # Clean the cached valuation path
    # Only for testing
    del state._symbols._valuation_paths[X]

    path = state._symbols.valuation_path(X)
    assert path == [
        ValuationPathStep(XY, True),
        ValuationPathStep(ZY, False),
        ValuationPathStep(ZUSDT, True),
    ]
