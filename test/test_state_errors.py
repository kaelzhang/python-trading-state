from pytest import fixture
from decimal import Decimal
import pytest

from trading_state import (
    TradingState,
    TradingConfig,
    Balance,
    OrderSide,
    MarketOrderTicket,
    MarketQuantityType,
    AssetNotDefinedError,
    NotionalLimitNotSetError,
    BalanceNotReadyError,
    ValuationPriceNotReadyError,
    FeatureNotAllowedError,
    FeatureType,
    ValuationNotAvailableError,
)
from trading_state.symbol import ValuationPathStep

from .fixtures import (
    init_state,
    balance_time,
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
    ZUSDT,
)


BTCUSDC_NAME = BTCUSDC.name
BTCUSDT_NAME = BTCUSDT.name


@fixture
def test_symbols() -> Symbols:
    return get_symbols()


def test_weights_must_match_alt_count():
    state = TradingState(
        TradingConfig(
            account_currency=USDT,
            alt_account_currencies=(USDC,),
        ),
    )

    with pytest.raises(ValueError, match='must be equal to'):
        state.set_alt_currency_weights((
            (Decimal('0.5'), Decimal('0.2')),
            (Decimal('0.5'), Decimal('0')),
        ))

    with pytest.raises(ValueError, match='less than 0'):
        state.set_alt_currency_weights((
            (Decimal('-1'),),
            (Decimal('0'),),
        ))


def test_exposure_errors_propagate_through_value_or_exception(
    test_symbols: Symbols,
):
    """
    Exposure exercises BalanceManager.check_asset_ready, which produces
    each of the configured error types depending on which precondition
    is missing.
    """
    state = TradingState(
        TradingConfig(
            account_currency=USDT,
            alt_account_currencies=(USDC,),
        ),
    )

    exception, _ = state.exposure(
        BTC,
        include_unsettled_inflow=False,
        include_unsettled_outflow=False,
    )
    assert isinstance(exception, AssetNotDefinedError)

    state.set_symbol(test_symbols[BTCUSDC_NAME])
    state.set_symbol(test_symbols[BTCUSDT_NAME])

    exception, _ = state.exposure(
        BTC,
        include_unsettled_inflow=False,
        include_unsettled_outflow=False,
    )
    assert isinstance(exception, NotionalLimitNotSetError)

    state.set_notional_limit(BTC, Decimal('10000'))

    exception, _ = state.exposure(
        BTC,
        include_unsettled_inflow=False,
        include_unsettled_outflow=False,
    )
    assert isinstance(exception, ValuationPriceNotReadyError)

    state.set_price(BTCUSDT_NAME, Decimal('10000'))

    exception, _ = state.exposure(
        BTC,
        include_unsettled_inflow=False,
        include_unsettled_outflow=False,
    )
    assert isinstance(exception, BalanceNotReadyError)

    state.set_balances([
        Balance(BTC, Decimal('1'), Decimal('0'), balance_time()),
    ])

    exception, exposure = state.exposure(
        BTC,
        include_unsettled_inflow=False,
        include_unsettled_outflow=False,
    )
    assert exception is None
    # 1 BTC * 10000 valuation / 10000 limit
    assert exposure.ratio == Decimal('1')


def test_feature_not_allowed_error(test_symbols: Symbols):
    state = init_state()

    # The one that does not support quote order quantity
    SYMBOL = 'BTCUPUSDT'
    ASSET = 'BTCUP'

    state.set_notional_limit(ASSET, Decimal('100000'))
    state.set_price(SYMBOL, Decimal('10000'))
    state.set_balances([
        Balance(ASSET, Decimal('1'), Decimal('0'), balance_time()),
    ])

    # MARKET ticket with QUOTE quantity against a symbol that doesn't
    # allow QUOTE_ORDER_QUANTITY — add_order should surface the
    # FeatureNotAllowedError via ValueOrException.
    ticket = MarketOrderTicket(
        symbol=test_symbols[SYMBOL],
        side=OrderSide.BUY,
        quantity=Decimal('20000'),
        quantity_type=MarketQuantityType.QUOTE,
        estimated_price=Decimal('10000'),
    )

    exception, order = state.add_order(ticket)

    assert order is None
    assert isinstance(exception, FeatureNotAllowedError)
    assert exception.feature == FeatureType.QUOTE_ORDER_QUANTITY
    assert exception.symbol.name == SYMBOL

    # symbol.support diagnostics
    symbol = exception.symbol
    with pytest.raises(ValueError, match='but got None'):
        symbol.support(FeatureType.ORDER_TYPE)
    with pytest.raises(ValueError, match='but got 1'):
        symbol.support(FeatureType.QUOTE_ORDER_QUANTITY, 1)
    assert not symbol.support(FeatureType.QUOTE_ORDER_QUANTITY)


def test_exception_constructions_attach_target_info():
    """SymbolNotDefinedError / SymbolPriceNotReadyError are exported for
    downstream protocol adapters that may need to surface symbol-side
    issues. Verify their public construction contract."""
    from trading_state import (
        SymbolNotDefinedError,
        SymbolPriceNotReadyError,
    )

    exc_a = SymbolNotDefinedError('BTCUSDT')
    assert 'BTCUSDT' in str(exc_a)
    assert exc_a.symbol_name == 'BTCUSDT'

    exc_b = SymbolPriceNotReadyError('ETHUSDT')
    assert 'ETHUSDT' in str(exc_b)
    assert exc_b.symbol_name == 'ETHUSDT'


def test_valuation_path_not_available(test_symbols: Symbols):
    state = init_state()

    state.set_symbol(XY)
    state.set_symbol(ZY)

    state.set_notional_limit(X, Decimal('1'))

    exception = state._balances.check_asset_ready(X)
    assert isinstance(exception, ValuationNotAvailableError)
    assert exception.asset == X

    state.set_symbol(ZUSDT)

    # Clean the cached valuation path (testing-only)
    del state._symbols._valuation_paths[X]

    path = state._symbols.valuation_path(X)
    assert path == [
        ValuationPathStep(XY, True),
        ValuationPathStep(ZY, False),
        ValuationPathStep(ZUSDT, True),
    ]
