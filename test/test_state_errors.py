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


def test_create_order_allocate_weights_validation():
    """`allocate=` is validated per-call by create_order. Length must
    match config.alt_account_currencies; every entry must be
    non-negative. Either violation surfaces an
    InvalidAllocationWeightsError via ValueOrException."""
    from trading_state import (
        InvalidAllocationWeightsError,
        LimitOrderTicket,
        OrderSide,
        Symbol,
        TimeInForce,
    )

    state = TradingState(
        TradingConfig(
            account_currency=USDT,
            alt_account_currencies=(USDC,),
        ),
    )
    state.set_symbol(Symbol(BTCUSDT_NAME, BTC, USDT))
    state.set_price(BTCUSDT_NAME, Decimal('10000'))
    state.set_notional_limit(BTC, Decimal('100000'))
    state.set_balances([
        Balance(USDT, Decimal('100000'), Decimal('0'), balance_time()),
    ])
    sym = state.get_symbol(BTCUSDT_NAME)
    ticket = LimitOrderTicket(
        symbol=sym,
        side=OrderSide.BUY,
        quantity=Decimal('1'),
        price=Decimal('10000'),
        time_in_force=TimeInForce.GTC,
    )

    # Wrong length (2 entries, config has 1 alt).
    exc, out = state.create_order(
        ticket,
        allocate=(Decimal('0.5'), Decimal('0.2')),
    )
    assert out is None
    assert isinstance(exc, InvalidAllocationWeightsError)

    # Negative weight.
    exc, out = state.create_order(
        ticket,
        allocate=(Decimal('-1'),),
    )
    assert out is None
    assert isinstance(exc, InvalidAllocationWeightsError)


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
    # allow QUOTE_ORDER_QUANTITY. allocate runs the symbol's filters in
    # the passthrough flow; a filter rejection is treated as best-effort
    # and surfaces as an empty list rather than an exception.
    ticket = MarketOrderTicket(
        symbol=test_symbols[SYMBOL],
        side=OrderSide.BUY,
        quantity=Decimal('20000'),
        quantity_type=MarketQuantityType.QUOTE,
        estimated_price=Decimal('10000'),
    )

    exception, orders = state.create_order(ticket, allocate=None)

    assert exception is None
    assert orders == []
    # The order was never registered with state.
    assert len(list(state.query_orders())) == 0

    # The FeatureNotAllowedError is still observable to callers that
    # want a diagnostic before / instead of allocate; the symbol's
    # filter chain surfaces it directly.
    filter_exc, _ = test_symbols[SYMBOL].apply_filters(
        ticket, validate_only=True
    )
    assert isinstance(filter_exc, FeatureNotAllowedError)
    assert filter_exc.feature == FeatureType.QUOTE_ORDER_QUANTITY
    assert filter_exc.symbol.name == SYMBOL

    # symbol.support diagnostics
    symbol = filter_exc.symbol
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

    # set_symbol invalidates the valuation-path cache, so the
    # previously-None entry for X is reconsidered on the next call.
    path = state._symbols.valuation_path(X)
    assert path == [
        ValuationPathStep(XY, True),
        ValuationPathStep(ZY, False),
        ValuationPathStep(ZUSDT, True),
    ]
