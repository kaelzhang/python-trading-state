from decimal import Decimal

from trading_state import (
    Symbol,
    OrderSide,
    TimeInForce,
    MarketQuantityType,
    FeatureType
)
from trading_state.exceptions import FeatureNotAllowedError
from trading_state.filters import (
    apply_range,
    PrecisionFilter,
    PriceFilter,
    QuantityFilter,
    MarketQuantityFilter,
    IcebergQuantityFilter,
    TrailingDeltaFilter,
    NotionalFilter
)
from trading_state.order_ticket import (
    LimitOrderTicket,
    MarketOrderTicket,
    StopLossOrderTicket
)


def test_apply_range_branches() -> None:
    error, value = apply_range(
        target=Decimal('0.5'),
        min_value=Decimal('1'),
        max_value=Decimal('0'),
        tick_size=Decimal('0'),
        validate_only=False,
        param_name='price',
        name='price'
    )
    assert isinstance(error, ValueError)
    assert value is None

    error, value = apply_range(
        target=Decimal('10'),
        min_value=Decimal('0'),
        max_value=Decimal('5'),
        tick_size=Decimal('0'),
        validate_only=False,
        param_name='price',
        name='price'
    )
    assert isinstance(error, ValueError)
    assert value is None

    error, value = apply_range(
        target=Decimal('5'),
        min_value=Decimal('1'),
        max_value=Decimal('10'),
        tick_size=Decimal('0'),
        validate_only=False,
        param_name='price',
        name='price'
    )
    assert error is None
    assert value == Decimal('5')

    error, value = apply_range(
        target=Decimal('1.23'),
        min_value=Decimal('1'),
        max_value=Decimal('10'),
        tick_size=Decimal('0.1'),
        validate_only=True,
        param_name='price',
        name='price'
    )
    assert isinstance(error, ValueError)
    assert value is None

    error, value = apply_range(
        target=Decimal('1.23'),
        min_value=Decimal('1'),
        max_value=Decimal('10'),
        tick_size=Decimal('0.1'),
        validate_only=False,
        param_name='price',
        name='price'
    )
    assert error is None
    assert value == Decimal('1.2')


def test_precision_filter() -> None:
    symbol = Symbol('FOOUSD', 'FOO', 'USD')
    precision_filter = PrecisionFilter(
        base_asset_precision=2,
        quote_asset_precision=3
    )

    limit_sell = LimitOrderTicket(
        symbol=symbol,
        side=OrderSide.SELL,
        quantity=Decimal('1.2345'),
        price=Decimal('10'),
        time_in_force=TimeInForce.GTC
    )
    assert precision_filter.when(limit_sell)

    error, new_ticket = precision_filter.apply(
        limit_sell, validate_only=True
    )
    assert isinstance(error, ValueError)
    assert new_ticket is None
    # Original is frozen and unchanged
    assert limit_sell.quantity == Decimal('1.2345')

    error, new_ticket = precision_filter.apply(
        limit_sell, validate_only=False
    )
    assert error is None
    assert new_ticket is not None
    assert new_ticket.quantity == Decimal('1.23')
    # Original ticket remains untouched
    assert limit_sell.quantity == Decimal('1.2345')

    market_quote = MarketOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('1.2345'),
        quantity_type=MarketQuantityType.QUOTE,
        estimated_price=Decimal('10')
    )
    error, new_ticket = precision_filter.apply(
        market_quote, validate_only=False
    )
    assert error is None
    assert new_ticket is not None
    assert new_ticket.quantity == Decimal('1.23')

    market_base = MarketOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('1.2345'),
        quantity_type=MarketQuantityType.BASE,
        estimated_price=Decimal('10')
    )
    error, new_ticket = precision_filter.apply(
        market_base, validate_only=False
    )
    assert error is None
    assert new_ticket is not None
    assert new_ticket.quantity == Decimal('1.234')


def test_price_filter() -> None:
    symbol = Symbol('FOOUSD', 'FOO', 'USD')
    price_filter = PriceFilter(
        min_price=Decimal('1'),
        max_price=Decimal('10'),
        tick_size=Decimal('0.1')
    )

    limit_ticket = LimitOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('1'),
        price=Decimal('1.23'),
        time_in_force=TimeInForce.GTC
    )
    assert price_filter.when(limit_ticket)

    error, new_ticket = price_filter.apply(
        limit_ticket, validate_only=True
    )
    assert isinstance(error, ValueError)
    assert new_ticket is None

    error, new_ticket = price_filter.apply(
        limit_ticket, validate_only=False
    )
    assert error is None
    assert new_ticket is not None
    assert new_ticket.price == Decimal('1.2')

    limit_high = LimitOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('1'),
        price=Decimal('11'),
        time_in_force=TimeInForce.GTC
    )
    error, new_ticket = price_filter.apply(
        limit_high, validate_only=False
    )
    assert isinstance(error, ValueError)
    assert new_ticket is None

    stop_ticket = StopLossOrderTicket(
        symbol=symbol,
        side=OrderSide.SELL,
        quantity=Decimal('1'),
        stop_price=Decimal('1.23')
    )
    assert price_filter.when(stop_ticket)

    error, new_ticket = price_filter.apply(
        stop_ticket, validate_only=True
    )
    assert isinstance(error, ValueError)
    assert new_ticket is None

    error, new_ticket = price_filter.apply(
        stop_ticket, validate_only=False
    )
    assert error is None
    assert new_ticket is not None
    assert new_ticket.stop_price == Decimal('1.2')


def test_quantity_filter_branches() -> None:
    symbol = Symbol('FOOUSD', 'FOO', 'USD')
    quantity_filter = QuantityFilter(
        min_quantity=Decimal('1'),
        max_quantity=Decimal('5'),
        step_size=Decimal('0.5')
    )

    market_ticket = MarketOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('1'),
        quantity_type=MarketQuantityType.BASE,
        estimated_price=Decimal('10')
    )
    assert not quantity_filter.when(market_ticket)

    limit_low = LimitOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('0.4'),
        price=Decimal('1'),
        time_in_force=TimeInForce.GTC
    )
    error, new_ticket = quantity_filter.apply(
        limit_low, validate_only=False
    )
    assert isinstance(error, ValueError)
    assert new_ticket is None

    limit_step = LimitOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('1.2'),
        price=Decimal('1'),
        time_in_force=TimeInForce.GTC
    )
    assert quantity_filter.when(limit_step)

    error, new_ticket = quantity_filter.apply(
        limit_step, validate_only=True
    )
    assert isinstance(error, ValueError)
    assert new_ticket is None

    error, new_ticket = quantity_filter.apply(
        limit_step, validate_only=False
    )
    assert error is None
    assert new_ticket is not None
    assert new_ticket.quantity == Decimal('1.0')


def test_market_quantity_filter_branches() -> None:
    symbol = Symbol('FOOUSD', 'FOO', 'USD')
    market_filter = MarketQuantityFilter(
        min_quantity=Decimal('0.01'),
        max_quantity=Decimal('1'),
        step_size=Decimal('0.01')
    )

    ticket_quote = MarketOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('0.5'),
        quantity_type=MarketQuantityType.QUOTE,
        estimated_price=Decimal('10')
    )
    assert market_filter.when(ticket_quote)

    error, new_ticket = market_filter.apply(
        ticket_quote, validate_only=False
    )
    assert isinstance(error, FeatureNotAllowedError)
    assert new_ticket is None

    symbol.allow(FeatureType.QUOTE_ORDER_QUANTITY, True)
    ticket_quote_ok = MarketOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('0.333'),
        quantity_type=MarketQuantityType.QUOTE,
        estimated_price=Decimal('10')
    )
    error, new_ticket = market_filter.apply(
        ticket_quote_ok, validate_only=False
    )
    assert error is None
    assert new_ticket is not None
    assert new_ticket.quantity == Decimal('0.3')

    ticket_base = MarketOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('0.5'),
        quantity_type=MarketQuantityType.BASE,
        estimated_price=Decimal('10')
    )
    error, new_ticket = market_filter.apply(
        ticket_base, validate_only=False
    )
    assert error is None
    assert new_ticket is None

    ticket_base_low = MarketOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('0.001'),
        quantity_type=MarketQuantityType.BASE,
        estimated_price=Decimal('10')
    )
    error, new_ticket = market_filter.apply(
        ticket_base_low, validate_only=False
    )
    assert isinstance(error, ValueError)
    assert new_ticket is None

    ticket_base_step = MarketOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('0.333'),
        quantity_type=MarketQuantityType.BASE,
        estimated_price=Decimal('10')
    )
    error, new_ticket = market_filter.apply(
        ticket_base_step, validate_only=False
    )
    assert error is None
    assert new_ticket is not None
    assert new_ticket.quantity == Decimal('0.33')


def test_iceberg_quantity_filter() -> None:
    symbol = Symbol('FOOUSD', 'FOO', 'USD')
    iceberg_filter = IcebergQuantityFilter(limit=3)

    zero_ticket = LimitOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('1'),
        price=Decimal('1'),
        time_in_force=TimeInForce.GTC,
        iceberg_quantity=Decimal('0')
    )
    error, new_ticket = iceberg_filter.apply(
        zero_ticket, validate_only=False
    )
    assert isinstance(error, ValueError)
    assert new_ticket is None

    ticket = LimitOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('9'),
        price=Decimal('1'),
        time_in_force=TimeInForce.GTC,
        iceberg_quantity=Decimal('1')
    )
    assert iceberg_filter.when(ticket)

    error, new_ticket = iceberg_filter.apply(
        ticket, validate_only=True
    )
    assert isinstance(error, ValueError)
    assert new_ticket is None

    error, new_ticket = iceberg_filter.apply(
        ticket, validate_only=False
    )
    assert error is None
    assert new_ticket is not None
    assert new_ticket.iceberg_quantity == Decimal('3')

    ticket_ok = LimitOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('1'),
        price=Decimal('1'),
        time_in_force=TimeInForce.GTC,
        iceberg_quantity=Decimal('2')
    )
    error, new_ticket = iceberg_filter.apply(
        ticket_ok, validate_only=False
    )
    assert error is None
    assert new_ticket is None


def test_trailing_delta_filter() -> None:
    symbol = Symbol('FOOUSD', 'FOO', 'USD')
    trailing_filter = TrailingDeltaFilter(
        min_trailing_above_delta=10,
        max_trailing_above_delta=20,
        min_trailing_below_delta=30,
        max_trailing_below_delta=40
    )

    no_trailing = StopLossOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('1'),
        stop_price=Decimal('1')
    )
    assert not trailing_filter.when(no_trailing)

    above_low = StopLossOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('1'),
        trailing_delta=5
    )
    error, new_ticket = trailing_filter.apply(
        above_low, validate_only=True
    )
    assert isinstance(error, ValueError)
    assert new_ticket is None

    above_high = StopLossOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('1'),
        trailing_delta=25
    )
    error, new_ticket = trailing_filter.apply(
        above_high, validate_only=False
    )
    assert error is None
    assert new_ticket is not None
    assert new_ticket.trailing_delta == 20

    below_low = StopLossOrderTicket(
        symbol=symbol,
        side=OrderSide.SELL,
        quantity=Decimal('1'),
        trailing_delta=25
    )
    error, new_ticket = trailing_filter.apply(
        below_low, validate_only=False
    )
    assert error is None
    assert new_ticket is not None
    assert new_ticket.trailing_delta == 30

    below_high = StopLossOrderTicket(
        symbol=symbol,
        side=OrderSide.SELL,
        quantity=Decimal('1'),
        trailing_delta=50
    )
    error, new_ticket = trailing_filter.apply(
        below_high, validate_only=True
    )
    assert isinstance(error, ValueError)
    assert new_ticket is None

    # in-range value: filter passes, ticket unchanged
    above_in_range = StopLossOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('1'),
        trailing_delta=15
    )
    error, new_ticket = trailing_filter.apply(
        above_in_range, validate_only=False
    )
    assert error is None
    assert new_ticket is None


def test_notional_filter() -> None:
    symbol = Symbol('FOOUSD', 'FOO', 'USD')

    # apply_min_to_market=False and apply_max_to_market=False:
    # MARKET orders bypass the check entirely
    no_apply = NotionalFilter(
        min_notional=Decimal('10'),
        max_notional=Decimal('100'),
        apply_min_to_market=False,
        apply_max_to_market=False,
        avg_price_mins=5
    )
    market_ticket = MarketOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('1'),
        quantity_type=MarketQuantityType.BASE,
        estimated_price=Decimal('10')
    )
    assert no_apply.when(market_ticket)
    error, new_ticket = no_apply.apply(
        market_ticket, validate_only=False
    )
    assert error is None
    assert new_ticket is None

    # apply_min_to_market=True only: MARKET notional must be >= min
    # market_ticket: estimated_price=10, quantity=1 (BASE) → notional 10, >= 10
    min_only = NotionalFilter(
        min_notional=Decimal('10'),
        max_notional=Decimal('100'),
        apply_min_to_market=True,
        apply_max_to_market=False,
        avg_price_mins=5
    )
    error, new_ticket = min_only.apply(
        market_ticket, validate_only=False
    )
    assert error is None
    assert new_ticket is None

    # QUOTE: notional = quantity directly. 5 < min 10 → fail
    market_quote_ticket = MarketOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('5'),
        quantity_type=MarketQuantityType.QUOTE,
        estimated_price=Decimal('10')
    )
    error, new_ticket = min_only.apply(
        market_quote_ticket, validate_only=False
    )
    assert isinstance(error, ValueError)
    assert new_ticket is None

    # apply_max_to_market=True only; estimated_price * quantity is the
    # local notional approximation for BASE MARKET orders.
    max_only = NotionalFilter(
        min_notional=Decimal('10'),
        max_notional=Decimal('100'),
        apply_min_to_market=False,
        apply_max_to_market=True,
        avg_price_mins=5
    )
    # quantity=11, estimated_price=10 → notional 110, > max 100 → fail
    market_over_max = MarketOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('11'),
        quantity_type=MarketQuantityType.BASE,
        estimated_price=Decimal('10')
    )
    error, new_ticket = max_only.apply(
        market_over_max, validate_only=False
    )
    assert isinstance(error, ValueError)
    assert new_ticket is None

    # QUOTE: notional is quantity directly; 200 > max 100 → fail
    error, new_ticket = max_only.apply(
        MarketOrderTicket(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=Decimal('200'),
            quantity_type=MarketQuantityType.QUOTE,
            estimated_price=Decimal('10')
        ),
        validate_only=False
    )
    assert isinstance(error, ValueError)
    assert new_ticket is None

    # LIMIT orders use ticket.price for notional regardless of the
    # apply_*_to_market flags
    limit_low = LimitOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('1'),
        price=Decimal('5'),
        time_in_force=TimeInForce.GTC
    )
    error, new_ticket = min_only.apply(
        limit_low, validate_only=False
    )
    assert isinstance(error, ValueError)
    assert new_ticket is None

    limit_high = LimitOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('3'),
        price=Decimal('50'),
        time_in_force=TimeInForce.GTC
    )
    error, new_ticket = min_only.apply(
        limit_high, validate_only=False
    )
    assert isinstance(error, ValueError)
    assert new_ticket is None

    limit_ok = LimitOrderTicket(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal('2'),
        price=Decimal('20'),
        time_in_force=TimeInForce.GTC
    )
    error, new_ticket = min_only.apply(
        limit_ok, validate_only=False
    )
    assert error is None
    assert new_ticket is None
