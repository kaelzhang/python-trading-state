import pytest

from trading_state.order_ticket import (
    LimitOrderTicket,
    MarketOrderTicket,
    StopLossOrderTicket,
    StopLossLimitOrderTicket,
    TakeProfitOrderTicket,
    TakeProfitLimitOrderTicket,
)

from trading_state.symbol import Symbol
from trading_state.enums import (
    OrderSide, TimeInForce, MarketQuantityType
)

from decimal import Decimal


symbol = Symbol(
    name='AB',
    base_asset='A',
    quote_asset='B',
)


stop_loss_case = ({
    'symbol': symbol,
    'side': OrderSide.BUY,
    'quantity': Decimal('1'),
    'stop_price': Decimal('10000'),
    'trailing_delta': Decimal('0.01'),
}, ['stop_price', 'trailing_delta'])

stop_loss_limit_case = ({
    'symbol': symbol,
    'side': OrderSide.BUY,
    'quantity': Decimal('1'),
    'price': Decimal('10000'),
    'time_in_force': TimeInForce.GTC,
    'stop_price': Decimal('10000'),
    'trailing_delta': Decimal('0.01'),
}, ['stop_price', 'trailing_delta'])


RIGHT_CASES = [
    ('limit', LimitOrderTicket, {
        'symbol': symbol,
        'side': OrderSide.BUY,
        'quantity': Decimal('1'),
        'price': Decimal('10000'),
        'time_in_force': TimeInForce.GTC,
    }, []),
    ('market', MarketOrderTicket, {
        'symbol': symbol,
        'side': OrderSide.BUY,
        'quantity': Decimal('1'),
        'quantity_type': MarketQuantityType.QUOTE,
        'estimated_price': Decimal('10000'),
    }, []),
    ('stop-loss', StopLossOrderTicket, *stop_loss_case),
    ('stop-loss-limit', StopLossLimitOrderTicket, *stop_loss_limit_case),
    ('take-profit', TakeProfitOrderTicket, *stop_loss_case),
    ('take-profit-limit', TakeProfitLimitOrderTicket, *stop_loss_limit_case),
]

TEST_REQUIRED_CASES = []

for (prefix, Order, kwargs, either) in RIGHT_CASES:
    TEST_REQUIRED_CASES.append((f'{prefix}-1', Order, kwargs, None))

    for i in range(len(kwargs)):
        new_kwargs = kwargs.copy()
        key = list(kwargs.keys())[i]
        del new_kwargs[key]
        TEST_REQUIRED_CASES.append(
            (
                f'{prefix}-{i+2} (missing {key})',
                Order,
                new_kwargs,
                None if key in either else [key, 'required']
            )
        )


def test_order_ticket():
    for prefix, Order, kwargs, exceptions in TEST_REQUIRED_CASES:
        try:
            Order(**kwargs)
        except Exception as e:
            if exceptions is None:
                assert False, f'{prefix}: unexpected exception "{e}"'
            else:
                for exception in exceptions:
                    assert exception in str(e), f'{prefix}: exception "{e}" does not contain "{exception}"'
        else:
            if exceptions is not None:
                assert False, f'{prefix}: expected exception "{exception}" but got none'
            else:
                assert True, f'{prefix}: passed'


def test_stop_loss_limit_order_ticket():
    with pytest.raises(ValueError, match='Either'):
        StopLossLimitOrderTicket(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=Decimal('1'),
            price=Decimal('10000'),
            time_in_force=TimeInForce.GTC,
            # stop_price=Decimal('10000'),
            # trailing_delta=Decimal('0.01'),
        )


def test_post_only_with_time_in_force():
    with pytest.raises(ValueError, match='post_only is not allowed with time_in_force'):
        LimitOrderTicket(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=Decimal('1'),
            price=Decimal('10000'),
            time_in_force=TimeInForce.GTC,
            post_only=True,
        )
