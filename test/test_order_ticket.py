from trading_state.order_ticket import (
    LimitOrderTicket,
    MarketOrderTicket,
    StopLossOrderTicket,
    StopLossLimitOrderTicket,
    TakeProfitOrderTicket,
    TakeProfitLimitOrderTicket,
)

from trading_state.symbol import Symbol
from trading_state.enums import OrderSide, OrderType, TimeInForce


symbol = Symbol(
    name='AB',
    base_asset='A',
    quote_asset='B',
)


RIGHT_CASES = [
    (LimitOrderTicket, {
        'symbol': symbol,
        'side': OrderSide.BUY,
        'quantity': '1',
        'price': '10000',
        'time_in_force': TimeInForce.GTC,
    })
]

TEST_REQUIRED_CASES = []

for (Order, kwargs) in RIGHT_CASES:
    TEST_REQUIRED_CASES.append((Order, kwargs, None))

    for i in range(len(kwargs)):
        new_kwargs = kwargs.copy()
        key = list(kwargs.keys())[i]
        del new_kwargs[key]
        TEST_REQUIRED_CASES.append(
            (Order, new_kwargs, [key, 'required'])
        )


def test_order_ticket():
    for index, (Order, kwargs, exceptions) in enumerate(TEST_REQUIRED_CASES):
        try:
            Order(**kwargs)
        except Exception as e:
            if exceptions is None:
                assert False, f'{index}: unexpected exception "{e}"'
            else:
                for exception in exceptions:
                    assert exception in str(e), f'{index}: exception "{e}" does not contain "{exception}"'
        else:
            if exceptions is not None:
                assert False, f'{index}: expected exception "{exception}" but got none'
            else:
                assert True, f'{index}: passed'

