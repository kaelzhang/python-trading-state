from decimal import Decimal

from trading_state import Balance

from .fixtures import BTC


def test_balance():
    balance = Balance(
        asset=BTC,
        free=Decimal('1.0'),
        locked=Decimal('0.0')
    )

    assert repr(balance) == 'Balance(BTC free=1.0, locked=0.0)'
