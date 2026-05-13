from decimal import Decimal

from trading_state import Symbol
from trading_state.allocate import (
    buy_allocate,
    sell_allocate,
    AllocationResource,
)

from .fixtures import (
    BTCUSDT,
    BTCUSDC,
    BTCFDUSD,
)


resources = [
    AllocationResource(
        BTCUSDT,
        free=Decimal('10000'),
        weight=Decimal('1'),
    ),
    AllocationResource(
        BTCUSDC,
        free=Decimal('10000'),
        weight=Decimal('1.5'),
    ),
    AllocationResource(
        BTCFDUSD,
        free=Decimal('10000'),
        weight=Decimal('2.5'),
    ),
]

resources = sorted(resources, key=lambda r: r.symbol.name)
price = Decimal('10000')

results = []


def assign(symbol: Symbol, quantity: Decimal) -> Decimal:
    """
    A test stub for the Assigner callback. Forces a small "leftover"
    return when quantity is below 0.5 so we can verify the compensate
    chain wiring without involving a real ticket / filter path.
    """
    ret = Decimal('0')
    if quantity <= Decimal('0.5'):
        ret = Decimal('0.1')

    quantity -= ret

    results.append((symbol, quantity, ret))
    return ret


def match_results(prefix, quantities, returns):
    for i, (s, q, r) in enumerate(
        sorted(results, key=lambda r: r[0].name)
    ):
        assert s == resources[i].symbol, f'{prefix}: symbol'
        assert q == quantities[i], f'{prefix}: quantity'
        assert r == returns[i], f'{prefix}: return'


def test_buy_allocate():
    def run(take: Decimal):
        results.clear()
        buy_allocate(
            resources,
            take=take,
            reference_price=price,
            assign=assign,
        )

    # Buy 5 BTC, but quote balance is not enough
    run(Decimal('5'))
    match_results(
        '5',
        [Decimal('1')] * 3,
        [Decimal('0')] * 3,
    )

    # Buy 2 BTC, enough but with returns
    run(Decimal('2'))
    match_results(
        '2',
        [Decimal('1'), Decimal('0.6'), Decimal('0.3')],
        [Decimal('0'), Decimal('0'), Decimal('0.1')],
    )

    # Buy 1 BTC, enough but with multiple returns
    run(Decimal('1'))
    match_results(
        '1',
        [Decimal('0.4'), Decimal('0.3'), Decimal('0.2')],
        [Decimal('0.1'), Decimal('0.1'), Decimal('0.1')],
    )

    run(Decimal('2.5'))
    match_results(
        '2.5',
        [Decimal('1'), Decimal('0.9'), Decimal('0.6')],
        [Decimal('0'), Decimal('0'), Decimal('0')],
    )


def test_sell_allocate():
    def run(take: Decimal):
        results.clear()
        sell_allocate(
            resources,
            take=take,
            assign=assign,
        )

    # No returns
    run(Decimal('5'))
    match_results(
        '5',
        [Decimal('2.5'), Decimal('1.5'), Decimal('1')],
        [Decimal('0')] * 3,
    )

    # Single return
    run(Decimal('2'))
    match_results(
        '2',
        [Decimal('1'), Decimal('0.7'), Decimal('0.3')],
        [Decimal('0'), Decimal('0'), Decimal('0.1')],
    )

    # Multiple returns
    run(Decimal('1'))
    match_results(
        '1',
        [Decimal('0.6'), Decimal('0.3'), Decimal('0.1')],
        [Decimal('0'), Decimal('0.1'), Decimal('0.1')],
    )
