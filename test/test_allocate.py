from decimal import Decimal

from trading_state import (
    Symbol,
    PositionTarget,
    OrderSide
)

from trading_state.allocate import (
    buy_allocate,
    sell_allocate,
    AllocationResource,
)

from .fixtures import (
    BTCUSDT,
    BTCUSDC,
    BTCFDUSD
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

target = PositionTarget(
    symbol=BTCUSDT,
    # Arbitrary value, has nothing to do with the allocation
    exposure=Decimal('0.1'),
    use_market_order=False,
    price=price,
    data={},
)

results = []

def assign(
    symbol: Symbol,
    quantity: Decimal,
    target: PositionTarget,
    side: OrderSide,
) -> Decimal:
    ret = Decimal('0')
    if quantity <= Decimal('0.5'):
        ret = Decimal('0.1')

    quantity -= ret

    results.append((symbol, quantity, ret, target, side))
    return ret


def run_allocate(take: Decimal, func):
        results.clear()
        func(
            resources,
            take=take,
            target=target,
            assign=assign,
        )


def match_results(prefix, quantities, returns, side = OrderSide.BUY):
    for i, (s, q, r, t, d) in enumerate(
        sorted(results, key=lambda r: r[0].name)
    ):
        assert t == target, f'{prefix}: target'
        assert d == side, f'{prefix}: side'
        assert s == resources[i].symbol, f'{prefix}: symbol'
        assert q == quantities[i], f'{prefix}: quantity'
        assert r == returns[i], f'{prefix}: return'


def test_buy_allocate():
    def run(take: Decimal):
        run_allocate(take, buy_allocate)


    # Buy 5 BTC, but quote balance is not enough
    run(Decimal('5'))
    match_results(
        '5',
        [Decimal('1')] * 3,
        [Decimal('0')] * 3
    )

    # Buy 2 BTC, enough but with returns
    run(Decimal('2'))
    match_results(
        '2',
        [Decimal('1'), Decimal('0.6'), Decimal('0.3')],
        [Decimal('0'), Decimal('0'), Decimal('0.1')]
    )

    # Buy 1 BTC, enough but with multiple returns
    run(Decimal('1'))
    match_results(
        '1',
        [Decimal('0.4'), Decimal('0.3'), Decimal('0.2')],
        [Decimal('0.1'), Decimal('0.1'), Decimal('0.1')]
    )

    run(Decimal('2.5'))
    match_results(
        '2.5',
        [Decimal('1'), Decimal('0.9'), Decimal('0.6')],
        [Decimal('0'), Decimal('0'), Decimal('0')]
    )


def test_sell_allocate():
    def run(take: Decimal):
        run_allocate(take, sell_allocate)

    # No returns
    run(Decimal('5'))
    match_results(
        '5',
        [Decimal('2.5'), Decimal('1.5'), Decimal('1')],
        [Decimal('0')] * 3,
        OrderSide.SELL
    )

    # Single return
    run(Decimal('2'))
    match_results(
        '2',
        [Decimal('1'), Decimal('0.7'), Decimal('0.3')],
        [Decimal('0'), Decimal('0'), Decimal('0.1')],
        OrderSide.SELL
    )

    # Multiple returns
    run(Decimal('1'))
    match_results(
        '1',
        [Decimal('0.6'), Decimal('0.3'), Decimal('0.1')],
        [Decimal('0'), Decimal('0.1'), Decimal('0.1')],
        OrderSide.SELL
    )
