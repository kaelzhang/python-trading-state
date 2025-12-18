from decimal import Decimal

from trading_state import (
    Balance,
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
    BTC,
    USDT,
    BTCUSDT,
    USDC,
    BTCUSDC
)

FDUSD = 'FDUSD'


def test_allocate():
    symbol_BTCUSDT = Symbol(BTCUSDT, BTC, USDT)
    symbol_BTCUSDC = Symbol(BTCUSDC, BTC, USDC)
    symbol_BTCFDUSD = Symbol(BTC + FDUSD, BTC, FDUSD)

    resources = [
        AllocationResource(
            symbol_BTCUSDT,
            balance=Balance(USDT, free=Decimal('10000'), locked=Decimal('0')),
            weight=Decimal('1'),
        ),
        AllocationResource(
            symbol_BTCUSDC,
            balance=Balance(USDC, free=Decimal('10000'), locked=Decimal('0')),
            weight=Decimal('1.5'),
        ),
        AllocationResource(
            symbol_BTCFDUSD,
            balance=Balance(FDUSD, free=Decimal('10000'), locked=Decimal('0')),
            weight=Decimal('2.5'),
        ),
    ]

    resources = sorted(resources, key=lambda r: r.symbol.name)

    target = PositionTarget(
        symbol=symbol_BTCUSDT,
        # Arbitrary value, has nothing to do with the allocation
        exposure=Decimal('0.1'),
        use_market_order=False,
        price=Decimal('10000'),
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
        if quantity < Decimal('0.5'):
            ret = Decimal('0.1')

        quantity -= ret

        results.append((symbol, quantity, ret, target, side))

    # Buy 5 BTC, but quote balance is not enough
    take = Decimal('5')
    buy_allocate(
        resources,
        take=take,
        target=target,
        assign=assign,
    )

    for i, (s, q, r, t, d) in enumerate(
        sorted(results, key=lambda r: r[0].name)
    ):
        assert s == resources[i].symbol, 'symbol'
        assert q == Decimal('1'), 'quantity'
        assert r == Decimal('0'), 'return'
        assert t == target, 'target'
        assert d == OrderSide.BUY, 'side'
