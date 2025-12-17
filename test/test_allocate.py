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
    return
    symbol_BTCUSDT = Symbol(BTCUSDT, BTC, USDT)
    symbol_BTCUSDC = Symbol(BTCUSDC, BTC, USDC)
    symbol_BTCFDUSD = Symbol(BTC + FDUSD, BTC, FDUSD)

    resources = [
        AllocationResource(
            symbol_BTCUSDT,
            balance=Balance(USDT, free=Decimal('100'), locked=Decimal('0')),
            weight=Decimal('0.5'),
        ),
        AllocationResource(
            symbol_BTCUSDC,
            balance=Balance(USDC, free=Decimal('100'), locked=Decimal('0')),
            weight=Decimal('0.5'),
        ),
        AllocationResource(
            symbol_BTCFDUSD,
            balance=Balance(FDUSD, free=Decimal('100'), locked=Decimal('0')),
            weight=Decimal('0.5'),
        ),
    ]

    results = []
    side = OrderSide.BUY

    def assign(
        symbol: Symbol,
        volume: Decimal,
        target: PositionTarget,
        side: OrderSide,
    ) -> Decimal:
        assert side is OrderSide.BUY

        results.append((symbol, volume, target, side))
        return volume

    buy_allocate(
        resources,
        take=Decimal('100'),
        target=PositionTarget(BTC, 100),
        assign=assign,
    )
