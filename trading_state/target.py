from typing import (
    Dict,
    Any
)
from decimal import Decimal

from .symbol import Symbol
from .common import class_repr
from .enums import PositionTargetStatus


PositionTargetMetaData = Dict[str, Any]


class PositionTarget:
    """An asset position target is the position expectation of an asset via trading with a certain symbol.

    The value of the position is based on the asset's notional limit.

    Args:
        symbol (Symbol): the symbol to trade with to achieve the position
        exposure (float): should between 0 and 1
        use_market_order (bool): whether to trade use_market_order (market order)
        price (Decimal): the price to trade at. For market order, it should be the estimated average price for
        data (dict[str, Any]): the meta data of the position

    For example::

        position = PositionTarget(
            symbol=Symbol(
                name='BTCBNB',
                base_asset='BTC',
                quote_asset='BNB',
                ...
            ),
            exposure=1.0,
            use_market_order=True,
            price=None,
            data={}
        )

    If the notional limit of the BTC is 1000 USDT, the `position` means that the trader should buy or sell BTCBNB so that the valuation value of the BTC is 1000 USDT.
    """

    __slots__ = (
        'symbol',
        'exposure',
        'use_market_order',
        'price',
        'data',
        'status'
    )

    symbol: Symbol
    exposure: Decimal
    use_market_order: bool
    price: Decimal
    data: PositionTargetMetaData
    status: PositionTargetStatus

    def __init__(
        self,
        symbol: Symbol,
        exposure: Decimal,
        use_market_order: bool,
        price: Decimal,
        data: PositionTargetMetaData
    ) -> None:
        self.symbol = symbol
        self.exposure = exposure
        self.use_market_order = use_market_order
        self.price = price
        self.data = data
        self.status = PositionTargetStatus.INIT

    def __repr__(self) -> str:
        return class_repr(
            self,
            main='symbol',
            keys=[
                'exposure',
                'use_market_order',
                'price',
                'status'
            ]
        )
