from typing import (
    Dict,
    Any
)

from decimal import Decimal

from .symbol import Symbol

from .common import (
    class_repr,
    # float_to_str,
    # datetime_now_str
)


PositionTargetMetaData = Dict[str, Any]


class PositionTarget:
    """An asset position target is the position expectation of an asset via trading with a certain symbol.

    The value of the position is based on the asset's notional limit.

    Args:
        symbol (Symbol): the symbol to trade with to achieve the position
        exposure (float): should between 0 and 1
        use_market_order (bool): whether to trade use_market_order (market order)
        price (Decimal | None): the price to trade at
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
        'achieved'
    )

    symbol: Symbol
    exposure: float
    use_market_order: bool
    price: Decimal | None
    data: PositionTargetMetaData
    achieved: bool

    def __init__(
        self,
        symbol: Symbol,
        exposure: float,
        use_market_order: bool,
        price: Decimal | None,
        data: PositionTargetMetaData
    ) -> None:
        self.symbol = symbol
        self.exposure = exposure
        self.use_market_order = use_market_order
        self.price = price
        self.data = data
        self.achieved = False

    def __repr__(self) -> str:
        return class_repr(
            self,
            main='symbol',
            keys=[
                'exposure',
                'use_market_order',
                'price'
            ]
        )

    def __eq__(
        self,
        target: Any
    ) -> bool:
        """To detect whether the given `PositionTarget` has the same goal of the current one
        """

        if not isinstance(target, PositionTarget):
            return False

        return (
            self.symbol == target.symbol
            and self.exposure == target.exposure
            and self.use_market_order == target.use_market_order
            and self.price == target.price
        )
