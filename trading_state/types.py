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


PositionMetaData = Dict[str, Any]


class AssetPositionTarget:
    """An asset position target is the position expectation of an asset via trading with a certain symbol.

    The value of the position is based on the asset's notional limit.

    Args:
        symbol (Symbol): the symbol to trade with to achieve the position
        utilization (float): should between 0 and 1
        immediate (bool): whether to trade immediate (market order)
        price (Decimal | None): the price to trade at
        data (dict[str, Any]): the meta data of the position

    For example::

        position = AssetPositionTarget(
            symbol=Symbol(
                name='BTCBNB',
                base_asset='BTC',
                quote_asset='BNB',
                ...
            ),
            utilization=1.0,
            immediate=True,
            price=None,
            data={}
        )

    If the notional limit of the BTC is 1000 USDT, the `position` means that the trader should buy or sell BTCBNB so that the valuation value of the BTC is 1000 USDT.
    """

    __slots__ = (
        'symbol',
        'utilization',
        'immediate',
        'price',
        'data',
        'fulfilled'
    )

    symbol: Symbol
    utilization: float
    immediate: bool
    price: Decimal | None
    data: PositionMetaData
    fulfilled: bool

    def __init__(
        self,
        symbol: Symbol,
        utilization: float,
        immediate: bool,
        price: Decimal | None,
        data: PositionMetaData
    ) -> None:
        self.symbol = symbol
        self.utilization = utilization
        self.immediate = immediate
        self.price = price
        self.data = data
        self.fulfilled = False

    def __repr__(self) -> str:
        return class_repr(
            self,
            main='symbol',
            keys=[
                'utilization',
                'immediate',
                'price'
            ]
        )

    def __eq__(
        self,
        target: Any
    ) -> bool:
        """To detect whether the given `AssetPositionTarget` has the same goal of the current one
        """

        if not isinstance(target, AssetPositionTarget):
            return False

        return (
            self.symbol == target.symbol
            and self.utilization == target.utilization
            and self.immediate == target.immediate
            and self.price == target.price
        )
