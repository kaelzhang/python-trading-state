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


class AssetPosition:
    """An asset position is the position expectation of an asset via trading with a certain symbol.

    The value of the position is based on the asset's notional limit.

    Args:
        symbol (Symbol): the symbol to trade with to achieve the position
        value (float): should between 0 and 1
        asap (bool): whether to trade asap (market order)
        price (Decimal | None): the price to trade at
        data (dict[str, Any]): the meta data of the position

    For example::

        position = AssetPosition(
            symbol=Symbol(
                name='BTCBNB',
                base_asset='BTC',
                quote_asset='BNB',
                ...
            ),
            value=1.0,
            asap=True,
            price=None,
            data={}
        )

    If the notional limit of the BTC is 1000 USDT, the `position` means that the trader should buy or sell BTCBNB so that the valuation value of the BTC is 1000 USDT.
    """

    __slots__ = (
        'symbol',
        'value',
        'asap',
        'price',
        'data',
        'reached'
    )

    symbol: Symbol
    value: float
    asap: bool
    price: Decimal | None
    data: PositionMetaData
    reached: bool

    def __init__(
        self,
        symbol: Symbol,
        value: float,
        asap: bool,
        price: Decimal | None,
        data: PositionMetaData
    ) -> None:
        self.symbol = symbol
        self.value = value
        self.asap = asap
        self.price = price
        self.data = data
        self.reached = False

    def __repr__(self) -> str:
        return class_repr(
            self,
            main='symbol',
            keys=[
                'value',
                'asap',
                'price'
            ]
        )

    def __eq__(
        self,
        position: Any
    ) -> bool:
        """To detect whether the given `AssetPosition` has the same goal of the current one
        """

        if not isinstance(position, AssetPosition):
            return False

        return (
            self.symbol == position.symbol
            and self.value == position.value
            and self.asap == position.asap
            and self.price == position.price
        )
