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


class SymbolPosition:
    """The position ratio relative to the whole balance value

    Args:
        value (float): should between 0 and 1, for now only support 0 or 1
    """

    __slots__ = (
        'symbol',
        'value',
        'asap',
        'price',
        'data'
    )

    symbol: Symbol
    value: float
    asap: bool
    price: Decimal | None
    data: PositionMetaData

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
        position: 'SymbolPosition'
    ) -> bool:
        """To detect whether the given `SymbolPosition` has the same goal of the current one
        """

        if not isinstance(position, SymbolPosition):
            return False

        return (
            self.symbol == position.symbol
            and self.value == position.value
            and self.asap == position.asap
            and self.price == position.price
        )
