from .types import Symbol

from .util import float_to_str


# SYMBOL_FILTER_TYPE_LOT_SIZE = 'lot_size'


class SymbolInfo:
    __slots__ = (
        'symbol',

        'min_price',
        'max_price',

        # Minimum price step of a symbol
        'min_price_step',
        '_min_price_step_precision',

        'min_quantity',
        'max_quantity',

        # Minimum quantity step of an asset based on quote asset
        'min_quantity_step',
        'min_quantity_step_precision',

        # Minimum value which is the product of price and quantity
        'min_notional'
    )

    symbol: Symbol

    min_price: float
    max_price: float
    min_price_step: float

    min_quantity: float
    max_quantity: float
    min_quantity_step: float
    min_quantity_step_precision: int

    min_notional: float

    def __init__(
        self,
        symbol: str,
        base_asset: str,
        quote_asset: str,

        min_price: str,
        max_price: str,
        min_price_step: str,

        min_quantity: str,
        max_quantity: str,
        min_quantity_step: str,

        # An order's notional value is the `price` * `quantity`
        min_notional: str
    ):
        self.symbol = Symbol(
            symbol,
            base_asset,
            quote_asset
        )

        self.min_price = float(min_price)
        self.max_price = float(max_price)

        self.min_price_step = float(min_price_step)
        self._min_price_step_precision = len(
            float_to_str(self.min_price_step)
        ) - 2

        self.min_quantity = float(min_quantity)
        self.max_quantity = float(max_quantity)

        self.min_quantity_step = float(min_quantity_step)
        self.min_quantity_step_precision = len(
            float_to_str(self.min_quantity_step)
        ) - 2

        self.min_notional = float(min_notional)

    def add_filter () -> None:


    def __repr__(self) -> str:
        price = format(
            self.min_price_step,
            f'.{self._min_price_step_precision}f'
        )

        quantity = format(
            self.min_quantity_step,
            f'.{self.min_quantity_step_precision}f'
        )

        return f'<SymbolInfo {self.symbol}: price_step >= {price}, quantity_step >= {quantity}>'
