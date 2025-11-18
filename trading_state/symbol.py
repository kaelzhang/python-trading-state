from typing import (
    List
)

from .filters import BaseFilter, FilterResult
from .order import OrderTicket


class Symbol:
    """
    SymbolInfo is a class that contains the information of a symbol
    """

    __slots__ = (
        'symbol',

        'min_price',
        'max_price'
    )

    name: str
    base_asset: str
    quote_asset: str

    _filters: List[BaseFilter]

    def __repr__(self) -> str:
        return f'<Symbol {self.base_asset} / {self.quote_asset}>'

    def __init__(
        self,
        name,
        base_asset: str,
        quote_asset: str
    ):
        self.name = name
        self.base_asset = base_asset
        self.quote_asset = quote_asset
        self._filters = []

    def add_filter (self, filter: BaseFilter) -> None:
        self._filters.append(filter)

    def apply_filters(
        self,
        ticket: OrderTicket,
        validate_only: bool = False
    ) -> FilterResult:
        """
        Apply the filter to the order ticket, and try to fix the ticket if possible if `validate_only` is `False`.

        Args:
            ticket: (OrderTicket) the order ticket to apply the filter to
            validate_only: (Optional[bool]=False) whether only to validate the ticket. If `True`, the filter will NOT try to fix the ticket and return an exception even for a tiny mismatch against the filter.

        Returns a tuple of
        - Optional[Exception]: the exception if the filter is not successfully applied
        - bool: whether the ticket has been modified
        """

        modified = False

        for filter in self._filters:
            if not filter.when(ticket):
                continue

            exception, new_modified = filter.apply(ticket, validate_only)

            if new_modified:
                modified = True

            if exception:
                return exception, modified

        return None, modified
