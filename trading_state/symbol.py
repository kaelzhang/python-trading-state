from typing import (
    List,
    Dict,
    Optional,
    overload,
)
from enum import Enum

from .filters import BaseFilter, FilterResult
from .order_ticket import OrderTicket
from .enums import FeatureType

class Symbol:
    """
    SymbolInfo is a class that contains the information of a symbol
    """

    __slots__ = (
        'name',
        'base_asset',
        'quote_asset'
    )

    name: str
    base_asset: str
    quote_asset: str

    _filters: List[BaseFilter]
    _allowed_features: Dict[FeatureType, bool | List[Enum]]


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

    @overload
    def allow(
        self,
        feature: FeatureType,
        allow: bool = True
    ) -> None:
        ...

    @overload
    def allow(
        self,
        feature: FeatureType,
        allow: List[Enum]
    ) -> None:
        ...

    def allow(
        self,
        feature: FeatureType,
        allow: bool | List[Enum]
    ) -> None:
        self._allowed_features[feature] = allow

    def support(
        self,
        feature: FeatureType,
        value: Optional[Enum] = None
    ) -> bool:
        allowed = self._allowed_features.get(feature, None)

        if allowed is None:
            # The feature is not specified, we treat it as not supported
            return False

        if isinstance(allowed, list):
            if value is None:
                raise ValueError(f'symbol.support {feature} requires a value, but got None')

            return value in allowed

        if value is not None:
            raise ValueError(f'symbol.support {feature} does not allow to test a value, but got {value}')

        return allowed

    def add_filter (self, filter: BaseFilter) -> None:
        self._filters.append(filter)

    def apply_filters(
        self,
        ticket: OrderTicket,
        validate_only: bool = False,
        **kwargs
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

            exception, new_modified = filter.apply(
                ticket, validate_only, **kwargs
            )

            if new_modified:
                modified = True

            if exception:
                return exception, modified

        return None, modified
