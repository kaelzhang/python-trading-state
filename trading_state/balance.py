# Balance Manager

from decimal import Decimal

from .common import class_repr


class Balance:
    __slots__ = (
        'asset',
        'free',
        'locked'
    )

    asset: str
    free: Decimal
    locked: Decimal

    def __init__(
        self,
        asset: str,
        free: Decimal,
        locked: Decimal
    ):
        self.asset = asset
        self.free = free
        self.locked = locked

    @property
    def total(self) -> Decimal:
        return self.free + self.locked

    def __repr__(self) -> str:
        return class_repr(self, main='asset')
