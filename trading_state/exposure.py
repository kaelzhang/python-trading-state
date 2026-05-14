"""
Exposure value object returned by `TradingState.exposure(...)`.

Only the four atomic inputs are stored; everything callers actually
use to make sizing decisions is exposed via @property and computed on
demand from those atoms.
"""
from dataclasses import dataclass
from decimal import Decimal

from .common import DECIMAL_ZERO


@dataclass(frozen=True, slots=True)
class Exposure:
    """
    Snapshot of an asset's exposure under the caller-selected
    include_unsettled_* policy.

    Atoms (stored):
        asset             — the asset queried.
        holding           — net base-asset units after applying the
                            chosen unsettled inflow / outflow toggles.
        valuation_price   — the asset's price in the account currency.
        notional_limit    — the configured cap (always > 0).

    Derived (`@property`, not stored):
        notional_value    — holding × valuation_price.
        ratio             — notional_value / notional_limit; may exceed
                            1 if the cap was lowered after the holding
                            was built up.
        headroom_notional — max(notional_limit − notional_value, 0).
        headroom_quantity — headroom_notional / valuation_price; size
                            BUY orders against this.
    """
    asset: str
    holding: Decimal
    valuation_price: Decimal
    notional_limit: Decimal

    @property
    def notional_value(self) -> Decimal:
        return self.holding * self.valuation_price

    @property
    def ratio(self) -> Decimal:
        return self.notional_value / self.notional_limit

    @property
    def headroom_notional(self) -> Decimal:
        return max(self.notional_limit - self.notional_value, DECIMAL_ZERO)

    @property
    def headroom_quantity(self) -> Decimal:
        return self.headroom_notional / self.valuation_price
