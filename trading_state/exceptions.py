"""
Exceptions for the trading state, which are
- not caused by input value errors of users
  - that should be raised directly
- usually caused by improper dealing of the intialization process
"""

from __future__ import annotations
from decimal import Decimal
from typing import TYPE_CHECKING

from .enums import FeatureType

if TYPE_CHECKING:
    from .symbol import Symbol


class SymbolNotDefinedError(Exception):
    def __init__(self, symbol_name: str) -> None:
        message = f'symbol "{symbol_name}" is not defined yet'
        super().__init__(message)

        self.symbol_name = symbol_name


class SymbolPriceNotReadyError(Exception):
    def __init__(self, symbol_name: str) -> None:
        message = f'symbol price for "{symbol_name}" is not ready yet'
        super().__init__(message)

        self.symbol_name = symbol_name


class AssetNotDefinedError(Exception):
    def __init__(self, asset: str) -> None:
        message = f'asset "{asset}" is not defined'
        super().__init__(message)

        self.asset = asset


class ValuationNotAvailableError(Exception):
    def __init__(self, asset: str) -> None:
        message = f'valuation path for asset "{asset}" is not available'
        super().__init__(message)

        self.asset = asset


class ValuationPriceNotReadyError(Exception):
    def __init__(self, asset: str, symbol: Symbol) -> None:
        message = f'valuation price for "{asset}" through "{symbol.name}" is not ready yet'
        super().__init__(message)

        self.asset = asset
        self.symbol = symbol


class NotionalLimitNotSetError(Exception):
    def __init__(self, asset: str) -> None:
        message = f'notional limit of asset "{asset}" is not set'
        super().__init__(message)

        self.asset = asset


class BalanceNotReadyError(Exception):
    def __init__(self, asset: str) -> None:
        message = f'balance of asset "{asset}" is not ready yet'
        super().__init__(message)

        self.asset = asset


class FeatureNotAllowedError(Exception):
    def __init__(
        self,
        symbol: Symbol,
        feature: FeatureType,
        message: str
    ) -> None:
        super().__init__(message)

        self.symbol = symbol
        self.feature = feature


class InvalidExchangeData(Exception):
    """
    Raised at the protocol-adapter boundary when a decoded payload
    fails internal sanity checks (missing required field, negative
    quantity, commission with no asset, etc.).

    Never raised inside `trading_state.state`. Always carried back
    through `ValueOrException` from `trading_state.binance.*`
    decoders / encoders so callers cannot accidentally drop it.
    """


class AccountAssetHasNoExposureError(Exception):
    """
    Raised when `state.exposure(asset, ...)` is called with an asset
    that is one of the configured account currencies. Account
    currencies are the unit of measurement, not a risk-bearing
    position; exposure is undefined for them.
    """
    def __init__(self, asset: str) -> None:
        super().__init__(
            f"asset '{asset}' is an account currency; "
            f'exposure is undefined for account currencies'
        )
        self.asset = asset


class NotionalLimitExceededError(Exception):
    """
    Raised by `state.allocate(...)` when the canonical ticket cannot
    be sized within the asset's notional_limit even under a worst-case
    unsettled policy (inflow=True, outflow=False).
    """
    def __init__(
        self,
        asset: str,
        attempted_notional: Decimal,
        notional_limit: Decimal,
    ) -> None:
        super().__init__(
            f"BUY of asset '{asset}' would reach notional "
            f'{attempted_notional} which exceeds the configured '
            f'limit {notional_limit}'
        )
        self.asset = asset
        self.attempted_notional = attempted_notional
        self.notional_limit = notional_limit


class InsufficientFreeBalanceError(Exception):
    """
    Raised by `state.allocate(...)` for BUY tickets when every
    weighted account-currency bucket has zero free balance, so no
    sub-ticket can be funded.
    """
    def __init__(self, asset: str) -> None:
        super().__init__(
            f"no account-currency bucket has free balance to fund a "
            f"BUY for asset '{asset}'"
        )
        self.asset = asset


class AllocationWeightsNotSetError(Exception):
    """
    Raised by `state.allocate(...)` when called before
    `set_alt_currency_weights(...)`. Weights must be configured
    explicitly (even if `((), ())` for a single account currency)
    before allocation is allowed.
    """
    def __init__(self) -> None:
        super().__init__(
            'allocation weights have not been set; call '
            'state.set_alt_currency_weights(...) first'
        )

