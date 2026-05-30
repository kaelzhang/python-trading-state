"""
Exceptions for the trading state, which are
- not caused by input value errors of users
  - that should be raised directly
- usually caused by improper dealing of the intialization process
"""

from __future__ import annotations
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


class InvalidAllocationWeightsError(Exception):
    """
    Raised by `state.create_order(..., allocate=weights)` when the
    weights tuple is malformed. `reason` is a short string describing
    which check failed — e.g. length-mismatch against
    `config.alt_account_currencies` or a negative weight.

    Weights live on the caller's side (recomputed per call from live
    book depth, stablecoin balances, basis, etc.); they are not part
    of `trading_state`'s setup. This exception only fires when the
    caller's vector is structurally wrong, not for business outcomes
    such as "all buckets exhausted".
    """
    def __init__(self, reason: str) -> None:
        super().__init__(
            f'invalid allocation weights: {reason}'
        )
        self.reason = reason


class DuplicateOrderIdError(Exception):
    """
    Raised by `state.import_order(...)` when the order's id is already
    present in state. Recovery callers must dispatch on
    `state.get_order_by_id(id) is None` and route already-known ids
    through `state.update_order(...)` instead — a duplicate import
    indicates a caller orchestration bug.
    """
    def __init__(self, order_id: str) -> None:
        super().__init__(
            f"cannot import order: id '{order_id}' is already present "
            f'in state; use state.update_order(...) to push fields '
            f'into an existing order'
        )
        self.order_id = order_id


class InvalidOrderForImportError(Exception):
    """
    Raised by `state.import_order(...)` when the order does not satisfy
    the import preconditions: a non-None `id` and a non-None `ticket`
    are mandatory. The decoder helpers in `trading_state.binance.order`
    always produce a well-formed Order; this guards against caller
    bugs that fabricate Orders manually.
    """
    def __init__(self, reason: str) -> None:
        super().__init__(
            f'cannot import order: {reason}'
        )
        self.reason = reason

