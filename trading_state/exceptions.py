"""
Exceptions for the trading state, which are
- not caused by input value errors of users
  - that should be raised directly
- usually caused by improper dealing of the intialization process
"""


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


class ValuationPriceNotReadyError(Exception):
    def __init__(self, asset: str) -> None:
        message = f'valuation price for "{asset}" is not ready yet'
        super().__init__(message)

        self.asset = asset


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
