class ExpectWithoutQuotaError(Exception):
    def __init__(self, asset: str) -> None:
        message = f'refused to set expectation to asset "{asset}" without quota'
        super().__init__(message)


class SymbolNotDefinedError(Exception):
    def __init__(self, symbol_name: str) -> None:
        message = f'symbol "{symbol_name}" is not defined yet'
        super().__init__(message)


class NumerairePriceNotReadyError(Exception):
    def __init__(self, asset: str) -> None:
        message = f'numeraire price for "{asset}" is not ready yet'
        super().__init__(message)
