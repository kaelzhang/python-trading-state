from dataclasses import (
    dataclass,
    field
)
from typing import (
    Tuple,
    Dict,
    Any,
    Callable
)


def DEFAULT_GET_SYMBOL_NAME(base_asset: str, quote_asset: str) -> str:
    return base_asset + quote_asset


@dataclass(frozen=True)
class TradingConfig:
    """
    Args:
        account_currency (str): the default account currency (ref: https://en.wikipedia.org/wiki/Num%C3%A9raire) to use to:
        - calculate value of limit exposures
        - calculate value of notional limits

        alt_account_currencies (Set[str]): the alternative account currencies to the account currency.

        max_order_history_size (int): the maximum size of the order history

        get_symbol_name (Callable[[str, str], str]): a function to get the name of a symbol from its base and quote assets

        benchmark_assets (Tuple[str, ...]): the assets to benchmark the performance of the strategy

        symbols (Tuple[str, ...]): the list of symbols to take into account. If not provided, then all the symbols will be taken into account.
    """
    account_currency: str
    alt_account_currencies: Tuple[str, ...] = field(default_factory=tuple)

    context: Dict[str, Any] = field(default_factory=dict)
    max_order_history_size: int = 10000
    get_symbol_name: Callable[[str, str], str] = DEFAULT_GET_SYMBOL_NAME

    benchmark_assets: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def account_currencies(self) -> Tuple[str, ...]:
        # Put the account currency at the end, so that
        # Most usually we will deal it at last
        return (*self.alt_account_currencies, self.account_currency)

    def __post_init__(self) -> None:
        if self.account_currency in self.alt_account_currencies:
            raise ValueError(
                f'The default account currency "{self.account_currency}" must not be in the `alt_account_currencies`'
            )
