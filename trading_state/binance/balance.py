from typing import (
    Set
)
from decimal import Decimal

from trading_state import (
    Balance,
    CashFlow
)
from trading_state.common import (
    timestamp_to_datetime
)


# Ref
# https://github.com/binance/binance-spot-api-docs/blob/master/user-data-stream.md#balance-update
def decode_account_update_event(
    payload: dict
) -> Set[Balance]:
    """
    Generate balances from Binance account update, ie. the user stream event of `outboundAccountPosition`

    Args:
        payload (dict): the payload of the event

    Returns:
        Set[Balance]
    """

    balances = set[Balance]()
    time = timestamp_to_datetime(payload['u'])

    for balance in payload['B']:
        balances.add(
            Balance(
                balance['a'],
                Decimal(balance['f']),
                Decimal(balance['l']),
                time
            )
        )

    return balances


def decode_balance_update_event(
    payload: dict
) -> CashFlow:
    """
    Generate balances from Binance balance update

    Args:
        payload (dict): the payload of the event

    Returns:
        Set[Balance]
    """

    asset = payload['a']
    update = Decimal(payload['d'])
    clear_time = timestamp_to_datetime(payload['T'])

    return CashFlow(asset, update, clear_time)


def decode_account_info_response(account_info: dict) -> Set[Balance]:
    """
    Generate balances from Binance account info

    Args:
        account_info (dict): the account info

    Returns:
        Set[Balance]
    """

    balances = set[Balance]()
    time = timestamp_to_datetime(account_info['updateTime'])

    for balance in account_info['balances']:
        balances.add(
            Balance(
                balance['asset'],
                Decimal(balance['free']),
                Decimal(balance['locked']),
                time
            )
        )

    return balances
