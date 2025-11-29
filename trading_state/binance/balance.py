from typing import Set
from decimal import Decimal

from trading_state import Balance

Balances = Set[Balance]


def generate_balances_from_account_update(payload: dict) -> Set[Balance]:
    """
    Generate balances from Binance account update, ie. the user stream event of `outboundAccountPosition`

    Args:
        payload (dict): the payload of the event

    Returns:
        Set[Balance]
    """

    balances = set[Balance]()

    for balance in payload['B']:
        balances.add(
            Balance(
                balance['a'],
                Decimal(balance['f']),
                Decimal(balance['l'])
            )
        )

    return balances


def generate_balances_from_account_info(account_info: dict) -> Set[Balance]:
    """
    Generate balances from Binance account info

    Args:
        account_info (dict): the account info

    Returns:
        Set[Balance]
    """

    balances = set[Balance]()

    for balance in account_info['balances']:
        balances.add(
            Balance(
                balance['asset'],
                Decimal(balance['free']),
                Decimal(balance['locked'])
            )
        )

    return balances
