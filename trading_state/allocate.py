from typing import (
    Callable, List
)

from bisect import bisect_left
from decimal import Decimal

from .balance import Balance
from .symbol import Symbol
from .target import PositionTarget
from .enums import OrderSide


class AllocationResource:
    def __init__(
        self,
        symbol: Symbol,
        balance: Balance,
        weight: Decimal,
    ):
        self.symbol = symbol
        self.balance = balance
        self.weight = weight


Assigner = Callable[[Symbol, Decimal, PositionTarget, OrderSide], Decimal]


def buy_allocate(
    resources: List[AllocationResource],
    take: Decimal,
    target: PositionTarget,
    assign: Assigner,
) -> None:
    n = len(resources)

    # In each allocation round, we compute target for active buckets:
    #     Vj = V * Wj / sum_W
    # A bucket would "overflow" its capacity if:
    #     Vj > Sj  <=>  V / sum_W > Sj / Wj
    # Therefore,
    # sorting Sj/Wj allows a fast split using a threshold T = V/sum_W.
    order = sorted(
        range(n),
        key=lambda i: (resources[i].balance.free / resources[i].weight)
    )

    caps_sorted = [resources[i].balance.free for i in order]
    w_sorted = [resources[i].weight for i in order]
    ratio_sorted = [
        caps_sorted[i] / w_sorted[i]
        for i in range(n)
    ]

    # Active buckets are in the half-open interval [k, n).
    # Buckets in [0, k) have already been poured once and are excluded from future rounds.
    k = 0

    # Maintain totals for the active set for O(1) access each round.
    total_cap = sum(caps_sorted)  # Σ Sj over active buckets
    total_w = sum(w_sorted)       # Σ Wj over active buckets

    remaining = take  # Remaining target V (updates after each poured bucket)

    while k < n and remaining > 0:
        # Pour all water from each bucket
        if remaining >= total_cap:
            for t in range(k, n):
                assign(
                    resources[t].symbol,
                    # For BUY, must be positive
                    caps_sorted[t],
                    target,
                    OrderSide.BUY
                )
            break

        # Threshold T = V / Σ Wj. Buckets with (Sj / Wj) < T are not enough.
        T = remaining / total_w

        # Find first position p in ratio_sorted[k:n] such that
        #   ratio_sorted[p] >= T.
        # Then [k, p) are not-enough buckets
        p = bisect_left(ratio_sorted, T, lo=k, hi=n)

        if p == k:
            # Each bucket is enough,
            # then pour Vj for each active bucket, then stop.
            for t in range(k, n):
                assign(
                    resources[t].symbol,
                    (remaining * w_sorted[t]) / total_w,
                    target,
                    OrderSide.BUY
                )

            break

        # Fully pour all not-enough buckets in [k, p),
        # then update remaining and remove them.
        for t in range(k, p):
            # For BUY, must be positive
            pour = caps_sorted[t]
            # Remaining target update: V := V - Vj + RVj
            remaining = remaining - pour + assign(
                resources[t].symbol,
                pour,
                target,
                OrderSide.BUY
            )

            # Remove this bucket from future rounds
            # (each bucket is poured only once).
            total_cap -= pour
            total_w -= w_sorted[t]

        # Advance the active window boundary.
        k = p


def sell_allocate(
    resources: List[AllocationResource],
    take: Decimal,
    target: PositionTarget,
    assign: Assigner,
) -> None:
    pass
