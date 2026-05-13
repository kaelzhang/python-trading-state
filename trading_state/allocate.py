"""
Allocation algorithms used by state.allocate() to split a single base
asset quantity across multiple account-currency symbols according to
caller-configured weights, while respecting per-bucket free balances
(BUY) and per-bucket exchange filters.

The math is the same as before; the orchestration changed: the
`assign` callback used to be wired into state.update_order, now it is
"build a child ticket, run filters, append on success, return leftover
on failure".
"""

from typing import Callable, List

from bisect import bisect_left
from decimal import Decimal

from .symbol import Symbol
from .common import DECIMAL_ZERO


class AllocationResource:
    def __init__(
        self,
        symbol: Symbol,
        free: Decimal,
        weight: Decimal,
    ):
        self.symbol = symbol
        self.free = free
        self.weight = weight


# (symbol, base_quantity) -> leftover_base_quantity_to_redistribute
Assigner = Callable[[Symbol, Decimal], Decimal]


"""
Terminology:

  Math | Variable         | Description
 ----- | ---------------- | -------------------------------
   Sj  | caps(_sorted)[j] | the volume in each bucket
   Wj  | w(_sorted)[j]    | the weight of each bucket
   V   | remaining        | the remaining target volume to allocate
   Vj  | pour             | the volume to pour from each bucket
   RVj | ret              | the volume returned by the `assign` method
"""


def buy_allocate(
    resources: List[AllocationResource],
    take: Decimal,
    reference_price: Decimal,
    assign: Assigner,
) -> None:
    n = len(resources)

    # In each allocation round, we compute target for active buckets:
    #     Vj = V * Wj / sum_W
    # A bucket would not afford its target volume if:
    #     Vj > Sj  <=>  V / sum_W > Sj / Wj
    # Therefore,
    # sorting Sj/Wj allows a fast split using a threshold T = V/sum_W.
    order = sorted(
        range(n),
        key=lambda i: (resources[i].free / resources[i].weight)
    )

    caps_sorted = [resources[i].free for i in order]
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

    # `take` is for base quantity, so we need to convert it to quote quantity
    remaining = take * reference_price

    while k < n and remaining > 0:
        # Pour all water from each bucket.
        # Even `assign` method might return some water,
        #   we still do not have extra water to compensate
        if remaining >= total_cap:
            for t in range(k, n):
                assign(
                    resources[t].symbol,
                    # For BUY, must be positive
                    caps_sorted[t] / reference_price,
                )
            break # End

        # Threshold T = V / Σ Wj. Buckets with (Sj / Wj) < T are not enough.
        T = remaining / total_w

        # Find first position p in ratio_sorted[k:n] such that
        #   ratio_sorted[p] >= T.
        # Then [k, p) are not-enough buckets
        p = bisect_left(ratio_sorted, T, lo=k, hi=n)

        if p == k:
            compensate = DECIMAL_ZERO

            # Each bucket is enough,
            # then pour Vj for each active bucket, then stop.
            for t in range(k, n):
                # `assign` might return some water to the previous bucket,
                # so we need to compensate it with the current bucket
                pour = min(
                    compensate + (remaining * w_sorted[t]) / total_w,
                    caps_sorted[t]
                )

                compensate = assign(
                    resources[t].symbol,
                    pour / reference_price,
                ) * reference_price

            break # End

        # Fully pour all not-enough buckets in [k, p),
        # then update remaining and remove them.
        for t in range(k, p):
            # For BUY, must be positive
            pour = caps_sorted[t]

            # Remaining target update: V := V - (Vj - RVj)
            remaining -= pour - assign(
                resources[t].symbol,
                pour / reference_price,
            ) * reference_price

            # Remove this bucket from future rounds
            # (each bucket is poured only once).
            total_cap -= pour
            total_w -= w_sorted[t]

        # Advance the active window boundary.
        k = p


def sell_allocate(
    resources: List[AllocationResource],
    take: Decimal,
    assign: Assigner,
) -> None:
    total_w = sum(resource.weight for resource in resources)
    compensate = DECIMAL_ZERO

    # Sort resources by weight, so that in the worst case,
    # we will allocate more to the heaviest-weighted resource (the last one)
    for resource in sorted(resources, key=lambda resource: resource.weight):
        compensate = assign(
            resource.symbol,
            # We do not need to check caps for SELL
            compensate + (take * resource.weight) / total_w,
        )
