"""
Allocation algorithms used by state.allocate() to split a single base
asset quantity across multiple account-currency symbols according to
caller-configured weights, while respecting per-bucket free balances
(BUY) and per-bucket exchange filters.

This module owns two flows used by `state.allocate`:

- `split_allocate` for tickets that can be sized in base units
  (LIMIT / LIMIT_MAKER / MARKET(BASE)). It runs aggregate BUY pre-
  flight against the asset's notional cap, then runs the math in
  `buy_allocate` / `sell_allocate` to fan a single canonical ticket
  out across weighted alt-currency buckets in
  `config.account_currencies` order.

- `passthrough_allocate` for tickets that cannot be split in base
  units (MARKET(QUOTE), stop / take-profit families). It applies
  filter normalization and runs pre-flight only on amounts that are
  precisely known up-front; anything that depends on trigger-time
  market price is passed through.

Both flows materialize Orders through `state._create_order`, which
attaches lifecycle listeners and emits `ORDER_CREATED`.
"""
from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
)

from bisect import bisect_left
from dataclasses import replace
from decimal import Decimal

from .symbol import Symbol
from .enums import MarketQuantityType, OrderSide
from .order_ticket import (
    OrderTicket,
    MarketOrderTicket,
    StopLossOrderTicket,
    StopLossLimitOrderTicket,
)
from .common import DECIMAL_INF, DECIMAL_ONE, DECIMAL_ZERO, ValueOrException

if TYPE_CHECKING:
    from .order import Order
    from .state import TradingState


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


# state-coupled allocation flows ---------------------------------------

def split_allocate(
    state: 'TradingState',
    ticket: OrderTicket,
    *,
    reference_price: Decimal,
    data: Optional[Dict[str, Any]],
) -> ValueOrException[List['Order']]:
    """
    Cross-currency split flow for LIMIT / LIMIT_MAKER / MARKET(BASE)
    tickets.

    BUY: aggregate notional pre-flight (best-effort: empty list on
    violation). Then split the base quantity across weighted buckets
    in `config.account_currencies` order, normalising each candidate
    via the symbol's filters; failed candidates are skipped.

    SELL: no notional pre-flight; same split logic, using
    `DECIMAL_INF` as the per-bucket free balance.
    """
    side = ticket.side
    base_asset = ticket.symbol.base_asset

    if side is OrderSide.BUY:
        exc, exposure_now = state.exposure(
            base_asset,
            include_unsettled_inflow=True,
            include_unsettled_outflow=False,
        )
        if exc is not None:
            return exc, None
        projected = (
            exposure_now.notional_value
            + ticket.quantity * reference_price
        )
        if projected > exposure_now.notional_limit:
            return None, []

    weights_vec = state._alt_currency_weights[
        0 if side is OrderSide.BUY else 1
    ]
    # alt weights first (matches alt_account_currencies order),
    # primary at the tail with implicit weight 1 — together matches
    # config.account_currencies.
    full_weights = (*weights_vec, DECIMAL_ONE)

    resources: List[AllocationResource] = []
    # symbol -> declaration position, used to sort the resulting
    # Orders into account_currencies order at the end.
    symbol_position: Dict[Symbol, int] = {}

    for i, acct_cur in enumerate(state._config.account_currencies):
        weight = full_weights[i]
        if weight <= DECIMAL_ZERO:
            continue
        symbol_name = state._config.get_symbol_name(base_asset, acct_cur)
        alt_symbol = state._symbols.get_symbol(symbol_name)
        if alt_symbol is None:
            continue
        if side is OrderSide.BUY:
            balance = state._balances.get_balance(acct_cur)
            if balance is None or balance.free <= DECIMAL_ZERO:
                continue
            free = balance.free
        else:
            free = DECIMAL_INF
        resources.append(AllocationResource(alt_symbol, free, weight))
        symbol_position[alt_symbol] = i

    if not resources:
        return None, []

    indexed: List[Tuple[int, 'Order']] = []

    def assign(symbol: Symbol, quantity: Decimal) -> Decimal:
        if quantity <= DECIMAL_ZERO:
            return DECIMAL_ZERO
        candidate = replace(ticket, symbol=symbol, quantity=quantity)
        exc, normalized = candidate.symbol.apply_filters(
            candidate, validate_only=False
        )
        if exc is not None:
            # Best-effort: skip this bucket silently.
            return quantity
        order = state._create_order(normalized, data)
        indexed.append((symbol_position[symbol], order))
        return quantity - normalized.quantity

    if side is OrderSide.BUY:
        buy_allocate(resources, ticket.quantity, reference_price, assign)
    else:
        sell_allocate(resources, ticket.quantity, assign)

    indexed.sort(key=lambda io: io[0])
    return None, [order for _, order in indexed]


def passthrough_allocate(
    state: 'TradingState',
    ticket: OrderTicket,
    *,
    data: Optional[Dict[str, Any]],
) -> ValueOrException[List['Order']]:
    """
    Passthrough flow for non-splittable tickets: `MarketOrderTicket
    (QUOTE)` and stop-loss / take-profit variants.

    Always applies filter normalization. Runs pre-flight only on
    precisely-known amounts (see `_check_passthrough_amounts`);
    anything that depends on the trigger-time market price (e.g. bare
    `STOP_LOSS` BUY notional) is silently passed through — the caller
    accepted exchange-side reject risk by choosing the ticket type.
    """
    side = ticket.side

    exc, normalized = ticket.symbol.apply_filters(
        ticket, validate_only=False
    )
    if exc is not None:
        return None, []

    notional, base_qty = _check_passthrough_amounts(normalized)
    base_asset = normalized.symbol.base_asset
    quote_asset = normalized.symbol.quote_asset

    if side is OrderSide.BUY:
        if notional is not None:
            exc, exposure_now = state.exposure(
                base_asset,
                include_unsettled_inflow=True,
                include_unsettled_outflow=False,
            )
            if exc is not None:
                return exc, None
            projected = exposure_now.notional_value + notional
            if projected > exposure_now.notional_limit:
                return None, []
            balance = state._balances.get_balance(quote_asset)
            if balance is None or balance.free < notional:
                return None, []
    else:
        if base_qty is not None:
            balance = state._balances.get_balance(base_asset)
            if balance is None or balance.free < base_qty:
                return None, []

    order = state._create_order(normalized, data)
    return None, [order]


def _check_passthrough_amounts(
    ticket: OrderTicket,
) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """
    Compute the (notional, base_quantity) pair that the passthrough
    pre-flight can rely on. `None` for either slot means "depends on
    trigger-time market price — do NOT use for pre-flight rejection".
    See README "Allocation pre-flight" for the carve-out table.
    """
    if isinstance(ticket, MarketOrderTicket):
        if ticket.quantity_type is MarketQuantityType.QUOTE:
            # quote_quantity IS the notional amount on BUY; base
            # quantity is unknown until the fill price lands.
            return ticket.quantity, None
        # MARKET(BASE) is split-flow; defensive None / quantity.
        return None, ticket.quantity

    # StopLossLimitOrderTicket / TakeProfitLimitOrderTicket: both have
    # a precise limit `price` -> notional & base both known.
    if isinstance(ticket, StopLossLimitOrderTicket):
        return ticket.quantity * ticket.price, ticket.quantity

    # Bare StopLossOrderTicket / TakeProfitOrderTicket: market-on-
    # trigger; notional depends on trigger-time price -> None.
    if isinstance(ticket, StopLossOrderTicket):
        return None, ticket.quantity

    # Defensive: any other passthrough ticket type.
    return None, None
