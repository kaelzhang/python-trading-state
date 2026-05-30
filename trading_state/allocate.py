"""
Allocation backends called by `state.create_order(...)`. Four
state-coupled flows, picked by `state.create_order` according to the
ticket type and the presence of the `allocate=` keyword:

  - `single_create`        — `allocate=None` for base-quantity tickets
                             (LIMIT / MARKET(BASE) / stop / take-profit
                             / their *Limit variants).  No split,
                             ticket.symbol preserved.

  - `single_create_quote`  — `allocate=None` for `MarketOrderTicket
                             (QUOTE)`.  No split, base/quote-aware
                             pre-flight on the original symbol.

  - `split_allocate`       — `allocate=weights_vec` for base-quantity
                             tickets.  Splits the base quantity across
                             every weighted account-currency bucket.

  - `split_allocate_quote` — `allocate=weights_vec` for `MarketOrder
                             Ticket(QUOTE)`.  Converts the quote
                             quantity to base via `estimated_price`,
                             runs the standard split, then converts
                             each sub-allocation back to a per-bucket
                             quote amount in that bucket's quote
                             currency.

The split flows receive `weights_vec` from the caller of
`state.create_order` rather than from a state field; the caller is
expected to compute weights live from book depth, available
stablecoin balances, basis, inventory skew, etc.

All four flows materialize Orders through `state._create_order`,
which attaches lifecycle listeners and emits `ORDER_CREATED`.

Splitting (when the caller asks for it) assumes the alt account
currencies are stablecoins pegged to the primary — see README §1.
Under that assumption, `estimated_price` (and limit `price`, and
`stop_price`) used as a cross-bucket reference produces orders whose
effective per-bucket quote amounts are equivalent up to basis noise.
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
from .enums import OrderSide
from .order_ticket import (
    MarketOrderTicket,
    OrderTicket,
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
    weights_vec: Tuple[Decimal, ...],
    reference_price: Decimal,
    data: Optional[Dict[str, Any]],
) -> ValueOrException[List['Order']]:
    """
    Cross-currency split flow for base-quantity tickets: LIMIT /
    LIMIT_MAKER / MARKET(BASE) / STOP_LOSS / STOP_LOSS_LIMIT /
    TAKE_PROFIT / TAKE_PROFIT_LIMIT.

    `weights_vec` is the per-call vector the caller passed to
    `state.create_order(..., allocate=...)`. It is aligned with
    `config.alt_account_currencies`; the primary account currency is
    given implicit weight 1 and lives at the tail of the full vector.

    `reference_price` is the per-ticket price reference the caller's
    dispatch chose: `ticket.price` for LIMIT-side variants,
    `ticket.estimated_price` for MARKET(BASE), `ticket.stop_price`
    for bare stop / take-profit (estimate — actual fill price at
    trigger may differ; the trade-off was accepted in exchange for
    splitting MARKET-on-trigger orders across alt buckets).

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


def split_allocate_quote(
    state: 'TradingState',
    ticket: 'MarketOrderTicket',
    *,
    weights_vec: Tuple[Decimal, ...],
    data: Optional[Dict[str, Any]],
) -> ValueOrException[List['Order']]:
    """
    Cross-currency split for `MarketOrderTicket(QUOTE)`.

    The ticket's `quantity` field is in the original symbol's quote
    units (e.g. "spend 1000 USDT to BUY BTC"). The split converts to
    base via `ticket.estimated_price`, runs the same split math as
    `split_allocate`, then converts each sub-allocation back into a
    per-bucket quote amount using the same `estimated_price`. The
    sub-ticket placed on each bucket is itself a
    `MarketOrderTicket(QUOTE)`, so its `quantity` is the per-bucket
    quote amount in that bucket's quote currency.

    `weights_vec` is the per-call vector passed via
    `state.create_order(..., allocate=...)`; same semantics as
    `split_allocate`.

    BUY: aggregate notional pre-flight uses `ticket.quantity`
    directly as the projected notional (stablecoin parity makes the
    primary-quote amount the same notional in every alt bucket). Each
    bucket's free quote balance is used as a cap inside the split
    math.

    SELL: aggregate free-base pre-flight checks that the base asset
    has at least `base_total = ticket.quantity / estimated_price`
    available across all sub-orders combined.

    Splitting depends on alt account currencies being stablecoins
    pegged to the primary (see README §1).
    """
    side = ticket.side
    base_asset = ticket.symbol.base_asset
    estimated_price = ticket.estimated_price
    base_total = ticket.quantity / estimated_price

    if side is OrderSide.BUY:
        exc, exposure_now = state.exposure(
            base_asset,
            include_unsettled_inflow=True,
            include_unsettled_outflow=False,
        )
        if exc is not None:
            return exc, None
        projected = exposure_now.notional_value + ticket.quantity
        if projected > exposure_now.notional_limit:
            return None, []
    else:
        balance = state._balances.get_balance(base_asset)
        if balance is None or balance.free < base_total:
            return None, []

    full_weights = (*weights_vec, DECIMAL_ONE)

    resources: List[AllocationResource] = []
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

    def assign(symbol: Symbol, sub_base: Decimal) -> Decimal:
        if sub_base <= DECIMAL_ZERO:  # pragma: no cover
            # Defensive: buy_allocate / sell_allocate never assign a
            # non-positive amount under valid inputs, but guard so a
            # future change to the math can't produce a sub-ticket
            # with quantity == 0.
            return DECIMAL_ZERO
        # Re-derive the per-bucket quote amount (in the alt symbol's
        # quote currency) from sub_base using the same estimated_price.
        sub_quote = sub_base * estimated_price
        candidate = replace(ticket, symbol=symbol, quantity=sub_quote)
        exc, normalized = candidate.symbol.apply_filters(
            candidate, validate_only=False
        )
        if exc is not None:
            return sub_base
        # Translate the (possibly-normalized) quote quantity back to
        # base for the leftover-return contract of `Assigner`.
        normalized_base = normalized.quantity / estimated_price
        order = state._create_order(normalized, data)
        indexed.append((symbol_position[symbol], order))
        return sub_base - normalized_base

    if side is OrderSide.BUY:
        buy_allocate(resources, base_total, estimated_price, assign)
    else:
        sell_allocate(resources, base_total, assign)

    indexed.sort(key=lambda io: io[0])
    return None, [order for _, order in indexed]


def single_create(
    state: 'TradingState',
    ticket: OrderTicket,
    *,
    reference_price: Decimal,
    data: Optional[Dict[str, Any]],
) -> ValueOrException[List['Order']]:
    """
    No-split flow for base-quantity tickets: LIMIT / LIMIT_MAKER /
    MARKET(BASE) / STOP_LOSS / STOP_LOSS_LIMIT / TAKE_PROFIT /
    TAKE_PROFIT_LIMIT, when the caller passes `allocate=None`.

    Unlike `split_allocate`, `ticket.symbol` is preserved — the
    caller's choice of pair is final, and the library never
    substitutes a primary-symbol equivalent.

    Behaviour mirrors a single-bucket split: filter normalization,
    then per-side pre-flight against the asset's notional cap and
    the caller's free balance. Filter / notional / balance failures
    are best-effort: returns `(None, [])` so the caller sees the same
    "nothing landed" signal as the split path.

    `reference_price` is the per-ticket price reference picked by
    `state.create_order`'s dispatch (`ticket.price`,
    `ticket.estimated_price`, `ticket.stop_price`, or the symbol's
    last `set_price` for trailing-delta stops).
    """
    side = ticket.side
    base_asset = ticket.symbol.base_asset
    quote_asset = ticket.symbol.quote_asset

    exc, normalized = ticket.symbol.apply_filters(
        ticket, validate_only=False
    )
    if exc is not None:
        return None, []

    if side is OrderSide.BUY:
        exc, exposure_now = state.exposure(
            base_asset,
            include_unsettled_inflow=True,
            include_unsettled_outflow=False,
        )
        if exc is not None:
            return exc, None
        notional = normalized.quantity * reference_price
        if (
            exposure_now.notional_value + notional
            > exposure_now.notional_limit
        ):
            return None, []
        balance = state._balances.get_balance(quote_asset)
        if balance is None or balance.free < notional:
            return None, []
    else:
        balance = state._balances.get_balance(base_asset)
        if balance is None or balance.free < normalized.quantity:
            return None, []

    order = state._create_order(normalized, data)
    return None, [order]


def single_create_quote(
    state: 'TradingState',
    ticket: 'MarketOrderTicket',
    *,
    data: Optional[Dict[str, Any]],
) -> ValueOrException[List['Order']]:
    """
    No-split flow for `MarketOrderTicket(QUOTE)` when the caller
    passes `allocate=None`.

    The ticket's `quantity` is already the notional, so the BUY
    notional pre-flight uses it directly. SELL needs a base-side
    free-balance check; the implied base is `quantity /
    estimated_price`.

    `ticket.symbol` is preserved.
    """
    side = ticket.side
    base_asset = ticket.symbol.base_asset
    quote_asset = ticket.symbol.quote_asset
    estimated_price = ticket.estimated_price
    base_total = ticket.quantity / estimated_price

    exc, normalized = ticket.symbol.apply_filters(
        ticket, validate_only=False
    )
    if exc is not None:
        return None, []

    if side is OrderSide.BUY:
        exc, exposure_now = state.exposure(
            base_asset,
            include_unsettled_inflow=True,
            include_unsettled_outflow=False,
        )
        if exc is not None:
            return exc, None
        if (
            exposure_now.notional_value + ticket.quantity
            > exposure_now.notional_limit
        ):
            return None, []
        balance = state._balances.get_balance(quote_asset)
        if balance is None or balance.free < ticket.quantity:
            return None, []
    else:
        balance = state._balances.get_balance(base_asset)
        if balance is None or balance.free < base_total:
            return None, []

    order = state._create_order(normalized, data)
    return None, [order]
