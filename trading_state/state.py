"""
TradingState — the aggregate root of the trading_state library.

Design principles:
- Be pure. No strategies. No diff loops. The state never schedules,
  retries, or polls.
- Be passive. Every state change is the consequence of a caller-driven
  write (set_price, set_symbol, set_notional_limit, set_balances,
  set_cash_flow, create_order, update_order, cancel_order,
  import_order). The state never reaches out to an exchange.
- Be sync. All methods are synchronous; no callbacks-by-default; events
  are diagnostic, not control-flow.
- Be terminologically aligned with professional trading
  (exposure / notional limit / ticket / fill / settled / unsettled).
- No defaults on internal computation; callers must opt in or out of
  every component explicitly. The only sanctioned-with-default args
  in this surface are caller-metadata bags (Order.data,
  create_order's optional `data=`).
- Fail-fast on incomplete setup (missing symbol, missing notional
  limit, malformed allocation weights); best-effort on per-bucket
  business outcomes (filter rejection, insufficient balance,
  aggregate notional cap reached).
- Cross-currency allocation weights are a per-call argument on
  `create_order`, not a global state field. Weights computation
  (book depth, stablecoin balances, basis, inventory skew) is the
  caller's responsibility.

Public flow (caller side):
    state = TradingState(config)
    state.set_symbol(...); state.set_price(...); state.set_notional_limit(...)
    state.set_balances([Balance(...)])

    # `create_order` is the sole entry point for creating Orders.
    # `allocate=None` -> single Order on ticket.symbol (no split).
    # `allocate=weights_vec` -> split across alt account currencies
    # using the caller's weights (sized against config.alt_account_currencies).
    exc, orders = state.create_order(canonical_ticket, allocate=None, data={...})
    for order in orders:
        state.update_order(order, status=..., updated_at=..., ...)

    # Recovery: re-attach exchange-known orders not in state.
    exc, order = state.import_order(decoded_order)

    # Read paths
    state.exposure(asset, include_unsettled_inflow=..., include_unsettled_outflow=...)
    state.unsettled(asset)
    state.query_orders(...)
    state.get_open_orders()
"""
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
)
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from .enums import (
    MarketQuantityType,
    OrderStatus,
    TradingStateEvent,
)
from .symbol import (
    Symbol,
    SymbolManager,
)
from .balance import (
    Balance,
    BalanceManager,
)
from .pnl import (
    PerformanceTracker,
    CashFlow,
    PerformanceSnapshot,
)
from .order import (
    Order,
    OrderUpdatedType,
    OrderManager,
)
from .order_ticket import (
    OrderTicket,
    LimitOrderTicket,
    MarketOrderTicket,
    StopLossOrderTicket,
    StopLossLimitOrderTicket,
)
from .allocate import (
    single_create,
    single_create_quote,
    split_allocate,
    split_allocate_quote,
)
from .common import (
    DECIMAL_ZERO,
    EventEmitter,
    ValueOrException,
)
from .config import TradingConfig
from .exceptions import (
    AccountAssetHasNoExposureError,
    DuplicateOrderIdError,
    InvalidAllocationWeightsError,
    InvalidOrderForImportError,
    SymbolNotDefinedError,
    SymbolPriceNotReadyError,
)
from .exposure import Exposure
from .reconciliation import (
    ReconciliationManager,
    UnsettledFlow,
    current_impact_on_asset,
)


StaleKind = Literal[
    'balance_time_regress',
    'order_status_regress',
    'order_filled_regress',
    'order_time_regress',
]


@dataclass(frozen=True, slots=True)
class StaleUpdate:
    """
    Payload of TradingStateEvent.STALE_UPDATE.

    Emitted when state silently drops an update that would otherwise
    regress a monotonic invariant (balance.time / order status / order
    filled_quantity / order updated_at). Caller may subscribe for
    diagnostics; the event itself does not change state behaviour.

    Fields:
        kind            one of the StaleKind literals.
        asset           non-None for balance-side stale; None for order-side.
        order           non-None for order-side stale; None for balance-side.
        incoming_value  the value that was rejected.
        current_value   the value the state currently has and kept.
    """
    kind: StaleKind
    asset: Optional[str]
    order: Optional[Order]
    incoming_value: Any
    current_value: Any


AllocationWeights = Tuple[Decimal, ...]


class TradingState(EventEmitter[TradingStateEvent]):
    """State Phase III

    - support base asset limit exposure between 0 and 1
    - support multiple base assets
    - support multiple quote assets
    - support reconciliation between order updates and balance updates;
      exposure callers choose explicitly which "unsettled" components
      they want to include

    Convention:
    - For a certain base asset, its related tickets should have the
      same direction.
    """

    _config: TradingConfig
    _symbols: SymbolManager

    _balances: BalanceManager

    _orders: OrderManager
    _recon: ReconciliationManager

    def __init__(
        self,
        config: TradingConfig, /,
    ) -> None:
        super().__init__()

        self._config = config
        self._symbols = SymbolManager(config)
        self._balances = BalanceManager(config, self._symbols)

        self._orders = OrderManager(
            config.max_order_history_size,
            self._symbols,
        )

        self._recon = ReconciliationManager()

        self._perf = PerformanceTracker(
            config,
            self._symbols,
            self._balances,
            self._on_performance_snapshot_recorded,
        )

    @property
    def config(self) -> TradingConfig:
        return self._config

    # Public methods
    # ------------------------------------------------------------------------

    def set_price(
        self,
        symbol_name: str,
        price: Decimal,
    ) -> bool:
        """
        Set the price of a symbol. Returns True iff the price changed.
        """
        updated = self._symbols.set_price(symbol_name, price)

        self._check_balance_cash_flow(symbol_name)

        if updated:
            self.emit(TradingStateEvent.PRICE_UPDATED, symbol_name, price)

        return updated

    def set_symbol(
        self,
        symbol: Symbol, /,
    ) -> None:
        """Set (add or update) the symbol info for a symbol.

        Registering a new symbol can shorten a cached valuation path
        for some other asset, or unlock readiness for an asset whose
        prior `check_asset_ready` failed because its valuation path
        was incomplete. Both caches are flushed here — the cost is
        O(asset_count) and `set_symbol` is typically session-startup
        rather than a hot path, so this is cheap in practice.
        """
        if self._symbols.set_symbol(symbol):
            self._symbols.invalidate_paths()
            self._balances.invalidate_readiness()
            self.emit(TradingStateEvent.SYMBOL_ADDED, symbol)

    def set_notional_limit(self, asset: str, limit: Decimal) -> None:
        """
        Set the notional limit for a non-account asset. Mandatory:
        every asset that the caller intends to take exposure on must
        have a positive notional_limit configured before that asset's
        `exposure(...)` query (and the BUY-side `allocate(...)` gate)
        will succeed.

        For example, if::

            state.set_notional_limit('BTC', Decimal('35000'))

        - current BTC price: $7000
        - quote balance (USDT): $70000

        Then the trader can only buy 5 BTC, although the balance would
        otherwise allow 10.

        Use `Decimal('Infinity')` to declare "no effective cap" while
        still satisfying the always-set invariant — useful for
        backtests or unbounded strategies. `set_notional_limit` will
        raise on None, on a non-Decimal value, or on a value <= 0.
        """
        self._balances.set_notional_limit(asset, limit)

    def freeze(self, *args, **kwargs) -> None:
        """
        Freeze a certain quantity of an asset. The frozen quantity is
        excluded from exposure and from balance available to spend.
        """
        self._balances.freeze(*args, **kwargs)

    def set_balances(
        self,
        new: Iterable[Balance], /,
        *,
        delta: bool = False,
    ) -> None:
        """
        Update balances from an authoritative source (typically a
        decoded exchange snapshot or WS event).

        Stale data (balance.time strictly earlier than the currently
        known balance.time for the same asset) is silently dropped and
        a `STALE_UPDATE` event is emitted for diagnostic observers.

        Args:
            new: Iterable of Balance instances. Each Balance MUST carry
                a non-None `time`.
            delta: When True and a prior balance exists, the incoming
                free/locked are added to the existing values rather
                than replacing them. Delta mode skips the stale-time
                check by design — deltas are not absolute snapshots.
        """
        for balance in new:
            self._set_balance(balance, delta=delta)

    def set_cash_flow(
        self,
        cash_flow: CashFlow, /,
    ) -> None:
        """Handle external cashflow update (deposit / withdrawal)."""
        self._perf.set_cash_flow(cash_flow)

    def get_account_value(self) -> Decimal:
        """Get the value of the account in the account currency."""
        return self._balances.get_account_value(False)

    def get_price(
        self,
        symbol_name: str, /,
    ) -> Optional[Decimal]:
        return self._symbols.get_price(symbol_name)

    def get_symbol(
        self,
        symbol_name: str, /,
    ) -> Optional[Symbol]:
        """
        Retrieve a previously-registered Symbol by name. Returns None
        when the symbol has not been added via `set_symbol`.

        Callers build OrderTickets directly and a ticket's `symbol`
        field is a Symbol instance, so a public way to fetch one back
        out of state is part of the supported API surface (rather than
        reaching into `state._symbols`).
        """
        return self._symbols.get_symbol(symbol_name)

    def support_symbol(self, symbol_name: str, /) -> bool:
        return self._symbols.has_symbol(symbol_name)

    def exposure(
        self,
        asset: str, /,
        *,
        include_unsettled_inflow: bool,
        include_unsettled_outflow: bool,
    ) -> ValueOrException[Exposure]:
        """
        Snapshot of `asset`'s exposure under the chosen unsettled
        policy. Returns an `Exposure` value object — see
        `trading_state.exposure.Exposure` for the four atoms it stores
        and the derived quantities it exposes via @property.

        Effective holding folded into `Exposure.holding`::

            holding = balance.free + balance.locked
                    - frozen(asset)
                    + (unsettled.inflow  if include_unsettled_inflow  else 0)
                    - (unsettled.outflow if include_unsettled_outflow else 0)

        Account currencies are the unit of measurement, not a position;
        calling `exposure(...)` with an account currency yields
        `AccountAssetHasNoExposureError`.

        Args:
            asset: asset name.
            include_unsettled_inflow: required; whether unsettled
                inflows (BUY fills not yet reflected in a balance
                snapshot) count toward the holding.
            include_unsettled_outflow: required; symmetric for
                outflows.

        Returns:
            (exception, None) on:
              - AccountAssetHasNoExposureError when `asset` is an
                account currency.
              - AssetNotDefinedError / NotionalLimitNotSetError /
                ValuationNotAvailableError /
                ValuationPriceNotReadyError / BalanceNotReadyError
                via `check_asset_ready`.
            (None, Exposure) otherwise.
        """
        if self._symbols.is_account_asset(asset):
            return AccountAssetHasNoExposureError(asset), None

        exception = self._balances.check_asset_ready(asset)
        if exception is not None:
            return exception, None

        notional_limit = self._balances.get_notional_limit(asset)
        # check_asset_ready ensures notional_limit is set for any
        # non-account asset, so notional_limit cannot be None here.

        holding = self._balances.get_asset_total_balance(
            asset, DECIMAL_ZERO
        )
        unsettled = self._recon.unsettled_for(asset)
        if include_unsettled_inflow:
            holding += unsettled.inflow
        if include_unsettled_outflow:
            holding -= unsettled.outflow

        valuation_price = self._symbols.valuation_price(asset)

        return None, Exposure(
            asset=asset,
            holding=holding,
            valuation_price=valuation_price,
            notional_limit=notional_limit,
        )

    def unsettled(
        self,
        asset: str, /,
    ) -> ValueOrException[UnsettledFlow]:
        """
        Diagnostic view of the unsettled flow for `asset`. Not intended
        for trading decisions — those should go through `exposure(...)`
        with the explicit include_unsettled_* flags.
        """
        exception = self._balances.check_asset_ready(asset)
        if exception is not None:
            return exception, None
        return None, self._recon.unsettled_for(asset)

    def update_order(
        self,
        order: Order, /,
        *,
        status: Optional[OrderStatus],
        updated_at: Optional[datetime],
        id: Optional[str],
        filled_quantity: Optional[Decimal],
        quote_quantity: Optional[Decimal],
        commission_asset: Optional[str],
        commission_quantity: Optional[Decimal],
    ) -> None:
        """
        Apply an exchange-driven update to `order`.

        All keyword arguments are required; pass `None` for any field
        not being updated. This enforces explicit-intent at every call
        site (matches the library's no-default-values discipline).

        Stale data is silently dropped and a `STALE_UPDATE` event is
        emitted. "Stale" means any of:
        - status that goes backwards under OrderedEnum.lt;
        - filled_quantity smaller than the current filled_quantity;
        - updated_at strictly earlier than the current updated_at.

        This method never raises a business error. It does no
        cross-field validation against the original ticket (e.g.
        "filled_quantity exceeds ticket.quantity by more than dust");
        that responsibility lives at the protocol-adapter boundary.
        """
        if status is not None and status.lt(order.status):
            self._emit_stale(
                'order_status_regress',
                asset=None,
                order=order,
                incoming_value=status,
                current_value=order.status,
            )
            return

        if (
            filled_quantity is not None
            and filled_quantity < order.filled_quantity
        ):
            self._emit_stale(
                'order_filled_regress',
                asset=None,
                order=order,
                incoming_value=filled_quantity,
                current_value=order.filled_quantity,
            )
            return

        if (
            updated_at is not None
            and order.updated_at is not None
            and updated_at < order.updated_at
        ):
            self._emit_stale(
                'order_time_regress',
                asset=None,
                order=order,
                incoming_value=updated_at,
                current_value=order.updated_at,
            )
            return

        order.update(
            self._symbols,
            status=status,
            updated_at=updated_at,
            id=id,
            filled_quantity=filled_quantity,
            quote_quantity=quote_quantity,
            commission_asset=commission_asset,
            commission_quantity=commission_quantity,
        )

    def cancel_order(self, order: Order, /) -> None:
        """
        Mark `order` as being cancelled (status -> CANCELLING).
        Idempotent: no-op once already at or past CANCELLING. The
        actual cancel request to the exchange is the caller's job;
        bring the CANCELLED confirmation back via `update_order`.
        """
        self._orders.cancel(order)

    def query_orders(
        self,
        descending: bool = False,
        limit: Optional[int] = None,
        **criteria,
    ) -> Iterator[Order]:
        """
        Query the history of orders that have been confirmed by the
        exchange (status >= CREATED). For orders still in INIT or
        SUBMITTING (locally created but not yet acked), the caller
        already holds a reference from `add_order`; those are not in
        the history.

        See OrderHistory.query for the criteria DSL — values can be
        scalars (exact match), callables `(value, key) -> bool`, or
        dicts (subset match on nested `ticket` / `data`).

        Usage::

            results = state.query_orders(
                status=OrderStatus.FILLED,
                created_at=lambda x, _: x.timestamp() > 1717171717,
            )

            results = state.query_orders(
                descending=True,
                limit=10,
                ticket={'side': OrderSide.BUY},
                data={'strategy': 'momentum'},
            )
        """
        return self._orders.history.query(
            descending=descending,
            limit=limit,
            **criteria,
        )

    def get_order_by_id(self, order_id: str, /) -> Optional[Order]:
        return self._orders.get_order_by_id(order_id)

    def get_open_orders(self) -> List[Order]:
        """
        List the locally-known orders that the exchange has
        acknowledged (status CREATED / PARTIALLY_FILLED / CANCELLING)
        and that have not yet reached a terminal status
        (FILLED / CANCELLED / REJECTED).

        Used by recovery callers to enumerate orders needing a status
        refresh after a WS gap — see README "Recovery" for the full
        cold-startup / reconnect / periodic-check flows.

        Does NOT include orders in INIT or SUBMITTING. Those are still
        caller-side in-flight: either just returned by `allocate` (the
        caller holds the reference and is about to drive the POST), or
        the POST is in flight and the exchange has not yet acked. Such
        orders never enter the order history either.
        """
        return [
            o for o in self._orders.open_orders
            if not o.status.lt(OrderStatus.CREATED)
        ]

    def import_order(
        self,
        order: Order, /,
    ) -> ValueOrException[Order]:
        """
        Inject an externally-constructed Order into state. Used for
        recovery (cold startup or mid-session reconnect): the caller
        decodes an exchange snapshot via
        `trading_state.binance.decode_order_snapshot(item, symbol=...)`
        and feeds the resulting Order in here.

        Fail-fast (returns `(exc, None)`):
          - `InvalidOrderForImportError` if `order.id is None` or
            `order.ticket is None`. The Order must carry the stable
            exchange-side identity used as the lookup key.
          - `DuplicateOrderIdError` if an order with the same id is
            already in state. Recovery callers must dispatch on
            `state.get_order_by_id(id) is None` and route already-
            known ids through `state.update_order(...)`.

        Accepts any status, including terminal (FILLED / CANCELLED /
        REJECTED). A terminal IMPORTED order is added to the history
        and the id index but NOT to `_open_orders`; this lets cold
        startup pull `/api/v3/allOrders` and reflect orders that
        opened and closed during an earlier downtime, for accounting
        completeness.

        Does NOT apply filter normalization or validation. The
        exchange already accepted this order, so a local-filter
        rejection here would be a false negative caused by stale
        symbol filter snapshots.

        The imported Order does NOT pass through the INIT or
        SUBMITTING stages — its status is whatever the snapshot says.
        Downstream subscribers that observe INIT will simply not see
        these orders; ORDER_CREATED is emitted instead so the "every
        Order in state was preceded by ORDER_CREATED" invariant holds.

        Subsequent `update_order` calls operate identically to natively
        created orders (stale-update silent drop, fill events, etc.).
        """
        if order.id is None:
            return InvalidOrderForImportError('order.id is None'), None
        if order.ticket is None:
            return (
                InvalidOrderForImportError('order.ticket is None'),
                None,
            )
        if self._orders.get_order_by_id(order.id) is not None:
            return DuplicateOrderIdError(order.id), None

        return None, self._attach_order(order)

    def create_order(
        self,
        ticket: OrderTicket, /,
        *,
        data: Optional[Dict[str, Any]] = None,
        allocate: Optional[AllocationWeights],
    ) -> ValueOrException[List[Order]]:
        """
        Construct one or more Orders from a caller-built ticket and
        register them with state. The sole public entry point for
        order creation.

        `allocate` is a required keyword:

        - `allocate=None` — no cross-currency split. The ticket is
          filtered, pre-flighted, and materialized into a single
          Order on `ticket.symbol` (no primary-symbol substitution).
          Result list has length 0 (pre-flight or filter failed
          best-effort) or 1.

        - `allocate=weights_vec` — cross-currency split using the
          caller-supplied weights. Aligned with
          `config.alt_account_currencies`; the primary account
          currency is given implicit weight 1. Each surviving alt
          bucket produces an Order on the corresponding alt symbol.
          Result list has length 0 .. len(account_currencies); order
          matches declaration order, skipped buckets omitted.

        Weights are intentionally not a state field. Real allocation
        decisions depend on live book depth, stablecoin balances,
        basis, and inventory skew — all caller-side inputs.

        Fail-fast (returns `(exc, None)`):
          - `SymbolNotDefinedError` if `ticket.symbol` is not
            registered.
          - `InvalidAllocationWeightsError` if `allocate` is not None
            and the vector's length does not match
            `config.alt_account_currencies` or contains a negative
            weight.
          - `SymbolPriceNotReadyError` for a trailing-delta-only stop
            with no `set_price` on the symbol.
          - Whatever `exposure(...)` returns for the ticket's base
            asset on a BUY when those errors signal an incomplete
            setup (e.g. `NotionalLimitNotSetError`).

        Best-effort (returns `(None, [orders])` — list may be shorter
        than expected, or empty):
          - Filter rejection on any bucket (or on the single ticket
            when `allocate=None`) skips that bucket / returns `[]`.
          - Insufficient free balance skips the bucket / returns `[]`.
          - Aggregate notional exceeding the asset's `notional_limit`
            returns `(None, [])`.
        """
        if not self._symbols.has_symbol(ticket.symbol.name):
            return SymbolNotDefinedError(ticket.symbol.name), None

        if allocate is not None:
            exc = self._validate_allocation_weights(allocate)
            if exc is not None:
                return exc, None

        # MARKET(QUOTE) does not need a separate reference price —
        # split / single flows read `ticket.estimated_price` directly
        # to handle the quote<->base conversion.
        if (
            isinstance(ticket, MarketOrderTicket)
            and ticket.quantity_type is MarketQuantityType.QUOTE
        ):
            if allocate is None:
                return single_create_quote(self, ticket, data=data)
            return split_allocate_quote(
                self, ticket, weights_vec=allocate, data=data,
            )

        ref_or_exc = self._resolve_reference_price(ticket)
        if isinstance(ref_or_exc, Exception):
            return ref_or_exc, None
        reference_price = ref_or_exc

        if allocate is None:
            return single_create(
                self, ticket,
                reference_price=reference_price,
                data=data,
            )
        return split_allocate(
            self, ticket,
            weights_vec=allocate,
            reference_price=reference_price,
            data=data,
        )

    def record(self, *args, **kwargs) -> PerformanceSnapshot:
        """Record current performance snapshot."""
        return self._perf.record(*args, **kwargs)

    def performance(
        self,
        descending: bool = False,
    ) -> Iterator[PerformanceSnapshot]:
        return self._perf.iterator(descending)

    # End of public methods ---------------------------------------------

    # Order construction & allocation internals -------------------------

    def _validate_allocation_weights(
        self,
        weights: AllocationWeights,
    ) -> Optional[Exception]:
        """
        Structural validation for the caller's `allocate=` vector.
        Length must match `config.alt_account_currencies`; every
        weight must be non-negative. Returns None on success or an
        `InvalidAllocationWeightsError` describing the first
        violation found.
        """
        expected_len = len(self._config.alt_account_currencies)
        if len(weights) != expected_len:
            return InvalidAllocationWeightsError(
                f'expected {expected_len} weights '
                f'(one per alt_account_currencies entry), '
                f'got {len(weights)}'
            )
        for weight in weights:
            if weight < DECIMAL_ZERO:
                return InvalidAllocationWeightsError(
                    f'weight must be >= 0, got {weight}'
                )
        return None

    def _resolve_reference_price(
        self,
        ticket: OrderTicket,
    ) -> Any:
        """
        Per-ticket-type reference price for the split / single flows
        (everything except `MarketOrderTicket(QUOTE)`, which handles
        its own base/quote conversion via `ticket.estimated_price`).

        Returns either a `Decimal` or an `Exception` (the latter for
        the trailing-delta-only stop case where the symbol has no
        registered price).
        """
        if isinstance(ticket, LimitOrderTicket):
            return ticket.price
        if isinstance(ticket, MarketOrderTicket):
            return ticket.estimated_price
        if isinstance(ticket, StopLossLimitOrderTicket):
            return ticket.price
        if isinstance(ticket, StopLossOrderTicket):
            ref = ticket.stop_price
            if ref is None:
                ref = self.get_price(ticket.symbol.name)
                if ref is None:
                    return SymbolPriceNotReadyError(ticket.symbol.name)
            return ref
        # Unknown ticket type: no reference price available.
        return SymbolPriceNotReadyError(ticket.symbol.name)

    def _create_order(
        self,
        normalized_ticket: OrderTicket,
        data: Optional[Dict[str, Any]],
    ) -> Order:
        """
        Construct a fresh Order at INIT, wire it into state's
        lifecycle, and emit ORDER_CREATED. Used by `allocate` for
        every sub-ticket that survives filter normalization +
        pre-flight.
        """
        return self._attach_order(
            Order(ticket=normalized_ticket, data=data)
        )

    def _attach_order(self, order: Order) -> Order:
        """
        Wire `order` into state: status / fill listeners, registration
        with the order manager and the reconciliation manager, and the
        ORDER_CREATED emission. Shared by allocate's `_create_order`
        and the recovery-path `import_order`.
        """
        order.on(
            OrderUpdatedType.STATUS_UPDATED,
            self._on_order_status_updated,
        )
        order.on(
            OrderUpdatedType.FILLED_QUANTITY_UPDATED,
            self._on_order_filled_quantity_updated,
        )
        self._orders.add(order)
        self._recon.register(order)
        self.emit(TradingStateEvent.ORDER_CREATED, order)
        return order

    def _set_balance(
        self,
        balance: Balance,
        *,
        delta: bool,
    ) -> None:
        asset = balance.asset
        existing = self._balances.get_balance(asset)

        # Stale-time guard only applies to absolute snapshots. Delta
        # writes are increments, not authoritative snapshots, and have
        # no meaningful time ordering against existing absolute state.
        if (
            not delta
            and existing is not None
            and balance.time < existing.time
        ):
            self._emit_stale(
                'balance_time_regress',
                asset=asset,
                order=None,
                incoming_value=balance.time,
                current_value=existing.time,
            )
            return

        self._balances.set_balance(balance, delta=delta)

        self._recon.on_balance_set(asset, balance.time)

        self._purge_fully_settled_completed_orders()

    def _purge_fully_settled_completed_orders(self) -> None:
        """
        After a balance update, drop tracking for any order that has
        already reached a terminal status AND whose current impact on
        every involved asset matches the settled value. This keeps the
        recon dict bounded without losing fidelity: while a completed
        order still has unsettled fills, it stays tracked so
        `unsettled(...)` continues to reflect them.
        """
        # Snapshot first to avoid mutating during iteration.
        to_check = [
            o for o in list(self._recon._orders)
            if o.status.completed()
        ]
        for order in to_check:
            if self._is_fully_settled(order):
                self._recon.purge(order)

    def _is_fully_settled(self, order: Order) -> bool:
        symbol = order.ticket.symbol
        assets: Set[str] = {symbol.base_asset, symbol.quote_asset}
        if order.commission_asset is not None:
            assets.add(order.commission_asset)
        for asset in assets:
            settled = self._recon._settled.get(order, {}).get(
                asset, DECIMAL_ZERO
            )
            current = current_impact_on_asset(order, asset)
            if settled != current:
                return False
        return True

    def _emit_stale(
        self,
        kind: StaleKind,
        *,
        asset: Optional[str],
        order: Optional[Order],
        incoming_value: Any,
        current_value: Any,
    ) -> None:
        self.emit(
            TradingStateEvent.STALE_UPDATE,
            StaleUpdate(
                kind=kind,
                asset=asset,
                order=order,
                incoming_value=incoming_value,
                current_value=current_value,
            ),
        )

    def _on_order_status_updated(
        self,
        order: Order,
        status: OrderStatus,
    ) -> None:
        if status.completed():
            self._perf.track_order(order)

        # So that the caller of trading state can also listen to the
        # changes of order status.
        self.emit(TradingStateEvent.ORDER_STATUS_UPDATED, order, status)

    def _on_order_filled_quantity_updated(
        self,
        order: Order,
        filled_quantity: Decimal,
    ) -> None:
        self.emit(
            TradingStateEvent.ORDER_FILLED_QUANTITY_UPDATED,
            order,
            filled_quantity,
        )

    def _on_performance_snapshot_recorded(
        self,
        snapshot: PerformanceSnapshot,
    ) -> None:
        self.emit(
            TradingStateEvent.PERFORMANCE_SNAPSHOT_RECORDED, snapshot
        )

    def _check_balance_cash_flow(self, symbol_name: str) -> None:
        not_ready_assets = self._balances.not_ready_assets
        assets = not_ready_assets.dependents(symbol_name)

        if assets is None:
            return

        for asset in list(assets):
            balance = self._balances.get_balance(asset)
            cf = CashFlow(
                asset=asset,
                quantity=balance.total,
                time=datetime.now(),
            )
            success = self._perf.set_cash_flow(cf)
            if success:
                not_ready_assets.clear(asset)
