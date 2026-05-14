"""
TradingState — the aggregate root of the trading_state library.

Design principles (kept stable across the post-execution-strategy
refactor):
- Be pure. No strategies. No diff loops. The state never schedules,
  retries, or polls.
- Be passive. Every state change is the consequence of a caller-driven
  write (set_price, set_symbol, set_notional_limit, set_balances,
  set_cash_flow, add_order, update_order, cancel_order, allocate,
  set_alt_currency_weights). The state never reaches out to an
  exchange.
- Be sync. All methods are synchronous; no callbacks-by-default; events
  are diagnostic, not control-flow.
- Be terminologically aligned with professional trading
  (exposure / notional limit / ticket / fill / settled / unsettled).
- No defaults on internal computation; callers must opt in or out of
  every component explicitly. The only sanctioned-with-default arg in
  this surface is `Order.data`, an opaque caller-metadata bag.

Public flow (caller side):
    state = TradingState(config)
    state.set_symbol(...); state.set_price(...); state.set_notional_limit(...)
    state.set_balances([Balance(...)])

    # Build a ticket externally and (optionally) split across alt
    # account currencies. allocate() returns filter-applied sub-tickets
    # and is a best-effort splitter that never raises a business error
    # — when there's nothing to split, the original ticket comes back.
    tickets = state.allocate(canonical_ticket)
    for t in tickets:
        exc, order = state.add_order(t, data={...})
        ...
        state.update_order(order, status=..., updated_at=..., ...)

    # Read paths
    state.exposure(asset, include_unsettled_inflow=..., include_unsettled_outflow=...)
    state.unsettled(asset)
    state.query_orders(...)
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
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal

from .enums import (
    MarketQuantityType,
    OrderSide,
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
)
from .allocate import (
    AllocationResource,
    buy_allocate,
    sell_allocate,
)
from .common import (
    DECIMAL_INF,
    DECIMAL_ONE,
    DECIMAL_ZERO,
    EventEmitter,
    ValueOrException,
)
from .config import TradingConfig
from .exceptions import (
    AccountAssetHasNoExposureError,
    AllocationWeightsNotSetError,
    InsufficientFreeBalanceError,
    NotionalLimitExceededError,
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

    # No allocations for account currencies by default
    _alt_currency_weights: Optional[Tuple[AllocationWeights, AllocationWeights]] = None

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

    def set_alt_currency_weights(
        self,
        weights: Optional[
            Tuple[AllocationWeights, AllocationWeights]
        ], /,
    ) -> None:
        """
        Set the BUY/SELL weights of the alternative account currencies.

        Args:
            weights: A pair (buy_weights, sell_weights). Each weights
                tuple matches the order of `config.alt_account_currencies`.
                The primary account currency's weight is implicitly 1.

        Usage::

            state.set_alt_currency_weights((
                (Decimal('0.5'), Decimal('0.5')),  # BUY
                (Decimal('1'),   Decimal('0')),    # SELL
            ))

        Passing None removes the configuration; allocate() will then
        simply pass tickets through unchanged.
        """
        if weights is not None:
            self._check_allocation_weights(weights[0])
            self._check_allocation_weights(weights[1])

        self._alt_currency_weights = weights

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

    def add_order(
        self,
        ticket: OrderTicket, /,
        *,
        data: Optional[Dict[str, Any]] = None,
    ) -> ValueOrException[Order]:
        """
        Register a caller-built ticket as a new Order in INIT status.

        Behaviour:
        - Runs the symbol's filters in normalize mode via
          `ticket.apply_filters()`. The returned ticket may be a frozen
          copy with adjusted fields.
        - On filter rejection: returns `(exception, None)` and does NOT
          touch state. Filter rejections are not emitted as events;
          they come back through the return value so the caller cannot
          accidentally drop them.
        - On success: creates the Order, attaches state's listeners,
          registers it with the order manager and reconciliation
          manager, and returns `(None, Order)`. Caller controls the
          state machine transitions (INIT -> SUBMITTING -> CREATED ...)
          via subsequent `update_order` calls.

        Args:
            ticket: A constructed OrderTicket; its symbol must already
                have been added via `set_symbol`.
            data: Caller metadata. Copied defensively into the Order
                so a shared default `{}` cannot leak across orders.

        Returns:
            ValueOrException[Order]:
              (exception, None) on filter rejection.
              (None, order) on success.
        """
        exception, normalized = ticket.apply_filters()
        if exception is not None:
            return exception, None

        order = Order(ticket=normalized, data=data)

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

        return None, order

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

    def allocate(
        self,
        ticket: OrderTicket, /,
    ) -> ValueOrException[List[OrderTicket]]:
        """
        Pre-flight gate plus cross-currency split. Mandatory entry
        point before `add_order` — it both validates exposure / free
        balance against the configured notional cap and produces the
        filter-applied sub-tickets to feed to `add_order`.

        Returns:
            (None, [t1, ...])  — one or more filter-applied sub-tickets
                                 whose combined base quantity does not
                                 push exposure(base_asset) above its
                                 notional_limit (worst-case unsettled
                                 inflow=True, outflow=False).
            (exc, None) on:
              - AllocationWeightsNotSetError when
                `set_alt_currency_weights(...)` has not been called.
              - NotionalLimitExceededError when the BUY would push the
                base-asset exposure past its cap even before any
                allocation.
              - InsufficientFreeBalanceError when no weighted
                account-currency bucket can fund a BUY (either no
                symbol is registered or every bucket has zero free).
              - Whatever the asset's `exposure(...)` check returns
                (AssetNotDefinedError / NotionalLimitNotSetError /
                ValuationPriceNotReadyError / ...) for BUY orders.
              - The first filter exception when every candidate
                sub-ticket gets rejected by its symbol's filters.
              - ValueError when allocation yields no sub-ticket and no
                filter rejection was recorded (e.g. ticket.quantity is
                effectively zero).

        Stop-loss / take-profit families and MARKET orders with
        QUOTE-denominated quantity bypass the split math; allocate
        returns `(None, [ticket])` so the caller still has a single
        normalized entry to feed to `add_order`.
        """
        if self._alt_currency_weights is None:
            return AllocationWeightsNotSetError(), None

        side = ticket.side

        if isinstance(ticket, LimitOrderTicket):
            reference_price = ticket.price
        elif isinstance(ticket, MarketOrderTicket):
            if ticket.quantity_type is MarketQuantityType.QUOTE:
                # Quote-denominated MARKET orders are not allocatable
                # in base units; pass through as a single sub-ticket.
                return None, [ticket]
            reference_price = ticket.estimated_price
        else:
            # Stop-loss / take-profit variants are out of scope of the
            # account-currency split logic; pass through.
            return None, [ticket]

        base_asset = ticket.symbol.base_asset

        if side is OrderSide.BUY:
            # Worst-case exposure pre-check: assume every unsettled
            # inflow lands, no unsettled outflow does.
            exc, exposure_now = self.exposure(
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
                return NotionalLimitExceededError(
                    asset=base_asset,
                    attempted_notional=projected,
                    notional_limit=exposure_now.notional_limit,
                ), None

        weights_vec = self._alt_currency_weights[
            0 if side is OrderSide.BUY else 1
        ]
        # alt weights first (in alt_account_currencies order), primary
        # at the tail with implicit weight 1 — matches
        # config.account_currencies. The primary bucket is always
        # weight 1 (>0); the loop filters per-bucket weights itself.
        full_weights = (*weights_vec, DECIMAL_ONE)

        resources: List[AllocationResource] = []
        for i, acct_cur in enumerate(self._config.account_currencies):
            weight = full_weights[i]
            if weight <= DECIMAL_ZERO:
                continue
            symbol_name = self._config.get_symbol_name(base_asset, acct_cur)
            alt_symbol = self._symbols.get_symbol(symbol_name)
            if alt_symbol is None:
                continue
            if side is OrderSide.BUY:
                balance = self._balances.get_balance(acct_cur)
                if balance is None or balance.free <= DECIMAL_ZERO:
                    continue
                free = balance.free
            else:
                free = DECIMAL_INF
            resources.append(AllocationResource(alt_symbol, free, weight))

        if not resources:
            if side is OrderSide.BUY:
                return InsufficientFreeBalanceError(base_asset), None
            # SELL with no resources: every weighted bucket is missing
            # a registered symbol. Fall back to a single passthrough
            # rather than fabricating an "insufficient balance" error.
            return None, [ticket]

        output: List[OrderTicket] = []
        last_filter_exc: List[Exception] = []

        def assign(symbol: Symbol, quantity: Decimal) -> Decimal:
            if quantity <= DECIMAL_ZERO:
                return DECIMAL_ZERO
            candidate = replace(
                ticket, symbol=symbol, quantity=quantity
            )
            exc, normalized = candidate.apply_filters()
            if exc is not None:
                # Filter rejected — capture for surfacing if every
                # bucket fails, and roll the candidate quantity
                # forward via the compensate chain.
                last_filter_exc.append(exc)
                return quantity
            output.append(normalized)
            return quantity - normalized.quantity

        if side is OrderSide.BUY:
            buy_allocate(
                resources, ticket.quantity, reference_price, assign
            )
        else:
            sell_allocate(resources, ticket.quantity, assign)

        if not output:
            if last_filter_exc:
                return last_filter_exc[0], None
            return (
                ValueError(
                    'allocate produced no sub-ticket; the input '
                    'quantity may be zero or every candidate fell '
                    "below the symbol's minimum"
                ),
                None,
            )

        return None, output

    def record(self, *args, **kwargs) -> PerformanceSnapshot:
        """Record current performance snapshot."""
        return self._perf.record(*args, **kwargs)

    def performance(
        self,
        descending: bool = False,
    ) -> Iterator[PerformanceSnapshot]:
        return self._perf.iterator(descending)

    # End of public methods ---------------------------------------------

    def _check_allocation_weights(
        self,
        weights: AllocationWeights,
    ) -> None:
        for weight in weights:
            if weight < DECIMAL_ZERO:
                raise ValueError(
                    'The allocation weight must not less than 0'
                )

        if len(weights) != len(self._config.alt_account_currencies):
            raise ValueError(
                'The number of allocation weights must be equal to '
                'the number of alternative account currencies'
            )

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
