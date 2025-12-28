from typing import (
    Dict,
    Iterable,
    Tuple,
    Set,
    Optional,
    Iterator,
    List
)
from decimal import Decimal
from datetime import datetime

from .enums import (
    OrderStatus,
    OrderSide,
    TimeInForce,
    MarketQuantityType,
    TradingStateEvent,
    PositionTargetStatus
)
from .symbol import (
    Symbol,
    SymbolManager
)
from .balance import (
    Balance,
    BalanceManager,
)
from .pnl import (
    PerformanceAnalyzer,
    CashFlow,
    PerformanceSnapshot
)
from .order import (
    Order,
    OrderUpdatedType,
    OrderManager
)
from .order_ticket import (
    LimitOrderTicket,
    MarketOrderTicket
)
from .target import (
    PositionTarget,
    PositionTargetMetaData
)
from .allocate import (
    AllocationResource,
    buy_allocate,
    sell_allocate
)
from .common import (
    DECIMAL_ZERO,
    DECIMAL_ONE,
    DECIMAL_INF,
    ValueOrException,
    FactoryDict,
    EventEmitter
)
from .config import TradingConfig


AllocationWeights = Tuple[Decimal, ...]


"""
Logical Principles:
- terminology aligned with professional trading
- be pure
- be passive, no triggers
- no strategies, strategies should be driven by external invocations
- suppose the initialization is done before using the state, including
  - setting up the symbols
  - setting up the balances
  - setting up the symbol prices
- do not handle position control, just proceed with position expectations
  - the purpose of a position change could be marked in position.data

Implementation Principles:
- be sync
- no checking for unexpected param types (according to typings)
  - but should check unexpected values
- no default parameters for all methods to avoid unexpected behavior
"""


class TradingState(EventEmitter[TradingStateEvent]):
    """State Phase II

    - support base asset limit exposure between 0 and 1
    - support multiple base assets
    - support multiple quote assets

    Convention:
    - For a certain base asset,
      its related tickets should has the same direction

    Design principle:
    - The expectation settings are the final state that the system try to achieve.
    """

    _config: TradingConfig
    _symbols: SymbolManager

    # asset -> balance
    _balances: BalanceManager

    # No allocations for account currencies by default
    _alt_currency_weights: Optional[List[Decimal]] = None

    # asset -> position expectation
    _expected: Dict[str, PositionTarget]
    # position target -> order
    _target_orders: FactoryDict[PositionTarget, Set[Order]]

    _orders: OrderManager

    def __init__(
        self,
        config: TradingConfig
    ) -> None:
        super().__init__()

        self._config = config
        self._symbols = SymbolManager(config)
        self._balances = BalanceManager(config, self._symbols)

        self._expected = {}
        self._target_orders = FactoryDict[
            PositionTarget, Set[Order]
        ](set[Order])

        self._orders = OrderManager(
            config.max_order_history_size,
            self._symbols
        )
        self._perf = PerformanceAnalyzer(
            config,
            self._symbols,
            self._balances
        )

    @property
    def config(self) -> TradingConfig:
        return self._config

    # Public methods
    # ------------------------------------------------------------------------

    def set_price(
        self,
        symbol_name: str,
        price: Decimal
    ) -> bool:
        """
        Set the price of a symbol

        Returns `True` if the price changes
        """

        updated = self._symbols.set_price(symbol_name, price)

        self._check_balance_cash_flow(symbol_name)

        if updated:
            self.emit(TradingStateEvent.PRICE_UPDATED, symbol_name, price)

        return updated

    def set_symbol(
        self,
        symbol: Symbol, /
    ) -> None:
        """
        Set (add or update) the symbol info for a symbol

        Args:
            symbol (Symbol): the symbol to set
        """

        if self._symbols.set_symbol(symbol):
            self.emit(TradingStateEvent.SYMBOL_ADDED, symbol)

    def set_notional_limit(self, *args, **kwargs) -> None:
        """
        Set the notional limit for a certain asset. Pay attention that, by design, it is mandatory to set the notional limit for an asset before trading with the trading state.

        The notional limit of an asset limits:
        - the maximum quantity of the **account_currency** asset the trader could **BUY** the asset,
        - no SELL.

        Args:
            asset (str): the asset to set the notional limit for
            limit (Decimal | None): the maximum quantity of the account currency the trader could BUY the asset. `None` means no notional limit. `None` is not suggested in production.

        For example, if::

            state.set_notional_limit('BTC', Decimal('35000'))

        - current BTC price: $7000
        - base asset balance (USDT): $70000

        Then, the trader could only buy 5 BTC,
        although the balance is enough to buy 10 BTC
        """

        self._balances.set_notional_limit(*args, **kwargs)

    def set_alt_currency_weights(
        self,
        weights: Optional[
            Tuple[AllocationWeights, AllocationWeights]
        ], /
    ) -> None:
        """
        Set the weights of the alternative account currencies to the account currency.

        This system will use a simple approach to attempt to achieve a dynamic equilibrium.

        Args:
            weights (Tuple[AllocationWeights, AllocationWeights]): the weights of the alternative account currencies to the account currency according to the order of `config.alt_account_currencies` for BUY and SELL respectively, each of which should not less than 0.

        Usage::

            state.set_alt_currency_weights((
                (Decimal('0.5'), Decimal('0.5')),
                (Decimal('1'), Decimal('0'))
            ))
            # We want to consume the 2nd alt account currency
        """

        if weights is not None:
            self._check_allocation_weights(weights[0])
            self._check_allocation_weights(weights[1])

        self._alt_currency_weights = weights

    def freeze(self, *args, **kwargs) -> None:
        """
        Freeze a certain quantity of an asset. The frozen quantity will be excluded from the calculation of
        - notional limit
        - exposure
        - balance
        """

        self._balances.freeze(*args, **kwargs)

    def set_balances(
        self,
        new: Iterable[Balance], /,
        *args, **kwargs
    ) -> None:
        """
        Update user balances, including normal assets and quote assets

        Args:
            new (Iterable[Balance]): the new balances to set
            delta (bool = False): whether to update the balances as a delta, i.e. the new balances are relative to the current balances

        Usage::

            state.set_balances([
                Balance('BTC', Decimal('8'), Decimal('0'))
            ])
        """

        for balance in new:
            self._set_balance(balance, *args, **kwargs)

    def set_cash_flow(
        self,
        cash_flow: CashFlow, /
    ) -> None:
        """Handle external cashflow update
        """

        self._perf.set_cash_flow(cash_flow)

    def get_account_value(self) -> Decimal:
        """
        Get the value of the account in the account currency
        """

        return self._balances.get_account_value(False)

    def get_price(
        self,
        symbol_name: str, /
    ) -> Decimal | None:
        """
        Get the price of a symbol
        """

        return self._symbols.get_price(symbol_name)

    def support_symbol(self, symbol_name: str, /) -> bool:
        """
        Check whether the symbol is supported
        """

        return self._symbols.has_symbol(symbol_name)

    def exposure(
        self,
        asset: str, /
    ) -> ValueOrException[Decimal]:
        """
        Get the current expected limit exposure or the calculated limit exposure of an asset

        Args:
            asset (str): the asset name to get the limit exposure for

        Returns:
            - Tuple[Exception, None]: the exception if the asset is not ready
            - Tuple[None, Decimal]: the limit exposure of the asset
        """

        exception = self._balances.check_asset_ready(asset)
        if exception is not None:
            return exception, None

        target = self._expected.get(asset)

        if target is not None:
            return None, target.exposure

        return None, self._get_asset_exposure(asset)

    def cancel_order(self, order: Order, /) -> None:
        """
        Cancel an order from the trading state, and trigger the cancellation the next tick

        The method should have no side effects if called multiple times
        """

        self._orders.cancel(order)

        asset = order.ticket.symbol.base_asset

        current_target = self._expected.get(asset)

        if current_target is None:
            return

        if current_target is order.target:
            target_orders = self._target_orders[current_target]
            target_orders.discard(order)

            if not target_orders:
                # No more orders for the target,
                # remove the target
                self._expected.pop(asset, None)
            else:
                self._update_target_exposure(current_target)

    def query_orders(
        self,
        descending: bool = False,
        limit: Optional[int] = None,
        **criteria
    ) -> Iterator[Order]:
        """
        Query the history orders by the given criteria

        Args:
            descending (bool = False): Whether to query the history in descending order, ie. the most recent orders first
            limit (Optional[int]): Maximum number of orders to return. `None` means no limit.
            **criteria: Criteria to match the orders

        Returns:
            Iterator[Order]: the matched orders

        Usage::

            results = state.query_orders(
                status=OrderStatus.FILLED,
                # Callable matcher
                created_at=lambda x: x.timestamp() > 1717171717,
            )

            results = state.query_orders(
                descending=True,
                limit=10,
                # Dict matcher (for 'ticket' and 'target' only)
                ticket={
                    'side': OrderSide.BUY
                }
            )
        """

        return self._orders.history.query(
            descending=descending,
            limit=limit,
            **criteria
        )

    def get_order_by_id(self, order_id: str, /) -> Optional[Order]:
        return self._orders.get_order_by_id(order_id)

    def expect(
        self,
        symbol_name: str, /,
        exposure: Decimal,
        price: Decimal,
        use_market_order: bool,
        data: PositionTargetMetaData = {}
    ) -> ValueOrException[bool]:
        """
        Update expectation, returns whether it is successfully updated

        Args:
            symbol_name (str): the name of the symbol to trade with
            exposure (Decimal): the limit exposure to expect, should be between `0.0` and `1.0`. The exposure refers to the current holding of the base asset as a proportion of its maximum allowed notional limit
            use_market_order (bool = False): whether to execute the expectation use_market_orderly, that it will generate a market order
            price (Decimal): the price to expect. For market order, it should be the estimated average price for the expected position target.
            data (Dict[str, Any] = {}): the data attached to the expectation, which will also attached to the created order, order history for future reference.

        Returns:
            Tuple[Optional[Exception], bool]:
            - the reason exception if the expectation is not successfully updated
            - whether the expectation is successfully updated

        Usage::

            # to all-in BTC within the notional limit
            state.expect('BTCUSDT', 1., ...)
        """

        exception = self._balances.check_symbol_ready(symbol_name)

        if exception is not None:
            return exception, None

        symbol = self._symbols.get_symbol(symbol_name)
        asset = symbol.base_asset

        # Normalize the exposure to be between 0 and 1
        exposure = max(DECIMAL_ZERO, min(exposure, DECIMAL_ONE))

        open_target = self._expected.get(asset)

        if open_target is not None:
            if (
                open_target.exposure == exposure
                and open_target.price == price
                and open_target.use_market_order == use_market_order
            ):
                # If the target is the same, no need to update
                # We treat it as a success
                return None, False

        calculated_exposure = self._get_asset_exposure(asset)

        if calculated_exposure == exposure:
            # If the target is the same, no need to update
            # We treat it as a success
            return None, False

        self._expected[asset] = PositionTarget(
            symbol=symbol,
            exposure=exposure,
            use_market_order=use_market_order,
            price=price,
            data=data
        )

        self.emit(TradingStateEvent.POSITION_TARGET_UPDATED)

        return None, True

    def get_orders(self) -> Tuple[
        Set[Order],
        Set[Order]
    ]:
        """
        Diff the orders, and get all unsubmitted orders, by calling this method
        - Orders of OrderStatus.INIT -> OrderStatus.SUBMITTING
        - Orders of OrderStatus.ABOUT_TO_CANCEL -> OrderStatus.CANCELLING

        Returns
            tuple:
            - a set of available orders
            - a set of orders to cancel
        """

        self._diff()

        orders, orders_to_cancel = self._orders.get_orders()

        for order in orders:
            self.update_order(order, status=OrderStatus.SUBMITTING)

        for order in orders_to_cancel:
            self.update_order(order, status=OrderStatus.CANCELLING)

        return orders, orders_to_cancel

    def update_order(self, order: Order, /, **kwargs) -> None:
        """
        Update the order

        Args:
            status (OrderStatus = None): The new status of the order
            created_at (datetime = None): The creation time of the order
            updated_at (datetime = None): The update time of the order
            filled_quantity (Decimal = None): The new filled base assert quantity of the order
            quote_quantity (Decimal = None): The cumulative quote asset transacted quantity of the order
            commission_asset (str = None): The commission asset name
            commission_quantity (Decimal = None): The cumulative quantity of the commission asset
            id (str = None): The client order id

        Usage::

            state.update_order(
                order,
                filled_quantity = Decimal('0.5'),
                quote_quantity = Decimal('1000')
            )
        """

        order.update(self._symbols, **kwargs)

    def record(self, *args, **kwargs) -> PerformanceSnapshot:
        """
        Record current performance snapshot

        Args:
            labels (dict = {}): List of labels to add to the snapshot
            time (datetime = None): Timestamp of the snapshot

        Returns:
            PerformanceSnapshot: The created performance snapshot
        """
        return self._perf.record(*args, **kwargs)

    def performance(
        self,
        descending: bool = False
    ) -> Iterator[PerformanceSnapshot]:
        """
        Returns an iterator of performance snapshots

        Args:
            descending (bool = False): Whether to iterate in descending order, ie. the most recent snapshots first

        Returns:
            Iterator[PerformanceSnapshot]
        """

        return self._perf.iterator(descending)

    # End of public methods ---------------------------------------------

    def _check_allocation_weights(self, weights: AllocationWeights) -> None:
        for weight in weights:
            if weight < DECIMAL_ZERO:
                raise ValueError(
                    'The allocation weight must not less than 0')

        if len(weights) != len(self._config.alt_account_currencies):
            raise ValueError(
                'The number of allocation weights must be equal to the number of alternative account currencies')

    def _set_balance(
        self,
        balance: Balance,
        *args, **kwargs
    ) -> None:
        """
        Set the balance of an asset
        """

        asset = balance.asset

        old_balance, balance = self._balances.set_balance(
            balance, *args, **kwargs
        )

        target = self._expected.get(asset)

        if old_balance is None:
            return

        if old_balance.total == balance.total:
            return

        if target is None or target.status.lt(PositionTargetStatus.ACHIEVED):
            # There is no expectation or
            # the expectation is still being achieved,
            # we do not need to recalculate the target

            # And we actually do not know, whether the balance change is
            # caused by the position target or not,
            # so we just keep it.
            return

        calculated_exposure = self._get_asset_exposure(asset)
        if calculated_exposure == target.exposure:
            # The exposure is the same, no need to update
            return

        # We need to remove the expectation,
        # so that self.exposure() will return the recalculated exposure
        self._expected.pop(asset, None)

    def _get_asset_exposure(self, asset: str) -> Optional[Decimal]:
        """
        Get the calculated limit exposure of an asset

        Should only be called after `asset_ready`

        Returns:
            Decimal: the calculated limit exposure of the asset
        """

        balance = self._get_asset_expected_balance(asset)
        price = self._symbols.valuation_price(asset)
        limit = self._balances.get_notional_limit(asset)

        if limit is None:
            # If no notional limit, then the exposure is not calculable
            return None

        return balance * price / limit

    def _get_asset_expected_balance(self, asset: str) -> Decimal:
        extra = DECIMAL_ZERO

        for order in self._orders.get_orders_by_base_asset(asset):
            # For BUY orders, the balance will increase
            if order.ticket.side is OrderSide.BUY:
                extra += order.ticket.quantity - order.filled_quantity

        return self._balances.get_asset_total_balance(asset, extra)

    def _diff(self) -> None:
        """
        Diff the position expectations
        """

        for asset, target in list(self._expected.items()):
            self._create_order_from_position_target(target)

            if target.status is PositionTargetStatus.INIT:
                # If there are orders created
                if self._target_orders[target]:
                    target.status = PositionTargetStatus.ALLOCATED

                    # Update the exposure to the actual exposure
                    self._update_target_exposure(target)
                else:
                    # We'd better remove the expectation to avoid
                    # unexpected behavior later on, such as
                    # - we can't meet the expectation, but after a long time,
                    #   the balance suddenly updated, and trigger an unexpected
                    #   new order creation
                    self._expected.pop(asset, None)

    def _update_target_exposure(self, target: PositionTarget) -> None:
        target.exposure = self._get_asset_exposure(
            target.symbol.base_asset
        )

    def _create_order_from_position_target(
        self,
        target: PositionTarget
    ) -> None:
        """
        Create a order from an asset position target.

        Actually it is always called after `symbol_ready`
        because of `self.expect(...)`
        """

        if target.status is not PositionTargetStatus.INIT:
            return

        symbol = target.symbol

        existing_order = self._orders.get_order_by_symbol(symbol)

        # We only keep one order for a single symbol
        if existing_order is not None:
            self.cancel_order(existing_order)

        # Calculate the eventual valuation value of the asset
        # --------------------------------------------------------
        asset = symbol.base_asset
        balance = self._get_asset_expected_balance(asset)
        valuation_price = self._symbols.valuation_price(asset)
        value = balance * valuation_price

        limit = self._balances.get_notional_limit(asset)

        if limit is not None:
            value_delta = Decimal(str(target.exposure)) * limit - value
            quantity = value_delta / valuation_price

            if quantity.is_zero():
                # Usually, it won't be zero,
                # but the balance might changed after `.expect()`
                return

            side = OrderSide.BUY
            if quantity < DECIMAL_ZERO:
                side = OrderSide.SELL
                quantity = - quantity
        else:
            # If no notional limit, then all in or all out
            quantity = DECIMAL_INF
            side = (
                OrderSide.BUY
                if target.exposure > DECIMAL_ZERO
                else OrderSide.SELL
            )

        if (
            # No allocation weights
            self._alt_currency_weights is None
            # No alternative account currencies
            or not self._config.alt_account_currencies
            # Not an account currency, we do not need to do allocation,
            # Example: BTCBNB
            or symbol.quote_asset not in self._config.account_currencies
        ):
            self._check_balance_and_create_order(
                symbol, quantity, target, side
            )
            return

        weights = self._alt_currency_weights[
            0 if side is OrderSide.BUY else 1
        ]

        self._balance_account_currencies_and_create_orders(
            asset,
            quantity,
            (*weights, DECIMAL_ONE),
            target,
            side
        )

    def _balance_account_currencies_and_create_orders(
        self,
        # The base asset to trade
        asset: str,
        # The quantity of the base asset to trade
        quantity: Decimal,
        weights: AllocationWeights,
        target: PositionTarget,
        side: OrderSide
    ) -> None:
        resources = list[AllocationResource](
            AllocationResource(symbol, balance.free, weight)
            for i, quote_asset in enumerate(
                self._config.account_currencies
            )
            if (
                (
                    symbol := self._symbols.get_symbol(
                        self._config.get_symbol_name(asset, quote_asset)
                    )
                ) is not None
                and (
                    # For BUY: balance must be positive
                    (
                        balance := self._balances.get_balance(quote_asset)
                    ) is not None and balance.free > 0
                    # For SELL, allow the balance of a quote asset to be 0
                    or side is OrderSide.SELL

                # For both BUY and SELL, the weight must be positive
                ) and (weight := weights[i]) > 0
            )
        )

        if not resources:
            # No available account currencies
            return

        if len(resources) == 1:
            # Only one account currency available,
            # we do not need to do allocation
            self._check_balance_and_create_order(
                resources[0].symbol, quantity, target, side
            )
            return

        if side is OrderSide.BUY:
            buy_allocate(
                resources,
                quantity,
                target,
                self._create_order
            )
        else:
            sell_allocate(
                resources,
                quantity,
                target,
                self._create_order
            )

    def _check_balance_and_create_order(
        self,
        symbol: Symbol,
        quantity: Decimal,
        target: PositionTarget,
        side: OrderSide
    ) -> None:
        if side is OrderSide.BUY:
            quantity = min(
                quantity,
                # We need to check the available balance of the quote asset
                self._balances.get_balance(
                    symbol.quote_asset
                ).free / target.price
            )
        else:
            quantity = min(
                quantity,
                # We need to check the available balance of the base asset
                self._balances.get_balance(symbol.base_asset).free
            )

        self._create_order(symbol, quantity, target, side)

    def _create_order(
        self,
        symbol: Symbol,
        target_quantity: Decimal,
        target: PositionTarget,
        side: OrderSide
    ) -> Decimal:
        """
        Returns:
            Decimal: the remaining base quantity relative to the target quantity
        """

        price = target.price
        quote_quantity = target_quantity * price

        ticket = (
            MarketOrderTicket(
                symbol=symbol,
                side=side,
                quantity=quote_quantity,
                # Use quote quantity 'quoteOrderQty' for market order
                #   to avoid -2010 error as much as possible
                quantity_type=MarketQuantityType.QUOTE,
                estimated_price=price
            )
            if target.use_market_order
            else LimitOrderTicket(
                symbol=symbol,
                side=side,
                quantity=target_quantity,
                price=price,
                time_in_force=TimeInForce.GTC
            )
        )

        exception, _ = symbol.apply_filters(
            ticket,
            validate_only=False,
            **self._config.context
        )

        if exception is not None:
            self.emit(TradingStateEvent.TICKET_CREATE_FAILED, exception)
            # The quantity consuming task is not finished
            return target_quantity

        order = Order(
            ticket=ticket,
            target=target
        )

        order.on(
            OrderUpdatedType.STATUS_UPDATED,
            self._on_order_status_updated
        )

        order.on(
            OrderUpdatedType.FILLED_QUANTITY_UPDATED,
            self._on_order_filled_quantity_updated
        )

        self._orders.add(order)

        # Track the order for the position target
        self._target_orders[target].add(order)

        return (
            target_quantity - ticket.quantity / price
            if target.use_market_order
            else target_quantity - ticket.quantity
        )

    def _on_order_status_updated(
        self,
        order: Order,
        status: OrderStatus
    ) -> None:
        match status:
            case OrderStatus.REJECTED:
                self.cancel_order(order)

            case OrderStatus.CANCELLED:
                self.cancel_order(order)

            case OrderStatus.FILLED:
                target = order.target
                target_orders = self._target_orders[target]

                # All orders are filled
                if all(
                    order.status is OrderStatus.FILLED
                    for order in target_orders
                ):
                    target.status = PositionTargetStatus.ACHIEVED

        if status.completed():
            self._perf.track_order(order)

        # So that the caller of trading state
        # can also listen to the changes of order status
        self.emit(TradingStateEvent.ORDER_STATUS_UPDATED, order, status)

    def _on_order_filled_quantity_updated(
        self,
        order: Order,
        filled_quantity: Decimal
    ) -> None:
        self.emit(
            TradingStateEvent.ORDER_FILLED_QUANTITY_UPDATED,
            order,
            filled_quantity
        )

    def _check_balance_cash_flow(self, symbol_name: str) -> None:
        not_ready_assets = self._balances.not_ready_assets

        assets = not_ready_assets.dependents(symbol_name)

        if assets is None:
            return

        # `assets` might be modified during the iteration,
        # so assets -> list(assets)
        for asset in list(assets):
            balance = self._balances.get_balance(asset)
            cf = CashFlow(
                asset=asset,
                quantity=balance.total,
                time=datetime.now()
            )

            # We treat the balance as a cash flow to the account,
            # so that the performance analyzer could get the
            # correct PnL
            success = self._perf.set_cash_flow(cf)
            if success:
                not_ready_assets.clear(asset)
