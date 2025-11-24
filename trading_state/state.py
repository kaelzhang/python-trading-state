from typing import (
    List,
    Dict,
    Tuple,
    Set,
    Optional,
    Callable,
)
from dataclasses import dataclass

from decimal import Decimal

from .enums import (
    OrderStatus,
)
from .symbol import Symbol
from .balance import Balance
from .order import (
    Order,
    OrderHistory,
)
from .types import (
    SymbolPosition,
    PositionMetaData
)
from .common import (
    DECIMAL_ZERO,
)


FLOAT_ONE = 1.0
FLOAT_ZERO = 0.0

CreateOrderReturn = Tuple[
    Set[Order], # orders to create
    Set[Order]  # orders to cancel
]


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
- no checking for unexpected param types, but should check unexpected values
- no default parameters for all methods to avoid unexpected behavior
"""


def DEFAULT_GET_SYMBOL_NAME(base_asset: str, quote_asset: str) -> str:
    return f"{base_asset}{quote_asset}"


@dataclass
class TradingConfig:
    """
    Args:
        numeraire (str): the valuation currency (ref: https://en.wikipedia.org/wiki/Num%C3%A9raire) to use to:
        - calculate value of positions
        - calculate value of quotas
        get_symbol_name (Callable[[str, str], str]): a function to get the name of a symbol from its base and quote assets
    """
    numeraire: str
    max_order_history_size: int = 10000
    get_symbol_name: Callable[[str, str], str] = DEFAULT_GET_SYMBOL_NAME


class TradingState:
    """State Phase II

    - support base asset position between 0 and 1
    - support multiple base assets
    - support multiple quote assets

    Convention:
    - For a certain base asset,
      its related tickets should has the same direction

    Design principle:
    - The expectation settings are the final state that the system try to achieve.
    """

    _config: TradingConfig

    _balances: Dict[str, Balance]

    # _tickets: Dict[
    #     # asset
    #     str,
    #     TicketGroup
    # ]

    # For those dictionaries might be accessed by users,
    # use `str` as key, else use `Symbol` as key
    _symbols: Dict[str, Symbol]
    _symbol_prices: Dict[str, Decimal]
    _quotas: Dict[str, Decimal]

    _expected: Dict[Symbol, SymbolPosition]
    _old_expected: Dict[Symbol, SymbolPosition]

    _orders_to_cancel: Set[Order]
    _symbol_orders: Dict[Symbol, Order]

    # Order.id -> Order
    _orders: Dict[int, Order]

    _history: OrderHistory

    def __init__(
        self,
        config: TradingConfig
    ) -> None:
        self._config = config

        self._symbols = {}
        self._symbol_prices = {}

        # Asset quota dict: asset -> quantity
        self._quotas = {}

        self._balances = {}
        self._expected = {}

        # In the beginning, they are the same
        self._old_expected = self._expected

        self._orders_to_cancel = set[Order]()
        self._symbol_orders = {}

        self._history = OrderHistory(max_size=config.max_order_history_size)

        # self._tickets = {}
        # self._all_tickets = set()

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

        old_price = self._symbol_prices.get(symbol_name)

        if price == old_price:
            # If the price does not change, should not reset diff
            return False

        self._symbol_prices[symbol_name] = price
        self._reset_diff()

        return True

    def get_price(
        self,
        symbol_name: str
    ) -> Optional[Decimal]:
        """
        Get the price of a symbol
        """

        return self._symbol_prices.get(symbol_name)

    def set_symbol(
        self,
        symbol: Symbol
    ) -> None:
        """
        Set (add or update) the symbol info for a symbol

        Args:
            symbol (Symbol): the symbol to set
        """

        self._symbols[symbol.name] = symbol

    def set_quota(
        self,
        asset: str,
        quota: Optional[Decimal]
    ) -> None:
        """
        Set the quota for a certain asset.

        The quota of an asset limits:
        - the maximum quantity of the **numeraire** asset the trader could **BUY** the asset,
        - no SELL.

        Args:
            asset (str): the asset to set the quota for
            quota (Decimal | None): the maximum quantity of the numeraire asset the trader could BUY the asset. `None` means no quota.

        For example, if::

            state.set_quota('BTC', Decimal('35000'))

        - current BTC price: $7000
        - base asset balance (USDT): $70000

        Then, the trader could only buy 5 BTC,
        although the balance is enough to buy 10 BTC
        """

        if quota is not None and quota < DECIMAL_ZERO:
            # Decimal('-1') means no limit
            quota = None

        old_quota = self._quotas.get(asset)

        if quota is None:
            # Just remove the quota if no limit
            del self._quotas[asset]
        else:
            self._quotas[asset] = quota

        if old_quota != quota:
            self._reset_diff()

    def update_balances(
        self,
        new: List[Balance]
    ) -> None:
        """
        Update user balances, including normal assets and quote assets

        Usage::

            state.update_balances([
                Balance('BTC', Decimal('8'), Decimal('0'))
            ])
        """

        balances = []

        for balance in new:
            balances.append(
                (balance.asset, balance)
            )

        self._balances.update(balances)

        # This will trigger diffing again
        self._reset_diff()

    def support_symbol(self, symbol_name: str) -> bool:
        """
        Check whether the symbol is supported
        """

        return symbol_name in self._symbols

    # def summarize(self):
    #     ...

    def position(
        self,
        symbol_name: str,
        # TODO: whether to keep this or not
        # realized: bool
    ) -> Optional[float]:
        """
        Get the position of a symbol

        Args:
            symbol_name (str): the name of the symbol
            # realized (bool): A realized position does not include the ammount of base asset that is locked by tickets.

        Returns:
            Optional[float]: the position of the symbol. If the symbol is not supported, returns `None`.
        """

        symbol = self._get_symbol(symbol_name)

        if symbol is None:
            return None

        return self._get_position(symbol)

    # Prerequisites:
    # - current price: $10
    # - balance: 1
    # - quota: $20
    # - open orders:
    #   - buy 1 @ $9
    #   - sell 1 @ $11
    # - quota: $11
    #
    # Case 1:
    # expect: + 0.1 @ current (+ $2)
    # Result:
    # we will cancel the buy order
    # since we plan to buy in a higher price

    # Case 2:
    def expect(
        self,
        symbol_name: str,
        position: float,
        price: Decimal | None,
        asap: bool,
        data: PositionMetaData = {}
    ) -> bool:
        """
        Update expectation, returns whether it is successfully updated

        Args:
            position (float): the position to expect, should be between `0.0` and `1.0`. The position refers to the current holding of the base asset as a proportion of its maximum allowed quota
            asap (bool = False): whether to execute the expectation immediately, that it will generate a market order
            price (Decimal | None = None): the price to expect. If not provided, the price will be determined by the current price.
            data (Dict[str, Any] = {}): the data attached to the expectation, which will also attached to the created order, order history for future reference.

        Returns:
            bool: whether the expectation is successfully updated

        Usage::
            state.expect('BTCUSDT', 1.) # to all-in BTC

            state.expect('BTCUSDT', -0.1, delta=True, asap=True) # to decrease the position by 10% by market order
        """

        symbol = self._get_symbol(symbol_name)

        if symbol is None:
            return False

        # Normalize the position to be between 0 and 1
        position = max(FLOAT_ZERO, min(position, FLOAT_ONE))

        current_position = self._get_position(symbol)

        if current_position == position:
            # If the position is the same, no need to update
            # We treat it as a success
            return True

        # Create a new dict so that will be considered as changed
        self._expected = {
            **self._expected
        }

        if asap:
            # If we want to trade asap (market order),
            # then we should not set the price
            price = None

        self._expected[symbol] = SymbolPosition(
            symbol,
            position,
            asap,
            price,
            data
        )

        return True

    # def clear(self) -> None:
    #     """
    #     Clear all expectations and tickets
    #     """

    #     self._expected.clear()
    #     self._all_tickets.clear()

    #     for group in self._tickets.values():
    #         group.clear()

    # def create_order(self) -> CreateOrderReturn:
    #     """
    #     Create orders according to the positions, and remove position expectations

    #     Returns:
    #         tuple:
    #         - A set of newly-created orders
    #         - A set of orders which needs to be cancelled
    #     """

    #     for symbol, position in self._expected.items():

    #     symbol = self._get_symbol(symbol_name)

    #     if symbol is None:
    #         # Symbol is not supported
    #         return None, set()

    #     # Remove symbol position expectation
    #     self._expected.pop(symbol_name, None)

    #     position = SymbolPosition(
    #         symbol_info.symbol,
    #         position,
    #         asap,
    #         price
    #     )

    #     return self._create_ticket(position)

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

        orders_to_cancel = self._orders_to_cancel
        self._orders_to_cancel = set[Order]()

        # TODO

        # return [
        #     ticket
        #     for ticket in self._all_tickets

        #     # Only collect tickets whose status are <= `status`
        #     if ticket.status.lt(status)
        # ], tickets_to_cancel

    # def close_ticket(
    #     self,
    #     ticket: OrderTicket
    # ) -> None:
    #     """
    #     Just close a ticket

    #     Args:
    #         ticket (OrderTicket): the ticket to close
    #         position_fulfilled (:obj:`bool`, optional):
    #     """

    #     self._remove_expectation(ticket.position)

    #     self._get_ticket_group(ticket.symbol.base_asset).close(ticket)
    #     self._get_ticket_group(ticket.locked_asset).close(ticket)
    #     self._all_tickets.discard(ticket)

    # End of public methods ---------------------------------------------

    def _get_symbol_price(
        self,
        symbol: Symbol
    ) -> Decimal:
        """
        Get the price of a symbol based on numeraire asset
        """

        if symbol.quote_asset == self._config.numeraire:
            return self._symbol_prices.get(symbol.name)

        valuation_symbol = self._get_symbol(
            self._config.get_symbol_name(
                symbol.base_asset,
                self._config.numeraire
            )
        )

        ...

    def _get_position(
        self,
        symbol: Symbol
    ) -> float:
        # TODO
        ...

    def _remove_expectation(self, position: SymbolPosition) -> None:
        symbol_name = position.symbol.name

        existed = self._expected.get(symbol_name)

        if existed is not None and existed.equals_to(position):
            del self._expected[symbol_name]

    def _get_symbol(self, symbol_name: str) -> Optional[Symbol]:
        return self._symbols.get(symbol_name)

    # def _get_ticket_group(self, asset: str) -> TicketGroup:
    #     group = self._tickets.get(asset)

    #     if group is None:
    #         group = TicketGroup()
    #         self._tickets[asset] = group

    #     return group

    def _get_ticket_locked(self, asset: str) -> float:
        group = self._get_ticket_group(asset)

        locked = FLOAT_ZERO

        if not group.sell:
            return locked

        for ticket in group.sell:
            locked += ticket.locked_quantity

        return locked

    def _get_balance(self, asset: str) -> Tuple[Decimal, Decimal]:
        """
        The balance updates are not always in time, so that we should double confirm the locked quantity of an asset.
        This method will return the ensured tuple of free and locked quantity
        """

        balance = self._balances.get(asset)

        if balance is None:
            # We dont have this asset
            return DECIMAL_ZERO, DECIMAL_ZERO

        locked_by_tickets = self._get_ticket_locked(asset)

        delta = balance.locked - locked_by_tickets

        if delta >= FLOAT_ZERO:
            # The locked quantity has been updated
            return balance.free, balance.locked

        # The locked quantity has not been updated,
        # we use the larget locked quantity
        return balance.free + delta, locked_by_tickets

    # def _adjust_quote_quantity(
    #     self,
    #     asset: str,
    #     price: float,
    #     quantity: float
    # ) -> float:
    #     """
    #     Adjust the quote quantity according to quota
    #     """

    #     quota = self._quotas.get(asset)

    #     if quota is None:
    #         return quantity

    #     free, locked = self._get_balance(asset)

    #     return max(
    #         min(quantity, quota) - (free + locked) * price,
    #         0.
    #     )

    def _has_diff(self) -> bool:
        """
        Check whether the trading state has diff
        """

        return self._expected is self._old_expected

    def _reset_diff(self):
        self._old_expected = {}

    def _diff(self) -> None:
        """
        Diff the position expectations based on what we create tickets

        Returns
            set: a set of tickets to cancel
        """

        if self._has_diff():
            # If there is no new expectation,
            # so skip diffing
            return

        self._old_expected = self._expected

        for position in self._expected.values():
            self._create_order_from_position(position)

    def _mark_order_to_cancel(self, order: Order) -> None:
        order.status = OrderStatus.ABOUT_TO_CANCEL
        self._orders_to_cancel.add(order)
        del self._symbol_orders[order.position.symbol]

    def _create_order_from_position(
        self,
        position: SymbolPosition
    ) -> None:
        """
        Create a order from a symbol position
        """

        symbol = position.symbol
        symbol_name = symbol.name
        asset = symbol.base_asset
        # group = self._get_ticket_group(asset)

        existing_order = self._symbol_orders.get(symbol)

        # We only keep one order for a single symbol
        if existing_order is not None:
            if existing_order.position == position:
                # already a valid order for the position
                return

            self._mark_order_to_cancel(existing_order)

        # TODO:
        # How many orders could a single symbol have?

        # tickets_to_cancel = self._get_tickets_to_cancel(
        #     group,
        #     position,
        #     TicketOrderSide.SELL if position.value == FLOAT_ZERO
        #     else TicketOrderSide.BUY
        # )

        # `info` is always not None, which is ensured by self.expect()
        # info = self._get_symbol_info(symbol_name)

        price = position.price

        if not position.asap and price is None:
            price = self.get_price(symbol_name)

            if price is None:
                # The price is not ready,
                # then skip to avoid unexpected result,
                # it will have another in the next tick
                return

        # price = (
        #     # Order with specified price
        #     position.price
        #     or self._symbol_prices.get(symbol_name)
        # )

        # if price is None:
        #     # The price is not ready,
        #     # then skip to avoid unexpected result
        #     return None, tickets_to_cancel

        # if position.value == FLOAT_ZERO:
        #     # Sell all out
        #     # ---------------------------------------------------------------

        #     quantity, _ = self._get_balance(asset)

        #     if quantity < info.min_quantity_step:
        #         # is less than minimum trading quantity
        #         return None, tickets_to_cancel

        #     quantity_str = apply_precision(
        #         quantity,
        #         info.min_quantity_step_precision
        #     )

        #     # Apply precision
        #     quantity = float(quantity_str)

        #     ticket = OrderTicket(
        #         TicketOrderSide.SELL,
        #         symbol,
        #         price,
        #         quantity_str,
        #         asset,
        #         quantity,
        #         position,
        #         info  # type: ignore
        #     )

        #     self._get_ticket_group(asset).sell.add(ticket)

        # else:
        #     # Buy all in
        #     # ---------------------------------------------------------------

        #     quote_asset = symbol.quote_asset

        #     free, _ = self._get_balance(quote_asset)

        #     available = self._adjust_quote_quantity(asset, price, free)

        #     if available < info.min_price_step:
        #         # The quantity of the current cash is less than
        #         # the minimun trading price
        #         return None, tickets_to_cancel

        #     if available < info.min_notional:
        #         return None, tickets_to_cancel

        #     if free < price * info.min_quantity_step:
        #         # The quantity of the current cash is less than
        #         # the value minium asset quantity
        #         return None, tickets_to_cancel

        #     quantity_str = apply_precision(
        #         available / price,
        #         info.min_quantity_step_precision
        #     )

        #     locked_quote_quantity = price * float(quantity_str)

        #     ticket = OrderTicket(
        #         TicketOrderSide.BUY,
        #         symbol,
        #         price,
        #         quantity_str,
        #         quote_asset,
        #         locked_quote_quantity,
        #         position,
        #         info  # type: ignore
        #     )

        #     self._get_ticket_group(asset).buy.add(ticket)
        #     self._get_ticket_group(quote_asset).sell.add(ticket)

        # self._all_tickets.add(ticket)

        # return ticket, tickets_to_cancel

    # def _get_tickets_to_cancel(
    #     self,
    #     group: TicketGroup,
    #     position: SymbolPosition,
    #     direction: TicketOrderSide
    # ) -> Set[OrderTicket]:
    #     opposite_group = group.get(
    #         TicketOrderSide.SELL if direction is TicketOrderSide.BUY
    #         else TicketOrderSide.BUY
    #     )

    #     if opposite_group:
    #         # If tickets of opposite direction exist,
    #         # then cancel them
    #         for ticket in opposite_group:
    #             self.close_ticket(ticket)

    #         return opposite_group

    #     to_cancel = set()
    #     tickets = group.get(direction)

    #     if position.price is not None:
    #         for ticket in tickets:
    #             if not ticket.position.equals_to(position):
    #                 to_cancel.add(ticket)
    #                 self.close_ticket(ticket)

    #     return to_cancel
