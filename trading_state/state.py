from typing import (
    List,
    Dict,
    Tuple,
    Set,
    Optional
)

from .util import (
    apply_precision
)

from .types import (
    Balance,
    SymbolPosition,
    OrderTicket,
    TicketOrderSide,
    TicketOrderStatus,

    TicketGroup,
    SymbolInfo,

    FuncCancelOrder,

    FLOAT_ZERO
)


CreateTicketReturn = Tuple[
    Optional[OrderTicket],
    Set[OrderTicket]
]


"""
Terminology
- asset: BTC, USDT, or others
- symbol: trading pairs of assets

Principle
- be pure
- be sync
- be passive, no triggers
- no default parameters for public methods
"""


class TraderState:
    """State Phase II

    - only support base asset position 0 or 1
    - support multiple base assets
    - only support a single quote asset

    Convention:
    - For a certain base asset,
      its related tickets should has the same direction

    Design principle:
    - The expectation settings are the final state that the system try to achieve.
    """

    _balances: Dict[str, Balance]

    _expected: Dict[str, SymbolPosition]
    _old_expected: Dict[str, SymbolPosition]

    _tickets: Dict[
        # asset
        str,
        TicketGroup
    ]

    _symbol_prices: Dict[str, float]
    _symbol_infos: Dict[str, SymbolInfo]

    _cancel_order: Optional[FuncCancelOrder]
    _all_tickets: Set[OrderTicket]

    def __init__(self) -> None:
        # Asset quota dict: asset -> quantity
        self._quotas = {}

        self._balances = {}
        self._expected = {}

        # In the beginning, they are the same
        self._old_expected = self._expected

        self._symbol_prices = {}

        self._symbol_infos = {}

        self._tickets = {}
        self._all_tickets = set()

    # Public methods
    # ------------------------------------------------------------------------

    def set_price(
        self,
        symbol_name: str,
        price: float
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

    def set_symbol_info(
        self,

        # symbol name, ie the trading pair name, e.g. BTCUSDT
        symbol_name: str,

        # base asset, ie the coin, token or asset currency to trade, e.g. BTC
        base_asset: str,

        # quote asset, ie the asset to quote, quote currency, e.g. USDT
        quote_asset: str,

        # Price filters
        # ------------------------------------------------
        min_price: str,
        max_price: str,

        # min ticker size
        min_price_step: str,

        # Lot size filters
        # ------------------------------------------------
        min_quantity: str,
        max_quantity: str,
        min_quantity_step: str,

        min_notional: str
    ) -> None:
        """
        Set trading info of a symbol

        Usage::

        state.set_symbol_info(
            'BTCUSDT',
            'BTC',
            'USDT',

            '0.01',
            '1000000',
            '0.01',

            '0.000001',
            '9000',
            '0.000001',

            '10.00000000'
        )
        """

        self._symbol_infos[symbol_name] = SymbolInfo(
            symbol_name,
            base_asset,
            quote_asset,

            min_price,
            max_price,
            min_price_step,

            min_quantity,
            max_quantity,
            min_quantity_step,

            min_notional
        )

    def set_quota(
        self,
        asset: str,
        quota: Optional[float]
    ) -> None:
        """
        Set the quota for a certain asset.

        The quota of an asset limits
        the maximum quantity the trader could **BUY** the asset,
        but DON'T limit sell.

        Args:
            quota (float): how much available base quote asset buy the asset

        For example, if

        - quota of BTC: $35000
        - current price: 7000
        - cash quantity: $70000

        Then, the trader could only buy 5 BTC,
        although the cash is enough to buy 10 BTC
        """

        old_quota = self._quotas.get(asset)

        self._quotas[asset] = quota

        if old_quota != quota:
            self._reset_diff()

    def update_balances(
        self,
        new: List[Balance]
    ) -> None:
        """
        Update user balances, including normal assets and cash asset(USDT)

        Usage::

            state.update([
                ('BTC', 8.)
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

        return symbol_name in self._symbol_infos

    def expect(
        self,
        symbol_name: str,
        position: float,
        asap: bool,
        price: Optional[float]
    ) -> bool:
        """
        Update expectation, returns whether it is successfully updated

        Usage::

            # to all-in BTC
            state.expect('BTCUSDT', 1.)

        Args:
            price (:obj:`float`, optional): the price to expect. For now, this argument is not actually used because the current strategy is not based on price prediction but technical shape.
        """

        symbol_info = self._get_symbol_info(symbol_name)

        if symbol_info is None:
            # This is not duplicated with `self.support_symbol()`
            # TraderState not cares about the usage of business, so it provides
            #   several util methods
            # The same is true for `self.create_ticket()`
            return False

        # Create a new dict so that will be considered as changed
        self._expected = {
            **self._expected
        }

        self._expected[symbol_name] = SymbolPosition(
            symbol_info.symbol,
            position,
            asap,
            price
        )

        return True

    def clear(self) -> None:
        """
        Clear all expectations and tickets
        """

        self._expected.clear()
        self._all_tickets.clear()

        for group in self._tickets.values():
            group.clear()

    def create_ticket(
        self,
        symbol_name: str,
        position: float,
        asap: bool,
        price: Optional[float]
    ) -> CreateTicketReturn:
        """
        Create a ticket according to the position of a symbol, and remove position expectations of the symbol

        Returns:
            tuple: An optional newly-created ticket and tickets which needs to be cancelled
        """

        symbol_info = self._get_symbol_info(symbol_name)

        if symbol_info is None:
            # Symbol is not supported
            return None, set()

        # Remove symbol position expectation
        self._expected.pop(symbol_name, None)

        position = SymbolPosition(
            symbol_info.symbol,
            position,
            asap,
            price
        )

        return self._create_ticket(position)

    def get_tickets(
        self,
        status: TicketOrderStatus = TicketOrderStatus.INIT
    ) -> Tuple[
        List[OrderTicket],
        Set[OrderTicket]
    ]:
        """
        Get all unsubmitted tickets which are not submitted

        Args:
            status (:obj:`TicketOrderStatus`, optional): The maximun status phase that the returned tickets will have. Defaults to `TicketOrderStatus.INIT`

        Returns
            tuple: a list of available tickets and a set of tickets to cancel
        """

        tickets_to_cancel = self._diff()

        return [
            ticket
            for ticket in self._all_tickets

            # Only collect tickets whose status are <= `status`
            if ticket.status.lt(status)
        ], tickets_to_cancel

    def close_ticket(
        self,
        ticket: OrderTicket
    ) -> None:
        """
        Just close a ticket

        Args:
            ticket (OrderTicket): the ticket to close
            position_fulfilled (:obj:`bool`, optional):
        """

        self._remove_expectation(ticket.position)

        self._get_ticket_group(ticket.symbol.base_asset).close(ticket)
        self._get_ticket_group(ticket.locked_asset).close(ticket)
        self._all_tickets.discard(ticket)

    # End of public methods ---------------------------------------------

    def _remove_expectation(self, position: SymbolPosition) -> None:
        symbol_name = position.symbol.name

        existed = self._expected.get(symbol_name)

        if existed is not None and existed.equals_to(position):
            del self._expected[symbol_name]

    def _reset_diff(self):
        self._old_expected = {}

    def _get_symbol_info(self, symbol_name: str) -> Optional[SymbolInfo]:
        return self._symbol_infos.get(symbol_name)

    def _get_ticket_group(self, asset: str) -> TicketGroup:
        group = self._tickets.get(asset)

        if group is None:
            group = TicketGroup()
            self._tickets[asset] = group

        return group

    def _get_ticket_locked(self, asset: str) -> float:
        group = self._get_ticket_group(asset)

        locked = FLOAT_ZERO

        if not group.sell:
            return locked

        for ticket in group.sell:
            locked += ticket.locked_quantity

        return locked

    def _get_balance(self, asset: str) -> Tuple[float, float]:
        """
        The balance updates are not always in time, so that we should double confirm the locked quantity of a asset.
        This method will return the ensured tuple of free and locked quantity
        """

        balance = self._balances.get(asset)

        if balance is None:
            # We dont have this asset
            return FLOAT_ZERO, FLOAT_ZERO

        locked_by_tickets = self._get_ticket_locked(asset)

        delta = balance.locked - locked_by_tickets

        if delta >= FLOAT_ZERO:
            # The locked quantity has been updated
            return balance.free, balance.locked

        # The locked quantity has not been updated,
        # we use the larget locked quantity
        return balance.free + delta, locked_by_tickets

    def _adjust_quote_quantity(
        self,
        asset: str,
        price: float,
        quantity: float
    ) -> float:
        """
        Adjust the quote quantity according to quota
        """

        quota = self._quotas.get(asset)

        if quota is None:
            return quantity

        free, locked = self._get_balance(asset)

        return max(
            min(quantity, quota) - (free + locked) * price,
            0.
        )

    def _diff(self) -> Set[OrderTicket]:
        """
        Diff the position expectations based on what we create tickets

        Returns
            set: a set of tickets to cancel
        """

        tickets_to_cancel = set()

        if self._expected is self._old_expected:
            # If there is no new expectation,
            # so skip diffing
            return tickets_to_cancel

        self._old_expected = self._expected

        if self._expected:
            for position in self._expected.values():
                _, to_cancel = self._create_ticket(position)
                tickets_to_cancel |= to_cancel

        return tickets_to_cancel

    def _create_ticket(
        self,
        position: SymbolPosition
    ) -> CreateTicketReturn:
        """
        Create a ticket from a symbol position
        """

        symbol = position.symbol
        symbol_name = symbol.name
        asset = symbol.base_asset
        group = self._get_ticket_group(asset)

        tickets_to_cancel = self._get_tickets_to_cancel(
            group,
            position,
            TicketOrderSide.SELL if position.value == FLOAT_ZERO
            else TicketOrderSide.BUY
        )

        # `info` is always not None, which is ensured by self.expect()
        info = self._get_symbol_info(symbol_name)

        price = (
            # Order with specified price
            position.price
            or self._symbol_prices.get(symbol_name)
        )

        if price is None:
            # The price is not ready,
            # then skip to avoid unexpected result
            return None, tickets_to_cancel

        if position.value == FLOAT_ZERO:
            # Sell all out
            # ---------------------------------------------------------------

            quantity, _ = self._get_balance(asset)

            if quantity < info.min_quantity_step:
                # is less than minimum trading quantity
                return None, tickets_to_cancel

            quantity_str = apply_precision(
                quantity,
                info.min_quantity_step_precision
            )

            # Apply precision
            quantity = float(quantity_str)

            ticket = OrderTicket(
                TicketOrderSide.SELL,
                symbol,
                price,
                quantity_str,
                asset,
                quantity,
                position,
                info  # type: ignore
            )

            self._get_ticket_group(asset).sell.add(ticket)

        else:
            # Buy all in
            # ---------------------------------------------------------------

            quote_asset = symbol.quote_asset

            free, _ = self._get_balance(quote_asset)

            available = self._adjust_quote_quantity(asset, price, free)

            if available < info.min_price_step:
                # The quantity of the current cash is less than
                # the minimun trading price
                return None, tickets_to_cancel

            if available < info.min_notional:
                return None, tickets_to_cancel

            if free < price * info.min_quantity_step:
                # The quantity of the current cash is less than
                # the value minium asset quantity
                return None, tickets_to_cancel

            quantity_str = apply_precision(
                available / price,
                info.min_quantity_step_precision
            )

            locked_quote_quantity = price * float(quantity_str)

            ticket = OrderTicket(
                TicketOrderSide.BUY,
                symbol,
                price,
                quantity_str,
                quote_asset,
                locked_quote_quantity,
                position,
                info  # type: ignore
            )

            self._get_ticket_group(asset).buy.add(ticket)
            self._get_ticket_group(quote_asset).sell.add(ticket)

        self._all_tickets.add(ticket)

        return ticket, tickets_to_cancel

    def _get_tickets_to_cancel(
        self,
        group: TicketGroup,
        position: SymbolPosition,
        direction: TicketOrderSide
    ) -> Set[OrderTicket]:
        opposite_group = group.get(
            TicketOrderSide.SELL if direction is TicketOrderSide.BUY
            else TicketOrderSide.BUY
        )

        if opposite_group:
            # If tickets of opposite direction exist,
            # then cancel them
            for ticket in opposite_group:
                self.close_ticket(ticket)

            return opposite_group

        to_cancel = set()
        tickets = group.get(direction)

        if position.price is not None:
            for ticket in tickets:
                if not ticket.position.equals_to(position):
                    to_cancel.add(ticket)
                    self.close_ticket(ticket)

        return to_cancel
