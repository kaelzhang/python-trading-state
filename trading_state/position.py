from typing import (
    List, Dict
)

from dataclasses import dataclass, field
from decimal import Decimal

from .symbol import SymbolManager
from .balance import BalanceManager
from .order import Order
from .common import (
    DECIMAL_ZERO,
    FactoryDict
)
from .enums import OrderSide


@dataclass(slots=True)
class Lot:
    """
    A single lot of a position, which is used to track the cost basis of an asset in details.

    - quantity (Decimal): the quantity of the lot
    - price (Decimal): the price of the lot

    Computed properties:
    - cost (Decimal): the cost of the lot
    """

    quantity: Decimal
    price: Decimal

    @property
    def cost(self) -> Decimal:
        return self.quantity * self.price


# Mutable, should not be frozen
@dataclass(slots=True)
class Position:
    """
    The position of an asset, which is used to track the cost basis and unrealized PnL.

    - quantity (Decimal): the total quantity of the position
    - cost (Decimal): the total cost of the position
    - lots (List[Lot]): the lots of the position
    """

    quantity: Decimal = DECIMAL_ZERO
    cost: Decimal = DECIMAL_ZERO

    lots: List[Lot] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """
    A snapshot of a position

    - quantity (Decimal): the quantity of the position
    - cost (Decimal): the cost of the position
    - valuation_price (Decimal): the valuation price of the position

    Computed properties:
    - value (Decimal): the value of the position
    - unrealized_pnl (Decimal): the unrealized PnL of the position
    """

    quantity: Decimal
    cost: Decimal
    valuation_price: Decimal

    @property
    def value(self) -> Decimal:
        return self.quantity * self.valuation_price

    @property
    def unrealized_pnl(self) -> Decimal:
        return self.value - self.cost


PositionSnapshots = Dict[str, PositionSnapshot]


class PositionTracker:
    """
    PositionTracker is used to track the position changes of the account,
    to calculate the cost basis and unrealized PnL of the account
    """

    _symbols: SymbolManager
    _positions: FactoryDict[str, Position]

    def __init__(
        self,
        symbols: SymbolManager,
        balances: BalanceManager
    ):
        self._symbols = symbols
        self._balances = balances

        self._positions = FactoryDict[str, Position](Position)

    def init(self) -> None:
        """Set the initial positions of the account according to the balances
        """

        for balance in self._balances.get_balances():
            price = self._symbols.valuation_price(balance.asset)

            if price.is_zero():
                continue

            self.update_position(
                balance.asset,
                balance.total,
                price
            )

    def track_order(self, order: Order) -> Decimal:
        """
        Track the order and update the position of the account

        It should only track the order if the order is filled or cancelled.

        Returns:
            Decimal: the realized PnL of the order
        """

        realized_pnl = DECIMAL_ZERO
        base_quantity = order.filled_quantity

        if base_quantity.is_zero():
            # The order is not filled at all
            return realized_pnl

        # sum of base quantities
        bq = DECIMAL_ZERO
        # sum of base costs (account currency)
        bc = DECIMAL_ZERO
        # sum of quote quantities
        qq = DECIMAL_ZERO
        # sum of quote costs (account currency)
        qc = DECIMAL_ZERO
        # sum of commission costs (account currency)
        cc = DECIMAL_ZERO

        for trade in order.trades:
            bq += trade.base_quantity
            bc += trade.base_quantity * trade.base_price
            qq += trade.quote_quantity
            qc += trade.quote_quantity * trade.quote_price
            cc += trade.commission_cost

        ticket = order.ticket
        side = ticket.side

        symbol = ticket.symbol
        base_asset = symbol.base_asset
        quote_asset = symbol.quote_asset

        if side is OrderSide.BUY:
            increase_asset = base_asset
            increase_qty = bq
            increase_cost = bc

            decrease_asset = quote_asset
            decrease_qty = qq
            proceeds = qc
        else:
            increase_asset = quote_asset
            increase_qty = qq
            increase_cost = qc

            decrease_asset = base_asset
            decrease_qty = bq
            proceeds = qc

        # Increase position (account assets are ignored)
        if increase_qty > 0:
            self.update_position(
                increase_asset,
                increase_qty,
                increase_cost / increase_qty
            )

        if self._symbols.is_account_asset(decrease_asset):
            return realized_pnl

        # Realized PnL for the decreased asset based on FIFO cost
        cost = self._decrease_position(decrease_asset, decrease_qty)
        realized_pnl = proceeds - cost - cc

        return realized_pnl

    def update_position(
        self,
        asset: str,
        quantity: Decimal,
        price: Decimal
    ) -> None:
        """
        Update the position of the account according to FIFO method

        Args:
            price (Decimal): the average price for the certain quantity of the asset
        """

        if self._symbols.is_account_asset(asset):
            # Do not track the account assets
            return

        if quantity.is_zero():
            return

        position = self._positions[asset]

        # Increase the position
        if quantity > 0:
            cost = quantity * price

            position.quantity += quantity
            position.cost += cost

            # Add the new lot of the asset position
            position.lots.append(Lot(quantity, price))

        # Decrease the position
        else:
            self._decrease_position(asset, - quantity)

    def _decrease_position(
        self,
        asset: str,
        quantity: Decimal
    ) -> Decimal:
        # quantity must not be zero, no need to check

        position = self._positions[asset]
        if position.quantity.is_zero():
            return DECIMAL_ZERO

        remaining_quantity = quantity
        new_lots = []
        cost_removed = DECIMAL_ZERO

        # FIFO
        for lot in position.lots:
            if remaining_quantity <= 0:
                new_lots.append(lot)
                continue

            # Sell the whole lot
            if lot.quantity <= remaining_quantity:
                remaining_quantity -= lot.quantity
                position.cost -= lot.cost
                position.quantity -= lot.quantity
                cost_removed += lot.cost
            else:
                cost = remaining_quantity * lot.price
                lot.quantity -= remaining_quantity

                # Reduce the total cost and quantity of the position
                position.cost -= cost
                position.quantity -= remaining_quantity
                cost_removed += cost
                new_lots.append(lot)

                remaining_quantity = DECIMAL_ZERO

        position.lots = [lot for lot in new_lots if lot.quantity > 0]

        return cost_removed

    def snapshots(self) -> PositionSnapshots:
        """Get the snapshot of all positions, including unrealized PnL, based on the account currency.
        """

        snapshots = {}

        for asset, position in self._positions.items():
            valuation_price = self._symbols.valuation_price(asset)

            snapshots[asset] = PositionSnapshot(
                quantity=position.quantity,
                cost=position.cost,
                valuation_price=valuation_price
            )

        return snapshots
