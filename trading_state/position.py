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

    Computed properties:
    - avg_cost (Decimal): the average cost of the position
    """

    quantity: Decimal = DECIMAL_ZERO
    cost: Decimal = DECIMAL_ZERO

    lots: List[Lot] = field(default_factory=list)

    @property
    def avg_cost(self) -> Decimal:
        return (
            self.cost / self.quantity
            if self.quantity > 0
            else DECIMAL_ZERO
        )


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """
    A snapshot of a position

    - quantity (Decimal): the quantity of the position
    - cost (Decimal): the cost of the position
    - valuation_price (Decimal): the valuation price of the position
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
        # sum of base costs
        bc = DECIMAL_ZERO
        # sum of quote quantities
        qq = DECIMAL_ZERO
        # sum of quote costs
        qc = DECIMAL_ZERO

        for trade in order.trades:
            bq += trade.base_quantity
            bc += trade.base_quantity * trade.base_price
            qq += trade.quote_quantity
            qc += trade.quote_quantity * trade.quote_price

        ticket = order.ticket
        side = ticket.side

        symbol = ticket.symbol
        # base asset
        ba = symbol.base_asset
        # quote asset
        qa = symbol.quote_asset

        if side is OrderSide.SELL:
            ba, bq, bc, qa, qq, qc = qa, qq, qc, ba, bq, bc

        # base -> increase position
        # quote -> decrease position

        # Increase
        # -------------------------------------------------
        self.update_position(ba, bq, bc / bq)

        # Should not handle
        if self._symbols.is_account_asset(qa):
            return realized_pnl

        # Position to decrease
        # -------------------------------------------------
        position = self._positions[qa]
        if position.quantity > 0:
            avg_cost = position.avg_cost
            cost = qq * avg_cost
            proceeds = qq * self._symbols.valuation_price(qa)
            realized_pnl = proceeds - cost

        self.update_position(qa, - qq, qc / qq)

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
            if position.quantity.is_zero():
                # No position to decrease
                return

            remaining_quantity = - quantity
            new_lots = []

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
                else:
                    lot.quantity -= remaining_quantity

                    # Reduce the total cost and quantity of the position
                    position.cost -= remaining_quantity * lot.price
                    position.quantity -= remaining_quantity
                    new_lots.append(lot)

                    remaining_quantity = DECIMAL_ZERO

            position.lots = [lot for lot in new_lots if lot.quantity > 0]

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
