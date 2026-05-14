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
        Track the order and update the position of the account.

        It should only track the order if the order is filled or cancelled.

        Commission overlap rules — what `commission_asset` does to each
        side:

        - If `commission_asset == increase_asset`, the exchange
          delivers (gross − commission) of that asset, so the tracked
          increase shrinks by `commission_quantity`. The cost basis
          stays at gross-account-currency / kept-quantity (i.e. the
          per-unit cost goes up because we paid the same money for
          fewer units).
        - If `commission_asset == decrease_asset` and that asset is
          FIFO-tracked (non-account), the FIFO removal widens by
          `commission_quantity`; realized PnL is taken from the wider
          cost basis and no separate `cc` is subtracted.
        - If `commission_asset == decrease_asset` but the decrease is
          an account currency (so no FIFO runs), `cc` is folded into
          the new position's cost basis instead — otherwise the fee
          would vanish from PnL accounting.
        - If commission is in a third asset (BNB-style), `cc` is added
          to the cost basis of the new position when there is one to
          track, or subtracted from realized PnL on disposal when the
          increase is account-tracked.

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
        comm_asset = order.commission_asset
        comm_qty = order.commission_quantity

        if side is OrderSide.BUY:
            increase_asset = base_asset
            increase_qty = bq
            increase_cost = bc

            decrease_asset = quote_asset
            decrease_qty = qq
            # `proceeds` here is the value of the buy in account currency
            # — it is the inflow on the increase side. The variable
            # name is preserved for symmetry with SELL where it really
            # is the proceeds; the realized-PnL formula below uses it
            # identically in both directions.
            proceeds = qc
        else:
            increase_asset = quote_asset
            increase_qty = qq
            increase_cost = qc

            decrease_asset = base_asset
            decrease_qty = bq
            proceeds = qc

        commission_overlaps_increase = comm_asset == increase_asset
        commission_overlaps_decrease = comm_asset == decrease_asset

        # Commission consumed from the increase asset: tracked
        # increase is gross − commission.
        if commission_overlaps_increase:
            increase_qty -= comm_qty

        # Commission consumed from the decrease asset: FIFO removal
        # widens by the commission amount (only meaningful when the
        # decrease is non-account-tracked; the widened decrease_qty
        # is silently ignored for account-asset decreases below).
        if commission_overlaps_decrease:
            decrease_qty += comm_qty

        # Fold cc into the increase cost basis when the commission was
        # not already absorbed by the increase-side shrinkage AND
        # cannot be absorbed by FIFO on the decrease side (because the
        # decrease is account-tracked, so _decrease_position is
        # skipped). Third-asset commissions always hit this branch.
        increase_cost_total = increase_cost
        if (
            comm_asset is not None
            and not commission_overlaps_increase
            and (
                not commission_overlaps_decrease
                or self._symbols.is_account_asset(decrease_asset)
            )
        ):
            increase_cost_total += cc

        if increase_qty > 0:
            self.update_position(
                increase_asset,
                increase_qty,
                increase_cost_total / increase_qty,
            )

        if self._symbols.is_account_asset(decrease_asset):
            return realized_pnl

        # Realized PnL for the decreased asset based on FIFO cost.
        cost = self._decrease_position(decrease_asset, decrease_qty)

        # `cc` is only subtracted separately from realized PnL when it
        # was neither absorbed via the widened FIFO (decrease overlap)
        # nor via the shrunken increase (increase overlap). The
        # increase-overlap absorption is only effective when the
        # increase is itself FIFO-tracked (non-account); when the
        # increase is account-tracked, the shrinkage has no PnL effect
        # so cc still needs to be deducted here.
        if (
            comm_asset is None
            or commission_overlaps_decrease
            or (
                commission_overlaps_increase
                and not self._symbols.is_account_asset(increase_asset)
            )
        ):
            pnl_commission = DECIMAL_ZERO
        else:
            pnl_commission = cc

        realized_pnl = proceeds - cost - pnl_commission

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
