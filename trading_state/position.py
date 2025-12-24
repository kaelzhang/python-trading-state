from typing import (
    List, Dict
)

from dataclasses import dataclass, field
from decimal import Decimal

from .symbol import Symbols
from .order import (
    Order,
    Trade
)
from .common import (
    DECIMAL_ZERO,
    FactoryDict
)
from .enums import (
    OrderSide,
    OrderType
)


@dataclass
class Lot:
    quantity: Decimal
    price: Decimal

    @property
    def cost(self) -> Decimal:
        return self.quantity * self.price


@dataclass
class Position:
    total_quantity: Decimal = DECIMAL_ZERO
    total_cost: Decimal = DECIMAL_ZERO

    lots: List[Lot] = field(default_factory=list)

    @property
    def avg_cost(self) -> Decimal:
        return (
            self.total_cost / self.total_quantity
            if self.total_quantity > 0
            else DECIMAL_ZERO
        )


class PositionTracker:
    """
    PositionTracker is used to track the position changes of the account,
    to calculate the cost basis and unrealized PnL of the account
    """

    _symbols: Symbols
    _positions: FactoryDict[str, Position]

    def __init__(
        self,
        symbols: Symbols
    ):
        self._symbols = symbols
        self._positions = FactoryDict[str, Position](Position)

    def track_order(self, order: Order) -> None:
        """
        Track the order and update the position of the account

        It should only track the order if the order is filled or cancelled.
        """

        ticket = order.ticket
        side = ticket.side

        symbol = ticket.symbol
        base_asset = symbol.base_asset
        quote_asset = symbol.quote_asset

        base_quantity = order.filled_quantity

        if quantity.is_zero():
            # The order is not filled at all
            return

        if ticket.type is OrderType.MARKET:
            cost = (
                order.quote_quantity
                * self._symbols.valuation_price(quote_asset)
            )
            valuation_price = cost / quantity
        else:
            valuation_price = ticket.price

        if side is OrderSide.BUY:
            self._update_position(order)

    def _update_position(
        self, asset: str,
        quantity: Decimal,
        valuation_price: Decimal,
        side: OrderSide
    ) -> None:
        """
        Update the position of the account according to FIFO method
        """

        position = self._positions[asset]

        # Buy, increase the position
        if side is OrderSide.BUY:
            cost = quantity * valuation_price

            position.total_quantity += quantity
            position.total_cost += cost

            # 添加新的持仓批次
            position.lots.append(Lot(quantity, valuation_price))

        # Sell, decrease the position
        else:
            # 卖出，减少持仓（先进先出）
            if position.total_quantity.is_zero():
                return  # 没有持仓可卖

            remaining_quantity = quantity
            new_lots = []

            # 先卖最早的批次
            for lot in position.lots:
                if remaining_quantity <= 0:
                    new_lots.append(lot)
                    continue

                # Sell the whole lot
                if lot.quantity <= remaining_quantity:
                    remaining_quantity -= lot.quantity
                    position.total_cost -= lot.cost
                    position.total_quantity -= lot.quantity
                else:
                    lot.quantity -= remaining_quantity

                    # Reduce the total cost and quantity of the position
                    position.total_cost -= remaining_quantity * lot.price
                    position.total_quantity -= remaining_quantity
                    new_lots.append(lot)

                    remaining_quantity = DECIMAL_ZERO

            position.lots = [lot for lot in new_lots if lot.quantity > 0]

    def get_position(self, asset: str) -> Position:
        """Get position for a certain asset"""

        return self._positions[asset]

    def get_total_cost_basis(self, price_book: PriceBook) -> Decimal:
        """获取所有持仓的总成本（以USDT计）"""
        total_cost = Decimal('0')

        for asset, pos in self.positions.items():
            if asset == 'USDT':
                total_cost += pos['total_quantity']
            else:
                total_cost += pos['total_cost']

        return total_cost

    def get_positions_snapshot(self, price_book: PriceBook) -> Dict[str, Dict[str, Any]]:
        """获取当前所有持仓的快照（包括未实现盈亏）"""
        snapshot = {}

        for asset, pos in self.positions.items():
            if asset == 'USDT':
                current_price = Decimal('1.0')
                current_value = pos['total_quantity']
                unrealized_pnl = Decimal('0')
                unrealized_pct = Decimal('0')
            else:
                current_price = price_book.get_price(asset)
                current_value = pos['total_quantity'] * current_price if current_price > 0 else Decimal('0')
                unrealized_pnl = current_value - pos['total_cost']
                unrealized_pct = (unrealized_pnl / pos['total_cost'] * 100) if pos['total_cost'] > 0 else Decimal('0')

            snapshot[asset] = {
                'quantity': pos['total_quantity'],
                'avg_cost': pos['avg_cost'],
                'current_price': current_price,
                'current_value': current_value,
                'cost_basis': pos['total_cost'],
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': unrealized_pct,
                'roe': unrealized_pct
            }

        return snapshot
