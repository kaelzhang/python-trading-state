"""
Reconciliation between order updates and balance updates.

State store and exchange communicate asynchronously: fill signals on
order updates and balance snapshots arrive on independent channels and
may interleave. ReconciliationManager tracks, per (order, asset) pair,
the cumulative impact of the order on the asset that has already been
reflected in a Balance update we observed. The diff between
`current_impact_on_asset(order, asset)` and that settled value is the
"unsettled" portion exposed via `state.unsettled(asset)` and consumed by
`state.exposure(..., include_unsettled_inflow=..., include_unsettled_outflow=...)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import (
    Dict,
    Set,
    TYPE_CHECKING,
)

from .common import DECIMAL_ZERO
from .enums import OrderSide

if TYPE_CHECKING:
    from .order import Order


@dataclass(frozen=True, slots=True)
class UnsettledFlow:
    """
    Aggregated unsettled impact on a single asset, across all tracked
    orders.

    `inflow` and `outflow` are both non-negative magnitudes:
    - `inflow`  = total positive impact (e.g. BUY fills increasing
                  the base asset, SELL fills increasing the quote
                  asset) reported via order updates but not yet visible
                  in a balance snapshot.
    - `outflow` = symmetric on the negative side.
    """
    inflow: Decimal
    outflow: Decimal


def current_impact_on_asset(order: 'Order', asset: str) -> Decimal:
    """
    Signed cumulative impact of `order` on `asset`, computed from the
    most recent filled / quote / commission values recorded on the
    order.

    Components combined under one path (avoids per-component dicts):
      base side  -> +/- filled_quantity     (sign by ticket.side)
      quote side -> -/+ quote_quantity      (sign by ticket.side)
      commission -> - commission_quantity   (only when commission_asset
                                             equals `asset`)
    """
    impact = DECIMAL_ZERO
    ticket = order.ticket
    symbol = ticket.symbol

    if asset == symbol.base_asset:
        if ticket.side is OrderSide.BUY:
            impact += order.filled_quantity
        else:
            impact -= order.filled_quantity

    if asset == symbol.quote_asset:
        if ticket.side is OrderSide.BUY:
            impact -= order.quote_quantity
        else:
            impact += order.quote_quantity

    if asset == order.commission_asset:
        impact -= order.commission_quantity

    return impact


def order_touches(order: 'Order', asset: str) -> bool:
    ticket = order.ticket
    return (
        ticket.symbol.base_asset == asset
        or ticket.symbol.quote_asset == asset
        or order.commission_asset == asset
    )


class ReconciliationManager:
    """
    Tracks, per (order, asset) pair, the cumulative impact of the order
    on the asset that is known to have been reflected in a balance
    snapshot seen so far.

    Orders are registered on add_order and purged on terminal status or
    cancellation. The manager itself never raises; ordering / staleness
    of balance updates is the responsibility of the caller.
    """

    _orders: Set['Order']
    _settled: Dict['Order', Dict[str, Decimal]]

    def __init__(self) -> None:
        self._orders = set()
        self._settled = {}

    def register(self, order: 'Order') -> None:
        self._orders.add(order)

    def purge(self, order: 'Order') -> None:
        self._orders.discard(order)
        self._settled.pop(order, None)

    def on_balance_set(
        self,
        asset: str,
        balance_time: datetime,
    ) -> None:
        """
        Called from state after a Balance for `asset` has been
        accepted. For every tracked order touching `asset` whose
        latest order-update is at-or-before `balance_time`, record the
        current impact on `asset` as the new settled value. Orders
        newer than the balance keep their previous settled value, so
        any later-arriving fill diff remains unsettled.
        """
        for order in self._orders:
            if not order_touches(order, asset):
                continue
            if (
                order.updated_at is None
                or order.updated_at <= balance_time
            ):
                impact = current_impact_on_asset(order, asset)
                self._settled.setdefault(order, {})[asset] = impact

    def unsettled_for(self, asset: str) -> UnsettledFlow:
        inflow = DECIMAL_ZERO
        outflow = DECIMAL_ZERO

        for order in self._orders:
            if not order_touches(order, asset):
                continue
            settled = self._settled.get(order, {}).get(
                asset, DECIMAL_ZERO
            )
            current = current_impact_on_asset(order, asset)
            delta = current - settled

            if delta > 0:
                inflow += delta
            elif delta < 0:
                outflow += -delta

        return UnsettledFlow(inflow=inflow, outflow=outflow)
