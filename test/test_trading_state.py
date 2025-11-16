# import asyncio
import pytest

from trading_state.state import (
    TraderState,

    # TicketOrderSide
)
from trading_state.types import (
    Balance
)

BTC = 'BTC'
USDT = 'USDT'
SYMBOL = BTC + USDT


def set_price(state: TraderState):
    state.set_price(SYMBOL, 7000.)


def set_symbol_info(state: TraderState):
    state.set_symbol_info(
        SYMBOL,
        BTC,
        USDT,

        '0.01',
        '1000000',
        '0.01',

        '0.000001',
        '9000',
        '0.000001',

        '10.00000000'
    )


def set_balances(state: TraderState):
    state.update_balances([
        Balance(
            BTC,
            15.,
            0.
        ),
        Balance(
            USDT,
            0.001,
            0.
        )
    ])


def set_info(state: TraderState):
    set_price(state)
    set_symbol_info(state)


def test_support():
    state = TraderState()

    set_info(state)

    assert state.support_symbol(SYMBOL)

    LTEBTC = 'LTEBTC'

    assert not state.support_symbol(LTEBTC)
    assert state.create_ticket(
        LTEBTC,
        1.,
        False,
        None
    ) == (None, set())


def test_non_diff():
    state = TraderState()

    set_balances(state)
    set_info(state)

    state.expect(
        SYMBOL,
        1.,
        False,
        None
    )

    diffed, to_cancel = state.get_tickets()

    assert not diffed
    assert not to_cancel


def test_specified_price():
    state = TraderState()

    set_balances(state)
    set_info(state)

    ticket, to_cancel = state.create_ticket(
        SYMBOL,
        0.,
        False,
        7000.
    )

    assert not to_cancel
    assert ticket is not None
    assert ticket.price == 7000.

    ticket2, to_cancel2 = state.create_ticket(
        SYMBOL,
        0.,
        False,
        6000.
    )

    assert ticket in to_cancel2
    assert ticket2.price == 6000.

    state.expect(
        SYMBOL,
        1.,
        False,
        5000.
    )

    # Different direction, cancel previous ticket
    # ---------------------------------------------

    tickets, to_cancel = state.get_tickets()

    assert ticket2 in to_cancel
    assert not tickets

    # get_tickets again, previous ticket already closed
    # ---------------------------------------------

    tickets, to_cancel = state.get_tickets()

    assert not tickets
    assert not tickets

    # Update balances should reset diff

    state.update_balances([
        Balance(
            USDT,
            1000.,
            0.
        )
    ])

    tickets, to_cancel = state.get_tickets()

    ticket = tickets[0]

    assert ticket.price == 5000.
    assert ticket.symbol.name == SYMBOL
    assert not to_cancel

    # Will remove expectations

    state.create_ticket(
        SYMBOL,
        0.,
        False,
        None
    )

    assert not state._expected


@pytest.mark.asyncio
async def test_diff():
    state = TraderState()

    def expect_sell_all(quantity: float = 15.):
        # sell all
        state.expect(
            SYMBOL,
            0.,
            False,
            None
        )

        if not state._expected:
            return

        assert repr(state._expected[SYMBOL]) == '<SymbolPosition BTCUSDT: value: 0.0, asap: False, price: None>'

        diff, _ = state.get_tickets()

        if not diff:
            return

        ticket = diff[0]

        assert str(ticket.order_side) == 'SELL'
        assert ticket.symbol.name == SYMBOL
        assert ticket.price == 7000.

        assert ticket.quantity == '15.000000'

        assert ticket.locked_asset == BTC
        assert ticket.locked_quantity == quantity

        return ticket

    def expect_all_in():
        # buy all
        state.expect(
            SYMBOL,
            1.,
            False,
            None
        )

        diff, _ = state.get_tickets()

        if not diff:
            return

        return diff[0]

    state.expect(
        SYMBOL,
        0.,
        False,
        None
    )

    assert not state._expected, 'BTC not supported'

    tickets, to_cancel = state.get_tickets()
    assert not tickets
    assert not to_cancel

    assert expect_sell_all() is None, 'symbol info is not set'

    set_symbol_info(state)

    assert repr(state._symbol_infos[SYMBOL]) == '<SymbolInfo BTCUSDT: price_step >= 0.01, quantity_step >= 0.000001>'

    assert expect_sell_all() is None, 'price is not set'

    state.set_price(SYMBOL, 7000.)

    assert expect_sell_all() is None, 'balances are not set'

    state.update_balances([
        Balance(
            BTC,
            15.,
            0.
        ),
        Balance(
            USDT,
            0.001,
            0.
        )
    ])

    assert repr(state._balances[BTC]) == '<Balance BTC: free: 15.0, locked: 0.0>'

    ticket0 = expect_sell_all()
    assert ticket0 is not None

    # Unsolved tickets will be always returned by state.tickets

    tickets, _ = state.get_tickets()
    assert tickets[0] is ticket0

    # ticket 0 will be closed
    assert expect_all_in() is None

    ticket1 = expect_sell_all()

    assert ticket1 is not None

    # Test to close ticket duplicately
    # ------------------------------------------------------

    state.close_ticket(ticket1)
    state.close_ticket(ticket1)

    # Test to create ticket
    # ------------------------------------------------------

    ticket_sell, to_cancel = state.create_ticket(
        SYMBOL,
        0.,
        False,
        None
    )

    assert ticket_sell.position.value == 0.
    assert ticket_sell.symbol.name == SYMBOL
    assert not to_cancel

    # There is already a ticket
    # ------------------------------------------------------

    ticket_sell2, to_cancel2 = state.create_ticket(
        SYMBOL,
        0.,
        False,
        None
    )

    assert ticket_sell2 is None
    assert not to_cancel2

    ticket_buy, to_cancel = state.create_ticket(
        SYMBOL,
        1.,
        False,
        None
    )

    assert ticket_buy is None
    assert ticket_sell in to_cancel

    assert not state._expected

    state.update_balances([
        Balance(
            BTC,
            0.,
            0.
        ),
        Balance(
            USDT,
            70000,
            0.
        )
    ])

    assert expect_all_in() is not None
    assert expect_sell_all() is None

    # Set quota to 0
    state.set_quota(BTC, 0.)
    state.set_quota('ETH', 0.)

    assert expect_all_in() is None, 'not enough quota'

    state.set_quota(BTC, None)

    state.update_balances([
        Balance(
            USDT,
            0.006,
            0.
        )
    ])

    assert expect_all_in() is None, 'min price step'

    state.update_balances([
        Balance(
            USDT,
            9.,
            0.
        )
    ])

    assert expect_all_in() is None, 'min notional'

    state.update_balances([
        Balance(
            USDT,
            15.,
            0.
        )
    ])

    state.set_price(SYMBOL, 70000000.)
    state.set_price(SYMBOL, 70000000.)

    assert expect_all_in() is None, 'not enough'

    state.set_price(SYMBOL, 7000.)

    ticket2 = expect_all_in()

    assert ticket2 is not None

    assert ticket2.quantity == '0.002142'

    state.clear()

    tickets, to_cancel = state.get_tickets()

    assert not tickets
    assert not to_cancel

    assert not state._expected
