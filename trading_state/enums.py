# Ref:
# https://developers.binance.com/docs/binance-spot-api-docs/enums

from enum import Enum


class StringEnum(Enum):
    def __str__(self):
        return self.value


class OrderType(StringEnum):
    # Market order
    MARKET = 'MARKET'

    # Price order
    LIMIT = 'LIMIT'

    # Stop Loss Orders
    # ----------------------------------------------

    # Stop market order
    STOP_LOSS = 'STOP_MARKET'

    # Stop limit order
    STOP_LOSS_LIMIT = 'STOP_LOSS_LIMIT'

    # Take Profit Orders
    # ----------------------------------------------

    # Take profit market order
    TAKE_PROFIT = 'TAKE_PROFIT'

    # Take profit limit order
    TAKE_PROFIT_LIMIT = 'TAKE_PROFIT_LIMIT'

    # Trailing stop is actualy not an order type,
    # but an extra settings about when to trigger the stop order,
    # which could apply to both stop loss orders and take profit orders


class MarketQuantityType(StringEnum):
    BASE = 'BASE'
    QUOTE = 'QUOTE'


class OrderSide(StringEnum):
    # USDT is locked
    BUY = 'BUY'

    # BTC is locked
    SELL = 'SELL'


class OrderedEnum(Enum):
    def __str__(self):
        return self.value[0]

    def lt(self, status: 'OrderedEnum') -> bool:
        """
        Returns `True` if the current status is less than the given status
        """

        return self.value[1] < status.value[1]


class OrderStatus(OrderedEnum):
    """
    The status of an order
    """

    # The ticket is initialized but has not been submitted to the exchange,
    # or the ticket is failed to create order so back to the initial state
    INIT = ('INIT', 0)

    # The order is creating,
    #   the request is about to send to the exchange,
    #   but not yet get response from the exchange
    SUBMITTING = ('SUBMITTING', 1)

    # The order is created via the exchange API
    CREATED = ('CREATED', 2)

    # The order is partially filled
    PARTIALLY_FILLED = ('PARTIALLY_FILLED', 3)

    # It is determined that the order should be cancelled
    ABOUT_TO_CANCEL = ('ABOUT_TO_CANCEL', 4)

    # The order is being cancelled
    CANCELLING = ('CANCELLING', 5)

    # The order is cancelled
    CANCELLED = ('CANCELLED', 6)

    # The order is filled, the status has the same status value as CANCELED,
    # which means the order is no longer active
    FILLED = ('FILLED', 6)

    # The order is rejected by the exchange
    REJECTED = ('REJECTED', 7)

    def completed(self) -> bool:
        no = self.value[1]

        return no == 6 or no == 7


class PositionTargetStatus(OrderedEnum):
    INIT = ('INIT', 0)
    ALLOCATED = ('ALLOCATED', 1)
    ACHIEVED = ('ACHIEVED', 2)


class TimeInForce(StringEnum):
    # Good Til Canceled
    # An order will be on the book unless the order is canceled.
    GTC = 'GTC'

    # Immediate Or Cancel
    # An order will try to fill the order as much as it can before the order expires.
    IOC = 'IOC'

    # Fill or Kill
    # An order will expire if the full order cannot be filled upon execution.
    FOK = 'FOK'


class STPMode(StringEnum):
    EXPIRE_MAKER = 'EXPIRE_MAKER'
    EXPIRE_TAKER = 'EXPIRE_TAKER'
    EXPIRE_BOTH = 'EXPIRE_BOTH'
    DECREMENT = 'DECREMENT'


class FeatureType(StringEnum):
    ICEBERG = 'ICEBERG'
    OCO = 'OCO'
    OTO = 'OTO'
    QUOTE_ORDER_QUANTITY = 'QUOTE_ORDER_QUANTITY'
    TRAILING_STOP = 'TRAILING_STOP'
    CANCEL_REPLACE = 'CANCEL_REPLACE'
    AMEND = 'AMEND'
    PEG_INSTRUCTIONS = 'PEG_INSTRUCTIONS'
    SPOT = 'SPOT'
    MARGIN = 'MARGIN'
    ORDER_TYPE = 'ORDER_TYPE'
    STP_MODE = 'STP_MODE'
    POST_ONLY = 'POST_ONLY'


class TradingStateEvent(Enum):
    POSITION_TARGET_UPDATED = 'POSITION_TARGET_UPDATED'

    # Ticket creation failed, which indicates it fails to create an order #
    #   ticket according to the position target
    TICKET_CREATE_FAILED = 'TICKET_CREATE_FAILED'

    # Order creation failed, which is different from TICKET_CREATE_FAILED,
    # this is when the order is rejected by the exchange
    ORDER_CREATE_FAILED = 'ORDER_CREATE_FAILED'

    ORDER_STATUS_UPDATED = 'ORDER_STATUS_UPDATED'
    ORDER_FILLED_QUANTITY_UPDATED = 'ORDER_FILLED_QUANTITY_UPDATED'
    PRICE_UPDATED = 'PRICE_UPDATED'
    SYMBOL_ADDED = 'SYMBOL_ADDED'
