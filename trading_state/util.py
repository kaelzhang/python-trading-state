import math
import decimal

from typing import (
    Optional,
    Tuple
)


def apply_precision(value: float, decimals: int) -> str:
    """
    Stringify a float according to the given precision without rounding

    Usage::

        apply_precision(0.123456, 2)  # '0.12'
    """

    magnitude = 10 ** decimals

    return format(
        # formatting is always with rounding,
        # so, we must
        math.floor(value * magnitude) / magnitude,
        f'.{decimals}f'
    )


def class_repr(
    self,
    main: Optional[str] = None,
    keys: Optional[Tuple[str]] = None
) -> str:
    """
    """

    Class = type(self)

    slots = Class.__slots__ if keys is None else keys

    string = f'<{Class.__name__}'

    if main is not None:
        string += f' {getattr(self, main)}'

    string += ': '

    string += ', '.join([
        f'{name}: {getattr(self, name)}'
        for name in slots if name != main
    ])

    string += '>'

    return string


decimal_ctx = decimal.Context()
decimal_ctx.prec = 20


def float_to_str(f: float) -> str:
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """

    d = decimal_ctx.create_decimal(repr(f))
    return format(d, 'f')
