from decimal import Decimal, ROUND_DOWN

from typing import (
    Tuple,
    Optional,
    TypeVar,
    Dict,
    Generic,
    MutableSet,
    Callable,
    Any,
    Hashable
)


DECIMAL_ZERO = Decimal('0')


def apply_precision(number: Decimal, precision: int) -> Decimal:
    """
    Scale `number` to exactly `precision` decimal places.

    Examples::

        apply_precision(Decimal('1.234'), 2) # -> Decimal('1.23')
        apply_precision(Decimal('1.235'), 2) # -> Decimal('1.24')
    """

    # Build a quantizer like:
    #   precision=0 -> Decimal("1")
    #   precision=1 -> Decimal("0.1")
    #   precision=2 -> Decimal("0.01"), etc.
    quantizer = Decimal('1').scaleb(-precision)

    return number.quantize(quantizer, rounding=ROUND_DOWN)


def apply_tick_size(number: Decimal, tick_size: Decimal) -> Decimal:
    """
    Snap `number` down to the nearest multiple of `tick_size`.

    Examples::

        apply_tick_size(Decimal('0.023422'), Decimal('0.01'))
        # -> Decimal('0.02')

        apply_tick_size(Decimal('0.053422'), Decimal('0.02'))
        # -> Decimal('0.04')
    """

    # scale = number / tick_size, then floor it, then multiply back
    scale = (number / tick_size).to_integral_value(rounding=ROUND_DOWN)
    return scale * tick_size


def class_repr(
    self,
    main: Optional[str] = None,
    keys: Optional[Tuple[str]] = None
) -> str:
    """
    Returns a string representation of an class instance comprises of slots

    Args:
        main (Optional[str]): the main attribute to represent
        keys (Optional[Tuple[str]]): the attributes to represent
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


K = TypeVar('K', bound=Hashable)
V = TypeVar('V')

class DictSet(Generic[K, V]):
    _data: Dict[K, MutableSet[V]]

    def __init__(self):
        self._data: Dict[K, MutableSet[V]] = {}

    def __getitem__(self, key: K) -> MutableSet[V]:
        return self._data.setdefault(key, set[V]())

    def __contains__(self, key: K) -> bool:
        return key in self._data

    def clear(self) -> None:
        self._data.clear()


EventEmitterCallback = Callable[[...], None]

class EventEmitter(Generic[K]):
    """
    An simple event emitter implementation which does
    not ensure execution order of listeners
    """

    _listeners: DictSet[K, EventEmitterCallback]

    def __init__(self):
        self._listeners = DictSet[K, EventEmitterCallback]()

    def on(
        self,
        event: str,
        listener: EventEmitterCallback
    ) -> None:
        self._listeners[event].add(listener)

    def emit(
        self,
        event: str,
        *args: Any
    ) -> None:
        for listener in self._listeners[event]:
            listener(*args)

    def off(self) -> None:
        self._listeners.clear()
