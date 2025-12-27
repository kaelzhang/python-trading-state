from decimal import Decimal, ROUND_DOWN
from datetime import datetime

from typing import (
    Tuple,
    Optional,
    TypeVar,
    Dict,
    Generic,
    Callable,
    Any,
    Hashable,
    Union,
    Set,
    Iterator,
    List
)


DECIMAL_ZERO = Decimal('0')
DECIMAL_ONE = Decimal('1')
DECIMAL_INF = Decimal('Infinity')

SuccessOrException = Optional[Exception]
type ValueOrException[T] = Union[
    Tuple[None, T],
    Tuple[Exception, None]
]


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

    string = f'{Class.__name__}('

    if main is not None:
        string += f'{getattr(self, main)}'

    attrs = [
        f'{name}={getattr(self, name)}'
        for name in slots if name != main
    ]

    if attrs:
        string += ' ' + ', '.join(attrs)

    string += ')'

    return string


K = TypeVar('K', bound=Hashable)
V = TypeVar('V')


class FactoryDict(Generic[K, V]):
    def __init__(
        self,
        factory: Callable[[], V]
    ):
        self._data: Dict[K, V] = {}
        self._factory = factory

    def get(self, key: K) -> Optional[V]:
        """Just get, no new creation
        """
        return self._data.get(key)

    def __getitem__(self, key: K) -> V:
        value = self._data.get(key)
        if value is None:
            value = self._factory()
            self._data[key] = value
        return value

    # def __contains__(self, key: K) -> bool:
    #     return key in self._data

    def __delitem__(self, key: K) -> None:
        self._data.pop(key, None)

    def items(self) -> Iterator[Tuple[K, V]]:
        return self._data.items()

    def clear(self) -> None:
        self._data.clear()


class EventEmitter(Generic[K]):
    """
    An simple event emitter implementation which does
    not ensure execution order of listeners
    """

    _listeners: FactoryDict[K, List[Callable]]

    def __init__(self):
        self._listeners = FactoryDict[K, Set[Callable]](list[Callable])

    def on(
        self,
        event: str,
        listener: Callable
    ) -> None:
        self._listeners[event].append(listener)

    def emit(
        self,
        event: str,
        *args: Any
    ) -> None:
        listeners = self._listeners.get(event)
        if listeners is None:
            return

        # Regenerate the list to avoid .off() during iteration
        for listener in list(listeners):
            listener(*args)

    def off(self) -> None:
        self._listeners.clear()


def timestamp_to_datetime(timestamp: int) -> datetime:
    """
    Convert a timestamp in milliseconds to a datetime object
    """
    return datetime.fromtimestamp(timestamp / 1000)
