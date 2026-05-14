import pytest

from trading_state.common import EventEmitter, FactoryDict


def test_event_emitter_off_per_listener():
    em: EventEmitter[str] = EventEmitter()
    seen: list[tuple[str, tuple]] = []

    def a(*args):
        seen.append(('a', args))

    def b(*args):
        seen.append(('b', args))

    em.on('x', a)
    em.on('x', b)

    em.emit('x', 1)
    assert seen == [('a', (1,)), ('b', (1,))]

    # Drop only `a`; `b` continues to fire.
    em.off('x', a)
    seen.clear()
    em.emit('x', 2)
    assert seen == [('b', (2,))]


def test_event_emitter_off_event_only_clears_event_bucket():
    em: EventEmitter[str] = EventEmitter()
    fired: list[str] = []
    em.on('x', lambda *_: fired.append('x'))
    em.on('y', lambda *_: fired.append('y'))

    em.off('x')
    em.emit('x', 1)
    em.emit('y', 1)
    assert fired == ['y']


def test_event_emitter_off_event_listener_unknown_is_noop():
    em: EventEmitter[str] = EventEmitter()
    # off() on an event that was never subscribed is a no-op.
    em.off('never', lambda: None)
    # off(event, listener) for a listener that was never attached is
    # also a no-op (idempotent).
    em.on('x', lambda *_: None)
    em.off('x', lambda *_: None)


def test_event_emitter_off_listener_without_event_raises():
    em: EventEmitter[str] = EventEmitter()
    with pytest.raises(TypeError, match='requires an explicit event'):
        em.off(listener=lambda: None)


def test_event_emitter_blanket_off_clears_everything():
    em: EventEmitter[str] = EventEmitter()
    fired: list[str] = []
    em.on('x', lambda *_: fired.append('x'))
    em.on('y', lambda *_: fired.append('y'))

    em.off()
    em.emit('x', 1)
    em.emit('y', 1)
    assert fired == []


def test_factory_dict_clear():
    fd: FactoryDict[str, list] = FactoryDict(list)
    fd['a'].append(1)
    fd['b'].append(2)
    assert dict(fd.items()) == {'a': [1], 'b': [2]}
    fd.clear()
    assert dict(fd.items()) == {}
