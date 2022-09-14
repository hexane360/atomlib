
import re
import typing as t

import pytest
import polars

from .elem import get_elem, get_elems, get_sym


@pytest.mark.parametrize(('sym', 'elem'), (
    ('Ar', 18),
    ('  Ag', 47),
    ('nA1+', 11),
    ('Na_test', 11),
    (11, 11),
))
def test_get_elem(sym: str, elem: int):
    assert get_elem(sym) == elem
    if isinstance(sym, str):
        assert get_sym(elem).lower() in sym.lower()


@pytest.mark.parametrize(('sym', 'elems'), (
    ('Ar', [18]),
    ('Ag+I', [47, 53]),
    ('AlN', [13, 7]),
))
def test_get_elems(sym: str, elems: t.Sequence[int]):
    assert get_elems(sym) == elems


def test_get_elem_series():
    sym = polars.Series(('Ar', 'Ag', 'nA1+', '  Na_test'))

    elem = get_elem(sym)
    print(elem)

    assert tuple(elem) == (18, 47, 11, 11)
    assert all(roundtrip.lower() in sym.lower() for (roundtrip, sym) in zip(get_sym(elem), sym))


def test_get_sym_series():
    elem = polars.Series((14, 8, 1, 102))
    sym = get_sym(elem)

    assert tuple(sym) == ('Si', 'O', 'H', 'No')


def test_get_elem_fail():
    with pytest.raises(ValueError, match="Invalid atomic number -5"):
        get_elem(-5)

    with pytest.raises(ValueError, match="Invalid element symbol 'We'"):
        get_elem('We')

    with pytest.raises(ValueError, match=re.escape("Invalid element symbol '<4*sd>'")):
        get_elem("<4*sd>")


def test_get_elems_fail():
    with pytest.raises(ValueError, match="Unknown element 'By' in 'BaBy'."):
        get_elems('BaBy')

    with pytest.raises(ValueError, match=re.escape("Invalid compound '<4*sd>'")):
        get_elems("<4*sd>")