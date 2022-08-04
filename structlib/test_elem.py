
import pytest
import polars

from .elem import get_elem, get_sym


@pytest.mark.parametrize(('sym', 'elem'), (
    ('Ar', 18),
    ('  Ag', 47),
    ('nA1+', 11),
    ('Na_test', 11),
))
def test_get_elem(sym: str, elem: int):
    assert get_elem(sym) == elem
    assert get_sym(elem).lower() in sym.lower()


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