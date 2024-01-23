
import re
import typing as t

import pytest
import numpy
import polars

from .elem import get_elem, get_elems, get_sym, get_mass
from .elem import get_radius, get_ionic_radius


@pytest.mark.parametrize(('sym', 'elem'), (
    ('Ar', 18),
    ('  Ag', 47),
    ('nA1+', 11),
    ('Na_test', 11),
    (11, 11),
    (polars.Series(['nA1+', 'SI2', 'Ag']), polars.Series([11, 14, 47]))
))
def test_get_elem(sym: str, elem: int):
    if isinstance(sym, polars.Series):
        assert (get_elem(sym) == elem).all()  # type: ignore
    else:
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


def test_get_elem_series_nulls():
    sym = polars.Series(['Al', None, 'Ag', 'Na'])

    assert (get_elem(sym) == polars.Series([13, None, 47, 11])).all()

    with pytest.raises(ValueError, match="Invalid element symbol 'None'"):
        get_elem(sym, skip_nulls=False)


def test_get_sym_series():
    elem = polars.Series((14, 8, 1, 102))
    sym = get_sym(elem)

    assert tuple(sym) == ('Si', 'O', 'H', 'No')


def test_get_sym_series_nulls():
    elem = polars.Series((74, 102, 62, None, 19))

    assert (get_sym(elem) == polars.Series(["W", "No", "Sm", None, "K"])).all()

    with pytest.raises(ValueError, match="Invalid atomic number None"):
        get_sym(elem, skip_nulls=False)


def test_get_elem_fail():
    with pytest.raises(ValueError, match="Invalid atomic number -5"):
        get_elem(-5)

    with pytest.raises(ValueError, match="Invalid element symbol 'We'"):
        get_elem('We')

    with pytest.raises(ValueError, match=re.escape("Invalid element symbol '<4*sd>'")):
        get_elem("<4*sd>")

    with pytest.raises(ValueError, match="Invalid element symbol 'Ay'"):
        get_elem(polars.Series(["Ag", "au", "Ay"]))


def test_get_sym_fail():
    with pytest.raises(ValueError, match="Invalid atomic number 255"):
        get_sym(255)

    with pytest.raises(ValueError, match="Invalid atomic number 255"):
        get_sym(polars.Series([12, 14, 255, 1]))


def test_get_elems_fail():
    with pytest.raises(ValueError, match="Unknown element 'By' in 'BaBy'."):
        get_elems('BaBy')

    with pytest.raises(ValueError, match=re.escape("Invalid compound '<4*sd>'")):
        get_elems("<4*sd>")


@pytest.mark.parametrize(('elem', 'mass'), (
    (1, 1.008),
    ([1, 47, 82], numpy.array([1.008, 107.8682, 207.2])),
    (numpy.array([1, 47, 82]), numpy.array([1.008, 107.8682, 207.2])),
    (polars.Series([1, 47, 82]), polars.Series([1.008, 107.8682, 207.2])),
))
def test_get_mass(elem, mass):
    result = get_mass(elem)

    if isinstance(mass, polars.Series):
        assert isinstance(result, polars.Series)
        assert result.to_numpy() == pytest.approx(mass.to_numpy())
        assert result.dtype == polars.Float32
    else:
        assert result == pytest.approx(mass)

    if isinstance(result, numpy.ndarray):
        assert result.dtype == numpy.float32


@pytest.mark.parametrize(('elem', 'radius'), (
    (47, 1.65),
    (1, 0.53),
    (55, 2.98),
    (70, 2.22),
))
def test_get_radius(elem, radius):
    assert get_radius(elem) == pytest.approx(radius)


@pytest.mark.parametrize(('elem', 'charge', 'radius'), (
    (47, +1, 1.29),
    (1, -1, 2.08),
    (34, +6, 0.56),
))
def test_get_ionic_radius(elem, charge, radius):
    assert get_ionic_radius(elem, charge) == pytest.approx(radius)