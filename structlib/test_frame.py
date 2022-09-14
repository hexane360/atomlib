import re

import pytest
import polars

from .frame import AtomFrame


def test_atom_frame_creation():
    frame = AtomFrame({
        'x': [0., 0., 0.],
        'y': [0., 0., 0.],
        'z': [0., 0., 0.],
        'elem': [1, 5, 22],
    })
    assert frame.select(('x', 'y', 'z')).dtypes == [polars.Float64] * 3
    assert frame.select('elem').dtypes[0] == polars.Int8

    assert list(frame.select('symbol').to_series()) == ["H", "B", "Ti"]

    frame = AtomFrame({
        'x': [0., 0., 0.],
        'y': [0., 0., 0.],
        'z': [0., 0., 0.],
        'symbol': ["H", "b+", "tI2+"],
    })

    assert list(frame.select('elem').to_series()) == [1, 5, 22]

    with pytest.raises(ValueError, match=re.escape("'AtomFrame' missing column(s) 'x', 'y'")):
        frame = AtomFrame({
            'z': [0., 0., 0.],
            'symbol': ["H", "b+", "tI2+"],
        })

    with pytest.raises(ValueError, match=re.escape("'AtomFrame' missing columns 'elem' and/or 'symbol'")):
        frame = AtomFrame({
            'x': [0., 0., 0.],
            'y': [0., 0., 0.],
            'z': [0., 0., 0.],
            'ele': [1, 5, 22],
        })

    empty = AtomFrame.empty()
    assert empty.select(('x', 'y', 'z')).dtypes == [polars.Float64] * 3
    assert empty.select('elem').dtypes[0] == polars.Int8


def test_with_mass():
    frame = AtomFrame({
        'x': [0., 0., 0.],
        'y': [0., 0., 0.],
        'z': [0., 0., 0.],
        'elem': [1, 5, 22],
    })

    new = frame.with_mass()
    assert 'mass' not in frame.columns
    assert frame.masses() is None

    mass = new.masses()
    assert mass is not None
    assert mass.to_numpy() == pytest.approx([1.008, 10.81, 47.867])
    assert mass.dtype == polars.Float32

    # idempotence
    assert new.with_mass() is new

    masses = [1., 2., 3.]
    new = frame.with_mass(masses)
    mass = new.masses()
    assert mass is not None
    assert mass.to_numpy() == pytest.approx(masses)
    assert mass.dtype == polars.Float32


def test_with_type():
    frame = AtomFrame({
        'x': [0., 0., 0., 0.],
        'y': [0., 0., 0., 0.],
        'z': [0., 0., 0., 0.],
        'elem': [22, 78, 22, 1],
    })

    new = frame.with_type()
    assert frame.types() is None
    types = new.types()
    assert types is not None
    assert types.dtype == polars.Int32
    assert list(types) == [2, 3, 2, 1]

    # idempotence
    assert new.with_type() is new

    frame = AtomFrame({
        'x': [0., 0., 0., 0., 0.],
        'y': [0., 0., 0., 0., 0.],
        'z': [0., 0., 0., 0., 0.],
        'symbol': ["Ag+", "Ag", "Na", "na", "Ag+"],
    }).with_type()


    types = frame.types()
    assert types is not None
    assert types.dtype == polars.Int32
    assert list(types) == [4, 3, 1, 2, 4]
