import re

import pytest
import numpy
import polars

from .atoms import Atoms


def test_atom_frame_creation():
    frame = Atoms({
        'x': [0., 0., 0.],
        'y': [0., 0., 0.],
        'z': [0., 0., 0.],
        'elem': [1, 5, 22],
    })
    assert frame.select(('x', 'y', 'z')).dtypes == [polars.Float64] * 3
    assert frame.select('elem').dtypes[0] == polars.Int8

    assert list(frame.select('symbol').to_series()) == ["H", "B", "Ti"]

    frame = Atoms({
        'x': [0., 0., 0.],
        'y': [0., 0., 0.],
        'z': [0., 0., 0.],
        'symbol': ["H", "b+", "tI2+"],
    })

    assert list(frame.select('elem').to_series()) == [1, 5, 22]

    with pytest.raises(ValueError, match=re.escape("'Atoms' missing column(s) 'x', 'y'")):
        frame = Atoms({
            'z': [0., 0., 0.],
            'symbol': ["H", "b+", "tI2+"],
        })

    with pytest.raises(ValueError, match=re.escape("'Atoms' missing columns 'elem' and/or 'symbol'")):
        frame = Atoms({
            'x': [0., 0., 0.],
            'y': [0., 0., 0.],
            'z': [0., 0., 0.],
            'ele': [1, 5, 22],
        })

    empty = Atoms.empty()
    assert empty.select(('x', 'y', 'z')).dtypes == [polars.Float64] * 3
    assert empty.select('elem').dtypes[0] == polars.Int8


def test_repr():
    from polars import Series, Float64, Int8, Int64, Utf8  # type: ignore

    atoms = Atoms({
        'x': [0., 1., 2.],
        'y': [1., 1., 1.],
        'z': [0., 2., 5.],
        'elem': [1, 5, 22],
        'type': [1, 2, 3],
    })

    new_atoms = eval(atoms.__repr__())

    assert atoms.schema == new_atoms.schema
    atoms.assert_equal(new_atoms)

    assert str(atoms) == """\
Atoms, shape: (3, 6)
┌─────┬─────┬─────┬──────┬──────┬────────┐
│ x   ┆ y   ┆ z   ┆ elem ┆ type ┆ symbol │
│ --- ┆ --- ┆ --- ┆ ---  ┆ ---  ┆ ---    │
│ f64 ┆ f64 ┆ f64 ┆ i8   ┆ i64  ┆ str    │
╞═════╪═════╪═════╪══════╪══════╪════════╡
│ 0.0 ┆ 1.0 ┆ 0.0 ┆ 1    ┆ 1    ┆ H      │
│ 1.0 ┆ 1.0 ┆ 2.0 ┆ 5    ┆ 2    ┆ B      │
│ 2.0 ┆ 1.0 ┆ 5.0 ┆ 22   ┆ 3    ┆ Ti     │
└─────┴─────┴─────┴──────┴──────┴────────┘\
"""


def test_concat():
    frame1 = Atoms({
        'x': [0., 1., 2.],
        'y': [1., 1., 1.],
        'z': [0., 2., 5.],
        'elem': [1, 5, 22],
        #'type': [1, 2, 3],
    })

    frame2 = Atoms({
        'x': [3., 4., 5.],
        'y': [1., 1., 1.],
        'z': [0., 2., 5.],
        'elem': [1, 5, 22],
    })

    frame1.concat(frame2).assert_equal(Atoms({
        'x': [0., 1., 2., 3., 4., 5.],
        'y': [1., 1., 1., 1., 1., 1.],
        'z': [0., 2., 5., 0., 2., 5.],
        'elem': [1, 5, 22, 1, 5, 22],
    }))

    Atoms.concat((frame2, frame1)).assert_equal(Atoms({
        'x': [3., 4., 5., 0., 1., 2.],
        'y': [1., 1., 1., 1., 1., 1.],
        'z': [0., 2., 5., 0., 2., 5.],
        'elem': [1, 5, 22, 1, 5, 22],
    }))


def test_coords():
    frame = Atoms({
        'x': [0., 1., -1., -3.],
        'y': [1., 1., 1., 2.],
        'z': [0., 2., 5., 8.],
        'elem': [1, 5, 22, 22],
    })

    expected = numpy.array([
        [0., 1., 0.],
        [1., 1., 2.],
        [-1., 1., 5.],
        [-3., 2., 8.]
    ])

    assert frame.coords() == pytest.approx(expected)

    assert frame.coords(polars.col('x') < 0.) == pytest.approx(expected[expected[:, 0] < 0.])
    assert frame.coords(polars.col('elem') == 22) == pytest.approx(expected[2:])
    assert frame.coords([False, True, False, True]) == pytest.approx(expected[[1, 3]])



def test_mass():
    frame = Atoms({
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


def test_type():
    frame = Atoms({
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

    frame = Atoms({
        'x': [0., 0., 0., 0., 0.],
        'y': [0., 0., 0., 0., 0.],
        'z': [0., 0., 0., 0., 0.],
        'symbol': ["Ag+", "Ag", "Na", "na", "Ag+"],
    }).with_type()


    types = frame.types()
    assert types is not None
    assert types.dtype == polars.Int32
    assert list(types) == [4, 3, 1, 2, 4]


def test_wobble():
    frame = Atoms({
        'x': [0., 0., 0., 0.],
        'y': [0., 0., 0., 0.],
        'z': [0., 0., 0., 0.],
        'elem': [22, 78, 22, 1],
    })

    new = frame.with_wobble()
    assert 'wobble' not in frame
    assert 'wobble' in new
    assert new.select('wobble').to_numpy() == pytest.approx([0., 0., 0., 0.])
    assert new.with_wobble() is new


def test_occupancy():
    frame = Atoms({
        'x': [0., 0., 0., 0.],
        'y': [0., 0., 0., 0.],
        'z': [0., 0., 0., 0.],
        'elem': [22, 78, 22, 1],
    })

    new = frame.with_occupancy()
    assert 'frac_occupancy' not in frame
    assert 'frac_occupancy' in new
    assert new.select('frac_occupancy').to_numpy() == pytest.approx([1., 1., 1., 1.])
    assert new.with_occupancy() is new
