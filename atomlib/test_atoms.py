import re

import pytest
import numpy
from numpy.testing import assert_array_equal
import polars
from polars.testing import assert_frame_equal

from .atoms import Atoms, _with_columns_stacked


def test_with_columns_stacked():
    df = polars.DataFrame({
        'before': [0, 1, 2, 3],
        'x': [1., 2., 3., 4.],
        'y': [2., 5., 6., 7.],
        'z': [3., 6., 8., 9.],
        'after': [0, 1, 2, 3],
    })

    arr = [[1., 2., 3.], [2., 5., 6.], [3., 6., 8.], [4., 7., 9.]]

    assert_frame_equal(
        _with_columns_stacked(df, ('x', 'y', 'z'), 'coords'),
        polars.DataFrame({
            'before': polars.Series([0, 1, 2, 3], dtype=polars.Int64),
            'coords': polars.Series(arr, dtype=polars.Array(polars.Float64, 3)),
            'after': polars.Series([0, 1, 2, 3], dtype=polars.Int64),
        })
    )


def test_atom_frame_creation():
    frame = Atoms({
        'x': [0., 0., 0.],
        'y': [0., 0., 0.],
        'z': [0., 0., 0.],
        'elem': [1, 5, 22],
    })
    assert frame['coords'].dtype == polars.Array(polars.Float64, 3)
    assert frame.select('elem').dtypes[0] == polars.Int8

    assert list(frame.select('symbol').to_series()) == ["H", "B", "Ti"]

    frame = Atoms({
        'x': [0., 0., 0.],
        'y': [0., 0., 0.],
        'z': [0., 0., 0.],
        'symbol': ["H", "b+", "tI2+"],
    })

    assert list(frame.select('elem').to_series()) == [1, 5, 22]

    with pytest.raises(ValueError, match=re.escape("'Atoms' missing column(s) 'coords'")):
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
    assert frame['coords'].dtype == polars.Array(polars.Float64, 3)
    assert empty.select('elem').dtypes[0] == polars.Int8


#@pytest.mark.xfail(reason="Polars array __repr__ broken")
def test_repr():
    from polars import Series, Float64, Int8, Int32, Int64, Utf8, String, Array  # type: ignore

    atoms = Atoms({
        'x': [0., 1., 2.],
        'y': [1., 1., 1.],
        'z': [0., 2., 5.],
        'elem': [1, 5, 22],
        'type': [1, 2, 3],
    })

    print(atoms.__repr__())

    new_atoms = eval(atoms.__repr__())

    assert atoms.schema == new_atoms.schema
    atoms.assert_equal(new_atoms)

    assert str(atoms) == """\
Atoms, shape: (3, 4)
┌─────────────────┬──────┬──────┬────────┐
│ coords          ┆ elem ┆ type ┆ symbol │
│ ---             ┆ ---  ┆ ---  ┆ ---    │
│ array[f64, 3]   ┆ i8   ┆ i32  ┆ str    │
╞═════════════════╪══════╪══════╪════════╡
│ [0.0, 1.0, 0.0] ┆ 1    ┆ 1    ┆ H      │
│ [1.0, 1.0, 2.0] ┆ 5    ┆ 2    ┆ B      │
│ [2.0, 1.0, 5.0] ┆ 22   ┆ 3    ┆ Ti     │
└─────────────────┴──────┴──────┴────────┘\
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

    #frame1.concat(frame2).assert_equal(Atoms({
    #    'x': [0., 1., 2., 3., 4., 5.],
    #    'y': [1., 1., 1., 1., 1., 1.],
    #    'z': [0., 2., 5., 0., 2., 5.],
    #    'elem': [1, 5, 22, 1, 5, 22],
    #}))

    Atoms.concat((frame2, frame1)).assert_equal(Atoms({
        'x': [3., 4., 5., 0., 1., 2.],
        'y': [1., 1., 1., 1., 1., 1.],
        'z': [0., 2., 5., 0., 2., 5.],
        'elem': [1, 5, 22, 1, 5, 22],
    }))


def test_add_atom():
    atoms = Atoms({
        'x': [0., 1.],
        'y': [1., 2.],
        'z': [0., 3.],
        'elem': [1, 5],
        'type': [1, 2],
    })

    atoms = atoms.add_atom('Fe2+', [4., 5., 6.], type=3)
    atoms = atoms.add_atom(8, 7., 8., 9., type=4)

    atoms.assert_equal(Atoms({
        'x': [0., 1., 4., 7.],
        'y': [1., 2., 5., 8.],
        'z': [0., 3., 6., 9.],
        'elem': [1, 5, 26, 8],
        'symbol': ['H', 'B', 'Fe2+', 'O'],
        'type': [1, 2, 3, 4],
    }))

    with pytest.raises(ValueError, match="Must specify vector of positions or x, y, & z."):
        atoms.add_atom(1, 4., 5.)  # type: ignore

    with pytest.raises(TypeError, match="Failed to cast 'Atoms' with schema '.*' to schema '.*'"):
        atoms.add_atom(1, [4., 5., 6.])


def test_with_props():
    atoms = Atoms({
        'x': [0., 1.],
        'y': [1., 2.],
        'z': [0., 3.],
        'elem': [1, 5],
        'symbol': ['H', 'B'],
        'type': [1, 2],
        'prop1': [1., 2.],
    })

    atoms.select_props(polars.col('prop1'), prop2=polars.Series([2., 3.])).assert_equal(Atoms({
        'x': [0., 1.],
        'y': [1., 2.],
        'z': [0., 3.],
        'elem': [1, 5],
        'symbol': ['H', 'B'],
        'prop1': [1., 2.],
        'prop2': [2., 3.]
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

    assert_array_equal(frame.coords(), expected)
    assert_array_equal(frame.coords(polars.col('coords').arr.get(0) < 0.), expected[expected[:, 0] < 0.])
    assert_array_equal(frame.coords(polars.col('elem') == 22), expected[2:])
    assert_array_equal(frame.coords([False, True, False, True]), expected[[1, 3]])

    assert_array_equal(frame.coords(frame.pos([1., 1., 2.])), [[1., 1., 2.]])
    assert_array_equal(frame.filter(frame.pos([1., 1., 2.])).get_column('elem').to_numpy(), [5])
    assert_array_equal(frame.coords(frame.pos(y=1.)), [[0., 1., 0.], [1., 1., 2.], [-1., 1., 5.]])


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
