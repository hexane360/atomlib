
import numpy
from numpy.testing import assert_array_almost_equal
import pytest

from . import AtomCell, Cell
from . import make
from .transform import AffineTransform3D


@pytest.fixture(scope='module')
def cu_conv() -> AtomCell:
    # translate to alternative basis
    return make.fcc('Cu', 3.615, cell='conv') \
        .transform_atoms(AffineTransform3D.translate(x=0.5), frame='cell_frac') \
        .wrap().round_near_zero()


@pytest.mark.parametrize(('in_coords', 'out_coords'), [
    # corner
    ([[0., 0., 0.]], [
        [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.],
        [0., 0., 1.], [1., 0., 1.], [0., 1., 1.], [1., 1., 1.],
    ]),
    # edge
    ([[0.5, 0., 0.]], [
        [0.5, 0., 0.], [0.5, 1., 0.], [0.5, 0., 1.], [0.5, 1., 1.],
    ]),
    # face, wrap + tolerancing
    ([[0.5, 0.5, 1.-1e-4], [0.5, 0.5, 1.1e-3]], [
        [0.5, 0.5, -1e-4], [0.5, 0.5, 1.1e-3], [0.5, 0.5, 1.-1e-4],
    ]),
])
def test_periodic_duplicate(in_coords, out_coords):
    cell = Cell.from_unit_cell([3.5, 4.0, 4.5], n_cells=(2, 3, 4), pbc=True)

    xs, ys, zs = numpy.array(in_coords).T
    elems = [29] * len(xs)

    atomcell = AtomCell({
        'x': xs, 'y': ys, 'z': zs, 'elem': elems
    }, cell=cell, frame='cell_box')

    duplicated = atomcell.periodic_duplicate(eps=1e-3)

    assert_array_almost_equal(
        duplicated.coords(frame='cell_box'), numpy.array(out_coords),
        decimal=6
    )


def test_periodic_duplicate_cu(cu_conv):
    duplicated = cu_conv.periodic_duplicate()

    assert_array_almost_equal(
        duplicated.coords(frame='cell_frac'), numpy.array([
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.0],

            [1.0, 0.0, 0.5],  # x
            [1.0, 0.5, 0.0],

            [0.5, 1.0, 0.0],  # y
            [0.0, 1.0, 0.5],
            [1.0, 1.0, 0.5],

            [0.5, 0.0, 1.0],  # z
            [0.0, 0.5, 1.0],
            [1.0, 0.5, 1.0],
            [0.5, 1.0, 1.0],
        ])
    )