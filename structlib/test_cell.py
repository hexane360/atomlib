
import numpy
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest

from .transform import AffineTransform3D
from .cell import cell_to_ortho, ortho_to_cell, Cell


def test_cell_ortho_roundtrip():
    rng = numpy.random.default_rng()

    cell_size = rng.uniform(low=0.1, high=5.0, size=3)
    cell_angle = numpy.array((1.8, 1.5, 2.0))

    ortho = cell_to_ortho(cell_size, cell_angle)
    print(ortho)
    new_cell_size, new_cell_angle = ortho_to_cell(ortho)

    assert new_cell_size == pytest.approx(cell_size)
    assert new_cell_angle == pytest.approx(cell_angle)


@pytest.fixture(scope='module')
def mono_cell() -> Cell:
    """Make a Cell from a complex monoclinic cell, with an added transformation."""
    mono_ortho = cell_to_ortho([3., 4., 5.], [numpy.pi/2., numpy.pi/2., 1.8]) \
        .rotate([0., 0., 1.], numpy.pi/2.)

    return Cell.from_ortho(mono_ortho, n_cells=[2, 3, 5])


@pytest.fixture(scope='module')
def affine_cell() -> Cell:
    ortho = cell_to_ortho([3., 4., 5.]) \
        .rotate([0., 0., 1.], numpy.pi/2.) \
        .translate(-1., -1., -1.)

    return Cell.from_ortho(ortho, n_cells=[2, 3, 5])


def test_cell_from_ortho(mono_cell: Cell):
    assert mono_cell.affine.to_linear().is_orthogonal()
    #assert mono_cell.affine.inner == pytest.approx(AffineTransform3D.rotate([0., 0., 1.], numpy.pi/2.).inner)

    assert_array_almost_equal(mono_cell.ortho.inner, [
        [1., numpy.cos(1.8), 0.],
        [0., numpy.sin(1.8), 0.],
        [0.,             0., 1.],
    ])

    assert_array_equal(mono_cell.n_cells, [2, 3, 5])
    assert_array_almost_equal(mono_cell.cell_size, [3., 4., 5.])
    assert_array_almost_equal(mono_cell.ortho_size, [3., 4.*numpy.sin(1.8), 5.])


@pytest.mark.parametrize(('frame_in', 'frame_out', 'pts_in', 'pts_out'), [
    ('ortho', 'local', None, [[ 0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]),
    ('cell', 'ortho', None, [[1., 0., 0.], [numpy.cos(1.8), numpy.sin(1.8), 0.], [0., 0., 1.]]),
    ('cell', 'local', None, [[0., 1., 0.], [-numpy.sin(1.8), numpy.cos(1.8), 0.], [0., 0., 1.]]),
    ('cell_frac', 'ortho', None, [[3., 0., 0.], [4.*numpy.cos(1.8), 4.*numpy.sin(1.8), 0.], [0., 0., 5.]]),
    ('cell_box', 'ortho', None, [[2*3., 0., 0.], [3*4.*numpy.cos(1.8), 3*4.*numpy.sin(1.8), 0.], [0., 0., 5*5.]]),
    ('ortho_frac', 'ortho', None, numpy.diag([3., 4. * numpy.sin(1.8), 5.])),
    ('ortho_box', 'ortho', None, [2., 3., 5.] * numpy.diag([3., 4. * numpy.sin(1.8), 5.])),
    ('ortho', 'ortho', None, numpy.eye(3)),
    ('local', 'local', None, numpy.eye(3)),
])
def test_mono_cell(mono_cell: Cell, frame_in, frame_out, pts_in, pts_out):
    pts_in = numpy.eye(3) if pts_in is None else pts_in
    assert_array_almost_equal(mono_cell.get_transform(frame_in, frame_out).transform(pts_out), pts_in)
    assert_array_almost_equal(mono_cell.get_transform(frame_out, frame_in).transform(pts_in), pts_out)


@pytest.mark.parametrize(('frame_in', 'frame_out', 'pts_in', 'pts_out'), [
    ('linear', 'local', [[0., 0., 0.]], [[-1., -1., -1.]]),
    ('local', 'linear', [[0., 0., 0.]], [[1., 1., 1.]]),
])
def test_affine_cell(affine_cell: Cell, frame_in, frame_out, pts_in, pts_out):
    pts_in = numpy.eye(3) if pts_in is None else pts_in
    assert_array_almost_equal(affine_cell.get_transform(frame_in, frame_out).transform(pts_out), pts_in)
    assert_array_almost_equal(affine_cell.get_transform(frame_out, frame_in).transform(pts_in), pts_out)


def test_cell_in_local(affine_cell: Cell):
    assert affine_cell.is_orthogonal_in_local()

    assert_array_equal(affine_cell._n_cells_in_local(), [3, 2, 5])
    assert_array_almost_equal(affine_cell._cell_size_in_local(), [4., 3., 5.])


def test_transform_affine_cell():
    cell = Cell.from_unit_cell([2., 3., 5.]) \
        .repeat((9, 6, 3)) \
        .transform_cell(AffineTransform3D.translate([-3, -2, -1]), frame='cell_frac')

    print(f"cell affine: {cell.affine}")
    print(f"cell_box transform: {cell.get_transform('local', 'cell_box')}")

    assert_array_almost_equal(
        cell.corners('local'), [
            [-6., -6., -5.],
            [-6., -6., 10.],
            [-6., 12., -5.],
            [-6., 12., 10.],
            [12., -6., -5.],
            [12., -6., 10.],
            [12., 12., -5.],
            [12., 12., 10.],
        ]
    )

    new_cell = cell.transform_cell(AffineTransform3D.rotate_euler(z=numpy.pi/2.), frame='local')
    assert_array_almost_equal(
        new_cell.corners('local'), [
            [  6., -6., -5.],
            [  6., -6., 10.],
            [-12., -6., -5.],
            [-12., -6., 10.],
            [  6., 12., -5.],
            [  6., 12., 10.],
            [-12., 12., -5.],
            [-12., 12., 10.],
        ]
    )