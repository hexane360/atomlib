
import pytest
import numpy

from . import AtomCell, Atoms
from .transform import LinearTransform3D


def test_core_bbox():
    cell = AtomCell.from_ortho(Atoms({
        'x': [0., 1., -2., 3.],
        'y': [0., 2., -3., 4.],
        'z': [0., 1., -1., 1.],
        'elem': [1, 16, 32, 48],
    }), ortho=LinearTransform3D())

    bbox = cell.bbox()
    assert bbox.min == pytest.approx([-2., -3., -1])
    assert bbox.max == pytest.approx([3., 4., 1.])

    bbox = cell.cell.bbox()
    assert bbox.min == pytest.approx([0., 0., 0.])
    assert bbox.max == pytest.approx([1., 1., 1.])

    corners = cell.cell.corners('cell_frac')
    assert corners == pytest.approx(numpy.array([
        [0., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 1., 1.],
        [1., 0., 0.],
        [1., 0., 1.],
        [1., 1., 0.],
        [1., 1., 1.],
    ]))


"""
def test_ortho_cell():
    cell = AtomCell.from_ortho(Atoms({
        'x': [0., 1., -2., 3.],
        'y': [0., 2., -3., 4.],
        'z': [0., 1., -1., 1.],
        'elem': [1, 16, 32, 48],
    }), ortho=LinearTransform3D())

    assert cell.is_orthogonal()

    ortho = cell.orthogonalize()

    assert ortho.is_orthogonal()

    with pytest.raises(ValueError, match="OrthoCell constructed with non-orthogonal angles"):
        OrthoCell.from_unit_cell(cell.atoms, cell_angle=[numpy.pi/2, numpy.pi/2, numpy.pi/3], cell_size=[2., 3., 5.])

    OrthoCell.from_ortho(cell.atoms, ortho=cell.cell.ortho.rotate([1, 1, 1], numpy.pi/8))
"""