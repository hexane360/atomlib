
import numpy
from pytest import approx

from .cell import cell_to_ortho, ortho_to_cell, Vec3


def test_cell_ortho_roundtrip():
    rng = numpy.random.default_rng()

    cell_size = rng.uniform(low=0.1, high=5.0, size=3).view(Vec3)
    cell_angle = Vec3.make((1.8, 1.5, 2.0))

    ortho = cell_to_ortho(cell_angle, cell_size)
    print(ortho)
    new_cell_angle, new_cell_size = ortho_to_cell(ortho)

    assert new_cell_size == approx(cell_size)
    assert new_cell_angle == approx(cell_angle)
