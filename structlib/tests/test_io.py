from pathlib import Path

import numpy
import polars
import pytest

from structlib import AtomCollection, Lattice
from structlib.io import *


PATH = Path(__file__).absolute().parent


def xyz_expected(s: AtomCollection):
    assert isinstance(s, Lattice)

    assert s.cell_angle == pytest.approx([numpy.pi/2] * 3)
    assert s.cell_size == pytest.approx([5.44] * 3)

    assert list(s.atoms['elem']) == [14] * 8
    assert list(s.atoms['symbol']) == ['Si'] * 8
    assert list(s.atoms['x']) == pytest.approx([0.00, 1.36, 2.72, 4.08, 2.72, 4.08, 0.00, 1.36])
    assert list(s.atoms['y']) == pytest.approx([0.00, 1.36, 2.72, 4.08, 0.00, 1.36, 2.72, 4.08])
    assert list(s.atoms['z']) == pytest.approx([0.00, 1.36, 0.00, 1.36, 2.72, 4.08, 2.72, 4.08])


def test_xyz():
    path = PATH / 'test.xyz'

    with open(path) as f:
        s = read_xyz(f)
    xyz_expected(s)

    s = read_xyz(path)
    xyz_expected(s)

    s = read(path)
    xyz_expected(s)