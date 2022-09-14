from pathlib import Path

import numpy
import polars
import pytest

from structlib import AtomCollection, AtomCell
from structlib.io import *


PATH = Path(__file__).absolute().parent


def xyz_expected(s: AtomCollection):
    assert isinstance(s, AtomCell)

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

    s = AtomCollection.read(path)
    xyz_expected(s)

    s = AtomCollection.read_xyz(path)
    xyz_expected(s)


def cfg_expected(s: AtomCollection):
    assert isinstance(s, AtomCell)

    assert s.cell_angle == pytest.approx([numpy.pi/2] * 3)
    assert s.cell_size == pytest.approx([5.44] * 3)

    assert list(s.atoms['elem']) == [14] * 4
    assert list(s.atoms['symbol']) == ['Si'] * 4
    assert list(s.atoms['x']) == pytest.approx([0.00, 0.00, 2.72, 2.72])
    assert list(s.atoms['y']) == pytest.approx([0.00, 2.72, 0.00, 2.72])
    assert list(s.atoms['z']) == pytest.approx([0.00, 2.72, 2.72, 0.00])
    assert list(s.atoms['v_x']) == pytest.approx([  5.44, 0.00, 0.00, 0.00])
    assert list(s.atoms['v_y']) == pytest.approx([-10.88, 0.00, 0.00, 0.00])
    assert list(s.atoms['v_z']) == pytest.approx([  5.44, 0.00, 0.00, 0.00])


def test_cfg():
    path = PATH / 'test.cfg'

    with open(path) as f:
        s = read_cfg(f)
        cfg_expected(s)

    s = read_cfg(path)
    cfg_expected(s)

    s = read(path)
    cfg_expected(s)

    s = AtomCollection.read(path)
    cfg_expected(s)

    s = AtomCollection.read_cfg(path)
    cfg_expected(s)