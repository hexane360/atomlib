from pathlib import Path

import numpy
import polars
import pytest

from structlib import AtomCollection, AtomCell, Atoms
from structlib.transform import LinearTransform3D
from structlib.io import *


PATH = Path(__file__).absolute().parent


def xyz_expected(s: AtomCollection):
    assert isinstance(s, AtomCell)

    assert s.cell.cell_angle == pytest.approx([numpy.pi/2] * 3)
    assert s.cell.cell_size == pytest.approx([5.44] * 3)

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

    with open(path) as f:
        s = read(f)
    xyz_expected(s)

    s = read_xyz(path)
    xyz_expected(s)

    s = read(path)
    xyz_expected(s)

    s = AtomCollection.read(path)
    xyz_expected(s)

    s = AtomCollection.read_xyz(path)
    xyz_expected(s)


def test_xyz_hex():
    path = PATH / 'AlN.xyz'
    s = read_xyz(path)

    a = 3.13; c = 5.02
    ortho = LinearTransform3D([
        [a, -a*numpy.cos(numpy.pi/3), 0.],
        [0., a*numpy.sin(numpy.pi/3), 0.],
        [0., 0., c],
    ])
    print(s)
    expected = AtomCell.from_ortho(Atoms({
        'symbol': ['Al', 'Al', 'N', 'N'],
        'x': [   1.565,      0.0,    1.565,      0.0],
        'y': [0.903146, 1.806291, 0.903146, 1.806291],
        'z': [2.504900, 5.013377, 4.418497, 1.910020],
    }), ortho, frame='local')

    expected.assert_equal(s)

def cfg_expected(s: AtomCollection):
    assert isinstance(s, AtomCell)

    assert s.cell.cell_angle == pytest.approx([numpy.pi/2] * 3)
    assert s.cell.cell_size == pytest.approx([5.44] * 3)

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


def test_cfg_hex():
    path = PATH / 'test_hex.cfg'
    s = read_cfg(path)

    assert s.cell.cell_angle == pytest.approx(numpy.pi/180. * numpy.array([90., 90., 120.]))
    assert s.cell.cell_size == pytest.approx([3.13, 3.13, 5.02])
    s.get_atoms('local').assert_equal(Atoms({
        'elem': [13, 7, 13, 7],
        'symbol': ['Al', 'N', 'Al', 'N'],
        'x': [0.0, 0.0, 1.565, 1.565],
        'y': [1.8071, 1.8071, 0.903553, 0.903553],
        'z': [5.01642, 1.91118, 2.50642, 4.42118],
        'v_x': [0.0, 0.0, 0.0, 0.0],
        'v_y': [0.0, 0.0, 0.0, 0.0],
        'v_z': [0.0, 0.0, 0.0, 0.0],
        'mass': [26.9815, 14.0067, 26.9815, 14.0067],
    }))


def cif_expected(s: AtomCollection):
    assert isinstance(s, AtomCell)

    assert s.cell.cell_size == pytest.approx([22.75298600, 9.79283000, 5.65716000])
    assert s.cell.cell_angle == pytest.approx(numpy.full(3, numpy.pi/2.))
    assert s.is_orthogonal()

    assert len(s.atoms.filter(polars.col('symbol') == 'Al')) == 21
    assert len(s.atoms.filter(polars.col('symbol') == 'O')) == 34
    assert len(s.atoms.filter(polars.col('symbol') == 'Ag')) == 4


def test_cif(caplog: pytest.LogCaptureFixture):
    path = PATH / 'test.cif'

    #caplog.set_level(logging.DEBUG)

    with open(path, 'rb') as f:
        s = read_cif(f)  # type: ignore
        #print(s.atoms.filter(polars.col('symbol') == 'O').to_csv(sep='\t'))
        cif_expected(s)

    s = read_cif(path)
    cif_expected(s)

    s = read(path)
    cif_expected(s)

    s = AtomCollection.read(path)
    cif_expected(s)

    s = AtomCollection.read_cif(path)
    cif_expected(s)
