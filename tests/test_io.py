from io import StringIO

import numpy
import polars
import pytest

from structlib import HasAtoms, AtomCell, Atoms
from structlib.transform import LinearTransform3D
from structlib.io import *

from structlib.testing import check_parse_structure, check_equals_file, INPUT_PATH, OUTPUT_PATH


def xyz_expected(s: HasAtoms):
    assert isinstance(s, AtomCell)

    assert s.cell.cell_angle == pytest.approx([numpy.pi/2] * 3)
    assert s.cell.cell_size == pytest.approx([5.44] * 3)

    assert list(s.atoms['elem']) == [14] * 8
    assert list(s.atoms['symbol']) == ['Si'] * 8
    assert list(s.atoms['x']) == pytest.approx([0.00, 1.36, 2.72, 4.08, 2.72, 4.08, 0.00, 1.36])
    assert list(s.atoms['y']) == pytest.approx([0.00, 1.36, 2.72, 4.08, 0.00, 1.36, 2.72, 4.08])
    assert list(s.atoms['z']) == pytest.approx([0.00, 1.36, 0.00, 1.36, 2.72, 4.08, 2.72, 4.08])


def test_xyz():
    path = INPUT_PATH / 'basic.xyz'

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

    s = AtomCell.read(path)
    xyz_expected(s)

    s = AtomCell.read_xyz(path)
    xyz_expected(s)


@check_equals_file('AlN.exyz')
def test_exyz_write(s: StringIO, aln: AtomCell):
    aln.write_xyz(s)

@check_equals_file('AlN.xyz')
def test_xyz_write(s: StringIO, aln: AtomCell):
    aln.write_xyz(s, fmt='xyz')


def cfg_expected(s: HasAtoms):
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
    path = INPUT_PATH / 'basic.cfg'

    with open(path) as f:
        s = read_cfg(f)
        cfg_expected(s)

    s = read_cfg(path)
    cfg_expected(s)

    s = read(path)
    cfg_expected(s)

    s = AtomCell.read(path)
    cfg_expected(s)

    s = AtomCell.read_cfg(path)
    cfg_expected(s)


def cif_expected(s: HasAtoms):
    assert isinstance(s, AtomCell)

    assert s.cell.cell_size == pytest.approx([22.75298600, 9.79283000, 5.65716000])
    assert s.cell.cell_angle == pytest.approx(numpy.full(3, numpy.pi/2.))
    assert s.is_orthogonal()

    assert len(s.atoms.filter(polars.col('symbol') == 'Al')) == 21
    assert len(s.atoms.filter(polars.col('symbol') == 'O')) == 34
    assert len(s.atoms.filter(polars.col('symbol') == 'Ag')) == 4


def test_cif(caplog: pytest.LogCaptureFixture):
    path = INPUT_PATH / 'basic.cif'

    #caplog.set_level(logging.DEBUG)

    with open(path, 'rb') as f:
        s = read_cif(f)  # type: ignore
        #print(s.atoms.filter(polars.col('symbol') == 'O').to_csv(sep='\t'))
        cif_expected(s)

    s = read_cif(path)
    cif_expected(s)

    s = read(path)
    cif_expected(s)

    s = AtomCell.read(path)
    cif_expected(s)

    s = AtomCell.read_cif(path)
    cif_expected(s)


@pytest.fixture
def aln_ortho() -> LinearTransform3D:
    a = 3.13; c = 5.02
    return LinearTransform3D([
        [a, -a*numpy.cos(numpy.pi/3), 0.],
        [0., a*numpy.sin(numpy.pi/3), 0.],
        [0., 0., c],
    ])


@pytest.fixture
def aln(aln_ortho) -> AtomCell:
    return AtomCell.from_ortho(Atoms({
        'symbol': ['Al', 'Al', 'N', 'N'],
        'x': [   1.565,      0.0,    1.565,      0.0],
        'y': [0.903146, 1.806291, 0.903146, 1.806291],
        'z': [2.504900, 5.013377, 4.418497, 1.910020],
    }), aln_ortho, frame='local')


@check_parse_structure('AlN.xyz')
def test_xyz_aln(aln):
    return aln


@check_parse_structure('AlN.xsf')
def test_xsf_aln(aln):
    return aln


@check_parse_structure('AlN.cif')
def test_cif_aln(aln):
    return aln


@check_parse_structure('label_only.cif')
def test_cif_aln_labelonly(aln):
    return AtomCell(
        aln.get_atoms('local')
            .with_column(polars.Series('label', ['Al(0)', 'Al(1)', 'N(0)', 'N(1)']))
            .select(['x', 'y', 'z', 'symbol', 'label', 'elem']),
        aln.cell, frame='local'
    )


@check_parse_structure('AlN.cfg')
def test_cfg_hex(aln_ortho):
    return AtomCell.from_ortho(Atoms({
        'elem': [13, 7, 13, 7],
        'symbol': ['Al', 'N', 'Al', 'N'],
        'x': [0.0, 0.0, 1.565, 1.565],
        'y': [1.8071, 1.8071, 0.903553, 0.903553],
        'z': [5.01642, 1.91118, 2.50642, 4.42118],
        'v_x': [0.0, 0.0, 0.0, 0.0],
        'v_y': [0.0, 0.0, 0.0, 0.0],
        'v_z': [0.0, 0.0, 0.0, 0.0],
        'mass': [26.9815, 14.0067, 26.9815, 14.0067],
    }), aln_ortho, frame='local')


@check_equals_file('AlN_roundtrip.cfg')
def test_cfg_roundtrip(s: StringIO, aln_ortho: LinearTransform3D):
    path = OUTPUT_PATH / 'AlN_roundtrip.cfg'
    cfg = CFG.from_file(path)
    cfg.write(s)


@check_equals_file('AlN_out.cfg')
def test_cfg_write_cell(s: StringIO, aln: AtomCell):
    aln.write_cfg(s)
