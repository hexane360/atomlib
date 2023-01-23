from io import StringIO
import logging

from pytest import approx
import polars

from .xsf import XSF, XSFParser
from .. import SimpleAtoms, Atoms


def test_xsf_molecule():
    logging.basicConfig(level=logging.DEBUG)
    s = """
        # start comment
    MOlECuLE
        # blank

    AtOmS
     6   0.0000 1.0000 0.5000
     12  1.0000 1.0000 0.5000
     80  1.0000 1.0000 0.5000"""

    xsf = XSF.from_file(StringIO(s))

    assert xsf.periodicity == 'molecule'
    assert list(xsf.atoms['elem']) == [6, 12, 80]  # type: ignore
    assert list(xsf.atoms['x']) == approx([0.0, 1.0, 1.0])  # type: ignore
    assert list(xsf.atoms['y']) == approx([1.0, 1.0, 1.0])  # type: ignore
    assert list(xsf.atoms['z']) == approx([0.5, 0.5, 0.5])  # type: ignore


def test_xsf_simple_write():
    atoms = SimpleAtoms(Atoms({
        'x': [1., 2., 3.],
        'y': [4., 5., 6.],
        'z': [7., 8., 9.],
        'elem': [12, 6, 34],
    }))
    buf = StringIO()
    XSF.from_atoms(atoms).write(buf)

    assert buf.getvalue() == """\
MOLECULE

ATOMS
12    1.000000    4.000000    7.000000
 6    2.000000    5.000000    8.000000
34    3.000000    6.000000    9.000000

"""

# TODO test sandwich sections (BEGIN_* / END_*)