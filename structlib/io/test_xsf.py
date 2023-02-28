from io import StringIO
import logging

import numpy
from numpy.testing import assert_array_equal

from .xsf import XSF
from .. import SimpleAtoms, Atoms
from ..cell import cell_to_ortho


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
    atoms = xsf.atoms
    assert atoms is not None
    assert_array_equal(atoms['elem'], [6, 12, 80])
    assert_array_equal(atoms['x'], [0.0, 1.0, 1.0])
    assert_array_equal(atoms['y'], [1.0, 1.0, 1.0])
    assert_array_equal(atoms['z'], [0.5, 0.5, 0.5])


def test_xsf_write_lattice():
    atoms = SimpleAtoms(Atoms({
        'x': [],
        'y': [],
        'z': [],
        'elem': [],
    }))

    buf = StringIO()
    xsf = XSF.from_atoms(atoms)
    xsf.periodicity = 'slab'
    xsf.primitive_cell = cell_to_ortho(
        [3.13, 3.13, 5.02], numpy.array([90., 90., 120.]) * numpy.pi/180.
    )
    xsf.write(buf)

    assert buf.getvalue() == """\
SLAB
PRIMVEC
   3.1300000   0.0000000   0.0000000
  -1.5650000   2.7106595   0.0000000
   0.0000000   0.0000000   5.0200000

ATOMS

"""


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
