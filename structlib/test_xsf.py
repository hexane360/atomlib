from io import StringIO
import logging

from pytest import approx

from .xsf import XSF, XSFParser


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
    assert list(xsf.atoms['atomic_number']) == [6, 12, 80]  # type: ignore
    assert list(xsf.atoms['a']) == approx([0.0, 1.0, 1.0])  # type: ignore
    assert list(xsf.atoms['b']) == approx([1.0, 1.0, 1.0])  # type: ignore
    assert list(xsf.atoms['c']) == approx([0.5, 0.5, 0.5])  # type: ignore

# TODO test sandwich sections (BEGIN_* / END_*)