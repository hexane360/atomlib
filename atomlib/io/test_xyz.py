from io import StringIO, BytesIO

import re
import pytest

from .xyz import XYZ, ExtXYZParser


def test_extended_xyz():
    comment = """
Lattice="5.44 0.0 0.0 0.0 5.44 0.0 0.0 0.0 5.44" Properties=species:S:1:pos:R:3 Time=0.0 "multi word key"=5.0
"""
    parser = ExtXYZParser(comment)
    assert parser.parse() == {
        'Lattice': "5.44 0.0 0.0 0.0 5.44 0.0 0.0 0.0 5.44",
        'Properties': "species:S:1:pos:R:3",
        'Time': "0.0",
        'multi word key': "5.0",
    }


def test_xyz_invalid():
    xyz_in = \
b"""4

Si      1.36     4.08     4.08
14      1.36     4.08     4.08
O       1.36     4.08     4.08
255     1.36     4.08     4.08
"""
    with pytest.raises(ValueError, match="Invalid atomic number 255"):
        XYZ.from_file(BytesIO(xyz_in))

    xyz_in = \
b"""4

Si      1.36     4.08     4.08
14      1.36     4.08     4.08
Ay      1.36     4.08     4.08
O       1.36     4.08     4.08
"""
    with pytest.raises(ValueError, match=re.escape("Invalid element symbol(s) 'Ay'")):
        print(XYZ.from_file(BytesIO(xyz_in)))


def test_xyz_write():
    xyz_in = \
"""2

Si        1.36000000      4.08000000      4.08000000
14        0.36000000      2.08000000     -1.08000000
"""
    xyz = XYZ.from_file(StringIO(xyz_in))

    xyz_out = StringIO()
    xyz.write(xyz_out)

    assert xyz_out.getvalue() == \
"""2

Si  1.36000000  4.08000000  4.08000000
Si  0.36000000  2.08000000 -1.08000000
"""
