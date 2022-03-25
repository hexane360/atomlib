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