"""
IO support for the pyMultislicer XML file format.

Writes mslice files with the help of a user-supplied template.
"""

from __future__ import annotations

from xml.etree import ElementTree as et
from copy import deepcopy
from pathlib import Path
import typing as t

import polars

from ..util import FileOrPath, open_file
from ..core import AtomCollection, AtomCell, OrthoCell


MSliceTemplate = t.Union[et.ElementTree, FileOrPath]


DEFAULT_TEMPLATE_PATH: Path = Path(__file__).parent / 'template.mslice'
DEFAULT_TEMPLATE: t.Optional[et.ElementTree] = None


def default_template() -> et.ElementTree:
    global DEFAULT_TEMPLATE

    if DEFAULT_TEMPLATE is not None:
        return deepcopy(DEFAULT_TEMPLATE)

    with open(DEFAULT_TEMPLATE_PATH, 'r') as f:
        DEFAULT_TEMPLATE = et.parse(f)
    return DEFAULT_TEMPLATE


def load_mslice(path: FileOrPath) -> OrthoCell:
    ...


def write_mslice(atoms: AtomCell, path: FileOrPath,
                 template: t.Optional[MSliceTemplate] = None):

    if not isinstance(atoms, AtomCell):
            raise TypeError("mslice format requires an AtomCell.")
    if not atoms.is_orthogonal():
        raise ValueError("AtomCell must be orthogonal.")

    if template is None:
        out = default_template()
    elif not isinstance(template, et.ElementTree):
        with open_file(template, 'r') as f:
            out = et.parse(f)
    else:
        out = deepcopy(template)

    db = out.find("./database")
    if db is None:
        raise ValueError("Couldn't find 'database' tag in template.")

    struct = db.find("./object[@type='STRUCTURE']")
    if struct is None:
        raise ValueError("Couldn't find STRUCTURE object in template.")

    def set_struct_attr(name: str, type: str, val: str):
        node = struct.find(f"./attribute[@name='{name}']")
        if node is None:
            node = et.Element('attribute', dict(name=name, type=type))
            struct.append(node)
        else:
            node.attrib['type'] = type
        node.text = val

    (n_a, n_b, n_c) = map(str, atoms.n_cells)
    set_struct_attr('repeata', 'int16', n_a)
    set_struct_attr('repeatb', 'int16', n_b)
    set_struct_attr('repeatc', 'int16', n_c)

    (a, b, c) = map(str, atoms.cell_size)
    set_struct_attr('aparam', 'float', a)
    set_struct_attr('bparam', 'float', b)
    set_struct_attr('cparam', 'float', c)

    # remove existing atoms
    for elem in db.findall("./object[@type='STRUCTUREATOM']"):
        db.remove(elem)

    frame = atoms.get_atoms('frac').with_wobble().with_occupancy()

    for (i, (elem, x, y, z, wobble, frac_occupancy)) in enumerate(atoms.get_atoms('frac').select(('elem', 'x', 'y', 'z', 'wobble', 'frac_occupancy'))):
        e = _atom_elem(i, elem, x, y, z, wobble, frac_occupancy)
        et.indent(e, space="    ", level=1)
        db.append(e)

    with open_file(path, 'w') as f:
        out.write(f)


def _atom_elem(i: int, atomic_number: int, x: float, y: float, z: float, wobble: float = 0., frac_occupancy=1.) -> et.Element:
	return et.XML(f"""
<object type="STRUCTUREATOM" id="atom{i}">
    <attribute name="z" type="float">{z}</attribute>
    <attribute name="y" type="float">{y}</attribute>
    <attribute name="x" type="float">{x}</attribute>
    <attribute name="wobble" type="float">{wobble}</attribute>
    <attribute name="fracoccupancy" type="float">{frac_occupancy}</attribute>
    <attribute name="atomicnumber" type="int16">{atomic_number}</attribute>
</object>""")
