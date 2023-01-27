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
from ..core import AtomCell, OrthoCell
from ..transform import AffineTransform3D


MSliceTemplate = t.Union[et.ElementTree, FileOrPath]


DEFAULT_TEMPLATE_PATH: Path = Path(__file__).parents[2] / 'data' / 'template.mslice'
DEFAULT_TEMPLATE: t.Optional[et.ElementTree] = None


def default_template() -> et.ElementTree:
    global DEFAULT_TEMPLATE

    if DEFAULT_TEMPLATE is None:
        with open(DEFAULT_TEMPLATE_PATH, 'r') as f:
            DEFAULT_TEMPLATE = et.parse(f)  # type: ignore

    return deepcopy(DEFAULT_TEMPLATE)


def load_mslice(path: FileOrPath) -> OrthoCell:
    raise NotImplementedError()


def write_mslice(cell: AtomCell, path: FileOrPath, template: t.Optional[MSliceTemplate] = None, *,
                 slice_thickness: t.Optional[float] = None, scan_points: t.Optional[t.Tuple[int, int]] = None):
    if not isinstance(cell, AtomCell):
        raise TypeError("mslice format requires an AtomCell.")

    if not cell.cell.is_orthogonal_in_local():
        raise ValueError("mslice requires an orthogonal AtomCell.")

    # get atoms in local frame (which we verified aligns with the cell's axes)
    # then scale into fractional coordinates
    bbox = cell.cell.bbox()
    cell_size = bbox.size
    atoms = cell.get_atoms('local') \
        .transform(AffineTransform3D.translate(bbox.min).scale(cell_size).inverse()) \
        .with_wobble().with_occupancy()

    if template is None:
        out = default_template()
    elif not isinstance(template, et.ElementTree):
        with open_file(template, 'r') as f:
            out = et.parse(f)
    else:
        out = deepcopy(template)

    db = out.getroot() if out.getroot().tag == 'database' else out.find("./database")
    if db is None:
        raise ValueError("Couldn't find 'database' tag in template.")

    struct = db.find("./object[@type='STRUCTURE']")
    if struct is None:
        raise ValueError("Couldn't find STRUCTURE object in template.")

    params = db.find("./object[@type='SIMPARAMETERS']")
    if params is None:
        raise ValueError("Couldn't find SIMPARAMETERS object in template.")

    def set_attr(struct: et.Element, name: str, type: str, val: str):
        node = struct.find(f"./attribute[@name='{name}']")
        if node is None:
            node = et.Element('attribute', dict(name=name, type=type))
            struct.append(node)
        else:
            node.attrib['type'] = type
        node.text = val

    # TODO how to store atoms in unexploded form
    #(n_a, n_b, n_c) = map(str, atoms.n_cells)
    (n_a, n_b, n_c) = map(str, (1, 1, 1))
    set_attr(struct, 'repeata', 'int16', n_a)
    set_attr(struct, 'repeatb', 'int16', n_b)
    set_attr(struct, 'repeatc', 'int16', n_c)

    (a, b, c) = map(lambda v: f"{v:.8f}", cell_size)
    set_attr(struct, 'aparam', 'float', a)
    set_attr(struct, 'bparam', 'float', b)
    set_attr(struct, 'cparam', 'float', c)

    if slice_thickness is not None:
        set_attr(params, 'slicethickness', 'float', f"{float(slice_thickness):.8f}")

    if scan_points is not None:
        (nx, ny) = map(int, scan_points)
        set_attr(params, 'numscanx', 'int16', str(nx))
        set_attr(params, 'numscany', 'int16', str(ny))

    # remove existing atoms
    for elem in db.findall("./object[@type='STRUCTUREATOM']"):
        db.remove(elem)

    atoms = atoms.with_wobble((polars.col('wobble') / 3.).sqrt())  # pyMultislicer wants wobble in one dimension
    rows = atoms.select(('elem', 'x', 'y', 'z', 'wobble', 'frac_occupancy')).rows()
    for (i, (elem, x, y, z, wobble, frac_occupancy)) in enumerate(rows):
        e = _atom_elem(i, elem, x, y, z, wobble, frac_occupancy)
        #et.indent(e, space="    ", level=1)
        db.append(e)

    et.indent(db, space="    ", level=0)

    if (first := db.find("./object[@type='STRUCTUREATOM']")):
        pass

    with open_file(path, 'w') as f:
        # hack to specify doctype of output
        f.write("""\
<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<!DOCTYPE database SYSTEM "file:///System/Library/DTDs/CoreData.dtd">

""")
        out.write(f, encoding='unicode', xml_declaration=False, short_empty_elements=False)


def _atom_elem(i: int, atomic_number: int, x: float, y: float, z: float, wobble: float = 0., frac_occupancy: float = 1.) -> et.Element:
    return et.XML(f"""\
<object type="STRUCTUREATOM" id="atom{i}">
    <attribute name="z" type="float">{z:.8f}</attribute>
    <attribute name="y" type="float">{y:.8f}</attribute>
    <attribute name="x" type="float">{x:.8f}</attribute>
    <attribute name="wobble" type="float">{wobble:.4f}</attribute>
    <attribute name="fracoccupancy" type="float">{frac_occupancy:.4f}</attribute>
    <attribute name="atomicnumber" type="int16">{atomic_number}</attribute>
</object>""")
