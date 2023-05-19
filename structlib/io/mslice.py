"""
IO support for the pyMultislicer XML file format.

Writes mslice files with the help of a user-supplied template.
"""

from __future__ import annotations

from xml.etree import ElementTree as et
import builtins
from copy import deepcopy
from pathlib import Path
import typing as t

import numpy
from numpy.typing import ArrayLike
import polars

from ..util import FileOrPath, open_file
from ..atoms import Atoms
from ..cell import Cell
from ..atomcell import HasAtomCell, AtomCell
from ..transform import AffineTransform3D, LinearTransform3D


MSliceFile = t.Union[et.ElementTree, FileOrPath]


DEFAULT_TEMPLATE_PATH: Path = Path(__file__).parents[2] / 'data' / 'template.mslice'
DEFAULT_TEMPLATE: t.Optional[et.ElementTree] = None


def default_template() -> et.ElementTree:
    global DEFAULT_TEMPLATE

    if DEFAULT_TEMPLATE is None:
        with open(DEFAULT_TEMPLATE_PATH, 'r') as f:
            DEFAULT_TEMPLATE = et.parse(f)  # type: ignore

    return deepcopy(DEFAULT_TEMPLATE)


def convert_xml_value(val, ty):
    """Convert an XML value `val` to a Python type determined by the XML type name `ty`."""
    if ty == 'string':
        ty = 'str'
    elif ty == 'int16' or ty == 'int32':
        val = val.split('.')[0]
        ty = 'int'
    elif ty == 'bool':
        val = int(val)

    return getattr(builtins, ty)(val)


def parse_xml_object(obj) -> t.Dict[str, t.Any]:
    """Parse the attributes of a passed XML object."""
    params = {}
    for attr in obj:
        if attr.tag == 'attribute':
            params[attr.attrib['name']] = convert_xml_value(attr.text, attr.attrib['type'])
        elif attr.tag == 'relationship':
            # todo give this a better API
            if 'idrefs' in attr.attrib:
                params[f"{attr.attrib['name']}ID"] = attr.attrib['idrefs']
    return params


def find_xml_object(xml, typename) -> t.Dict[str, t.Any]:
    """Find and parse XML objects named `typename`, flattening them into a single Dict."""
    params = {}
    for obj in xml.findall(f".//*[@type='{typename}']"):
        params.update(parse_xml_object(obj))
    return params


def find_xml_object_list(xml, typename) -> t.List[t.Any]:
    """Find and parse a list of XML objects named `typename`."""
    return [parse_xml_object(obj) for obj in xml.findall(f".//*[@type='{typename}']")]


def find_xml_object_dict(xml, typename, key="id") -> t.Dict[str, t.Any]:
    """Find and parse XML objects named `typename`, combining them into a dict."""
    return {
        obj.attrib[key]: parse_xml_object(obj)
        for obj in xml.findall(f".//*[@type='{typename}']")
    }


def read_mslice(path: MSliceFile) -> AtomCell:
    if isinstance(path, et.ElementTree):
        tree = path
    else:
        with open_file(path, 'r') as t:
            tree = et.parse(t)

    xml = tree.getroot()

    structure = find_xml_object(xml, "STRUCTURE")
    structure_atoms = find_xml_object_list(xml, "STRUCTUREATOM")

    n_cells = tuple(structure.get(k, 1) for k in ('repeata', 'repeatb', 'repeatc'))
    cell_size = tuple(structure[k] for k in ('aparam', 'bparam', 'cparam'))

    atoms = Atoms(
        polars.from_dicts(structure_atoms, schema={
            'atomicnumber': polars.UInt8,
            'x': polars.Float64, 'y': polars.Float64, 'z': polars.Float64,
            'wobble': polars.Float64, 'fracoccupancy': polars.Float64,
        })
        .rename({'atomicnumber': 'elem', 'fracoccupancy': 'frac_occupancy'}) \
        .with_columns((3. * polars.col('wobble')**2).alias('wobble'))
    )
    cell = Cell.from_ortho(LinearTransform3D.scale(cell_size), n_cells, [True, True, False])

    return AtomCell(atoms, cell, frame='cell_frac')


def write_mslice(cell: HasAtomCell, f: FileOrPath, template: t.Optional[MSliceFile] = None, *,
                 slice_thickness: t.Optional[float] = None,
                 scan_points: t.Optional[ArrayLike] = None,
                 scan_extent: t.Optional[ArrayLike] = None):
    if not isinstance(cell, HasAtomCell):
        raise TypeError("mslice format requires an AtomCell.")

    if not cell.is_orthogonal_in_local():
        raise ValueError("mslice requires an orthogonal AtomCell.")

    # get atoms in local frame (which we verified aligns with the cell's axes)
    # then scale into fractional coordinates
    bbox = cell.bbox()
    cell_size = bbox.size
    atoms = cell.get_atoms('local') \
        .transform(AffineTransform3D.translate(bbox.min).scale(cell_size).inverse()) \
        .with_wobble().with_occupancy()

    if template is None:
        out = default_template()
    elif not isinstance(template, et.ElementTree):
        with open_file(template, 'r') as t:
            out = et.parse(t)
    else:
        out = deepcopy(template)

    # TODO clean up this code
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
        (nx, ny) = numpy.broadcast_to(scan_points, 2,).astype(int)
        set_attr(params, 'numscanx', 'int16', str(nx))
        set_attr(params, 'numscany', 'int16', str(ny))

    if scan_extent is not None:
        (finx, finy) = numpy.broadcast_to(scan_extent, 2,).astype(float)
        # flipped
        set_attr(params, 'intx', 'float', f"{1.0-float(finx):.8f}")
        set_attr(params, 'inty', 'float', f"{1.0-float(finy):.8f}")
        set_attr(params, 'finx', 'float', "1.0")
        set_attr(params, 'finy', 'float', "1.0")

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

    with open_file(f, 'w') as f:
        # hack to specify doctype of output
        f.write("""\
<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<!DOCTYPE database SYSTEM "file:///System/Library/DTDs/CoreData.dtd">

""")
        out.write(f, encoding='unicode', xml_declaration=False, short_empty_elements=False)
        f.write('\n')


def _atom_elem(i: int, atomic_number: int, x: float, y: float, z: float, wobble: float = 0., frac_occupancy: float = 1.) -> et.Element:
    return et.XML(f"""\
<object type="STRUCTUREATOM" id="atom{i}">
    <attribute name="x" type="float">{x:.8f}</attribute>
    <attribute name="y" type="float">{y:.8f}</attribute>
    <attribute name="z" type="float">{z:.8f}</attribute>
    <attribute name="wobble" type="float">{wobble:.4f}</attribute>
    <attribute name="fracoccupancy" type="float">{frac_occupancy:.4f}</attribute>
    <attribute name="atomicnumber" type="int16">{atomic_number}</attribute>
</object>""")
