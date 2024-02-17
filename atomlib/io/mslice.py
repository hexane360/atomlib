"""
IO support for the pyMultislicer XML file format.

Writes mslice files with the help of a user-supplied template.
"""

from __future__ import annotations

from lxml import etree as et  # type: ignore
#from xml.etree import ElementTree as et
import builtins
from copy import deepcopy
from warnings import warn
import typing as t

from importlib_resources import files
import numpy
from numpy.typing import ArrayLike
import polars

from ..util import FileOrPath, open_file, open_file_binary, BinaryFileOrPath
from ..atoms import Atoms
from ..cell import Cell
from ..atomcell import HasAtomCell, AtomCell
from ..transform import AffineTransform3D, LinearTransform3D


ElementTree = et._ElementTree
Element = et._Element
MSliceFile = t.Union[ElementTree, FileOrPath]


DEFAULT_TEMPLATE_PATH = files('atomlib.data') / 'template.mslice'
DEFAULT_TEMPLATE: t.Optional[ElementTree] = None


def default_template() -> ElementTree:
    global DEFAULT_TEMPLATE

    if DEFAULT_TEMPLATE is None:
        with DEFAULT_TEMPLATE_PATH.open('r') as f:  # type: ignore
            DEFAULT_TEMPLATE = t.cast(ElementTree, et.parse(f, None))

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


def parse_xml_object(obj: Element) -> t.Dict[str, t.Any]:
    """Parse the attributes of a passed XML object."""
    params = {}
    for attr in t.cast(t.Iterator[Element], obj.iter(None)):
        if attr.tag == 'attribute':
            params[attr.attrib['name']] = convert_xml_value(attr.text, attr.attrib['type'])
        elif attr.tag == 'relationship':
            # todo give this a better API
            if 'idrefs' in attr.attrib:
                params[f"{attr.attrib['name']}ID"] = attr.attrib['idrefs']
    return params


def find_xml_object(xml: Element, typename: str) -> t.Dict[str, t.Any]:
    """Find and parse XML objects named `typename`, flattening them into a single Dict."""
    params = {}
    for obj in xml.findall(f".//*[@type='{typename}']", None):
        params.update(parse_xml_object(obj))
    return params


def find_xml_object_list(xml: Element, typename: str) -> t.List[t.Any]:
    """Find and parse a list of XML objects named `typename`."""
    return [parse_xml_object(obj) for obj in xml.findall(f".//*[@type='{typename}']", None)]


def find_xml_object_dict(xml: Element, typename: str, key: str = "id") -> t.Dict[str, t.Any]:
    """Find and parse XML objects named `typename`, combining them into a dict."""
    return {
        obj.attrib[key]: parse_xml_object(obj)
        for obj in xml.findall(f".//*[@type='{typename}']", None)
    }


def read_mslice(path: MSliceFile) -> AtomCell:
    tree: ElementTree
    if isinstance(path, ElementTree):
        tree = path
    else:
        with open_file(path, 'r') as temp:
            tree = et.parse(temp, None)

    xml: Element = tree.getroot()

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
        # 1d sigma -> <u^2>
        .with_columns((3. * polars.col('wobble')**2).alias('wobble'))
    )
    cell = Cell.from_ortho(LinearTransform3D.scale(cell_size), n_cells, [True, True, False])

    return AtomCell(atoms, cell, frame='cell_frac')


def write_mslice(cell: HasAtomCell, f: BinaryFileOrPath, template: t.Optional[MSliceFile] = None, *,
                 slice_thickness: t.Optional[float] = None,  # angstrom
                 scan_points: t.Optional[ArrayLike] = None,
                 scan_extent: t.Optional[ArrayLike] = None,
                 noise_sigma: t.Optional[float] = None,  # angstrom
                 conv_angle: t.Optional[float] = None,  # mrad
                 energy: t.Optional[float] = None,  # keV
                 defocus: t.Optional[float] = None,  # angstrom
                 tilt: t.Optional[t.Tuple[float, float]] = None,  # (mrad, mrad)
                 tds: t.Optional[bool] = None,
                 n_cells: t.Optional[ArrayLike] = None):
    """
    Write a structure to an mslice file. The structure must be orthogonal and aligned
    with the local coordinate system. It should be periodic in X and Y.

    ``template`` may be a file, path, or ElementTree containing an existing mslice file.
    Its structure will be modified to make the final output. If not specified, a default
    template will be used.

    Additional options modify simulation properties. If an option is not specified, the
    template's properties are used.
    """
    #if not issubclass(type(cell), HasAtomCell):
    #    raise TypeError("mslice format requires an AtomCell.")

    if not cell.is_orthogonal_in_local():
        raise ValueError("mslice requires an orthogonal AtomCell.")

    if not numpy.all(cell.pbc[:2]):
        warn("AtomCell may not be periodic", UserWarning, stacklevel=2)

    box_size = cell._box_size_in_local()

    # get atoms in local frame (which we verified aligns with the cell's axes)
    # then scale into fractional coordinates
    atoms = cell.get_atoms('linear') \
        .transform(AffineTransform3D.scale(1/box_size)) \
        .with_wobble().with_occupancy()

    out: ElementTree
    if template is None:
        out = default_template()
    elif not isinstance(template, ElementTree):
        with open_file(template, 'r') as temp:
            out = et.parse(temp, None)
    else:
        out = deepcopy(template)

    # TODO clean up this code
    db: t.Optional[Element] = out.getroot() if out.getroot().tag == 'database' else out.find("./database", None)
    if db is None:
        raise ValueError("Couldn't find 'database' tag in template.")

    struct = db.find(".//object[@type='STRUCTURE']", None)
    if struct is None:
        raise ValueError("Couldn't find STRUCTURE object in template.")

    params = db.find(".//object[@type='SIMPARAMETERS']", None)
    if params is None:
        raise ValueError("Couldn't find SIMPARAMETERS object in template.")

    microscope = db.find(".//object[@type='MICROSCOPE']", None)
    if microscope is None:
        raise ValueError("Couldn't find MICROSCOPE object in template.")

    scan = db.find(".//object[@type='SCAN']", None)
    aberrations = db.findall(".//object[@type='ABERRATION']", None)

    def set_attr(struct: Element, name: str, type: str, val: str):
        node = t.cast(t.Optional[Element], struct.find(f".//attribute[@name='{name}']", None))
        if node is None:
            node = t.cast(Element, et.Element('attribute', dict(name=name, type=type), None))
            struct.append(node)
        else:
            node.attrib['type'] = type
        node.text = val  # type: ignore

    def parse_xml_object(obj: Element) -> t.Dict[str, t.Any]:
        """Parse the attributes of a passed XML object."""
        params = {}
        for attr in obj.iterchildren(None):
            if attr.tag == 'attribute':
                params[attr.attrib['name']] = convert_xml_value(attr.text, attr.attrib['type'])
            elif attr.tag == 'relationship':
                # todo give this a better API
                if 'idrefs' in attr.attrib:
                    params[f"{attr.attrib['name']}ID"] = attr.attrib['idrefs']
        return params

    # TODO how to store atoms in unexploded form
    (n_a, n_b, n_c) = map(str, (1, 1, 1) if n_cells is None else numpy.asarray(n_cells).astype(int))
    set_attr(struct, 'repeata', 'int16', n_a)
    set_attr(struct, 'repeatb', 'int16', n_b)
    set_attr(struct, 'repeatc', 'int16', n_c)

    (a, b, c) = map(lambda v: f"{v:.8g}", box_size)
    set_attr(struct, 'aparam', 'float', a)
    set_attr(struct, 'bparam', 'float', b)
    set_attr(struct, 'cparam', 'float', c)

    if tilt is not None:
        (tiltx, tilty) = tilt
        set_attr(struct, 'tiltx', 'float', f"{tiltx:.4g}")
        set_attr(struct, 'tilty', 'float', f"{tilty:.4g}")

    if slice_thickness is not None:
        set_attr(params, 'slicethickness', 'float', f"{float(slice_thickness):.8g}")
    if tds is not None:
        set_attr(params, 'includetds', 'bool', str(int(bool(tds))))
    if conv_angle is not None:
        set_attr(microscope, 'aperture', 'float', f"{float(conv_angle):.8g}")
    if energy is not None:
        set_attr(microscope, 'kv', 'float', f"{float(energy):.8g}")
    if noise_sigma is not None:
        if scan is None:
            raise ValueError("New scan specification required for 'noise_sigma'.")
        set_attr(scan, 'noise_sigma', 'float', f"{float(noise_sigma):.8g}")

    if defocus is not None:
        for aberration in aberrations:
            obj = parse_xml_object(aberration)
            if obj['n'] == 1 and obj['m'] == 0:
                set_attr(aberration, 'cnma', 'float', f"{float(defocus):.8g}")  # A, + is over
                set_attr(aberration, 'cnmb', 'float', "0.0")
                break
        else:
            raise ValueError("Couldn't find defocus aberration to modify.")

    if scan_points is not None:
        (nx, ny) = numpy.broadcast_to(scan_points, 2,).astype(int)
        if scan is not None:
            set_attr(scan, 'nx', 'int16', str(nx))
            set_attr(scan, 'ny', 'int16', str(ny))
        else:
            set_attr(params, 'numscanx', 'int16', str(nx))
            set_attr(params, 'numscany', 'int16', str(ny))

    if scan_extent is not None:
        scan_extent = numpy.asarray(scan_extent, dtype=float)
        try:
            if scan_extent.ndim < 2:
                if not scan_extent.shape == (4,):
                    scan_extent = numpy.broadcast_to(scan_extent, (2,))
                    scan_extent = numpy.stack(((0., 0.), scan_extent), axis=-1)
            else:
                scan_extent = numpy.broadcast_to(scan_extent, (2, 2))
        except ValueError as e:
            raise ValueError(f"Invalid scan_extent '{scan_extent}'. Expected an array of shape (2,), (4,), or (2, 2).") from e

        if scan is not None:
            names = ('x_i', 'x_f', 'y_i', 'y_f')
            elem = scan
        else:
            names = ('intx', 'finx', 'inty', 'finy')
            elem = params

        for (name, val) in zip(names, scan_extent.ravel()):
            set_attr(elem, name, 'float', f"{float(val):.8g}")

    # remove existing atoms
    for elem in db.findall("./object[@type='STRUCTUREATOM']", None):
        db.remove(elem)

    # <u^2> -> 1d sigma
    atoms = atoms.with_wobble((polars.col('wobble') / 3.).sqrt())
    rows = atoms.select(('elem', 'coords', 'wobble', 'frac_occupancy')).rows()
    for (i, (elem, (x, y, z), wobble, frac_occupancy)) in enumerate(rows):
        e = _atom_elem(i, elem, x, y, z, wobble, frac_occupancy)
        db.append(e)

    et.indent(db, space="    ", level=0)  # type: ignore

    with open_file_binary(f, 'w') as f:
        doctype = b"""<!DOCTYPE database SYSTEM "file:///System/Library/DTDs/CoreData.dtd">\n"""
        out.write(f, encoding='UTF-8', xml_declaration=True, standalone=True, doctype=doctype)  # type: ignore
        f.write(b'\n')


def _atom_elem(i: int, atomic_number: int, x: float, y: float, z: float, wobble: float = 0., frac_occupancy: float = 1.) -> Element:
    return et.XML(f"""\
<object type="STRUCTUREATOM" id="atom{i}">
    <attribute name="x" type="float">{x:.8f}</attribute>
    <attribute name="y" type="float">{y:.8f}</attribute>
    <attribute name="z" type="float">{z:.8f}</attribute>
    <attribute name="wobble" type="float">{wobble:.4f}</attribute>
    <attribute name="fracoccupancy" type="float">{frac_occupancy:.4f}</attribute>
    <attribute name="atomicnumber" type="int16">{atomic_number}</attribute>
</object>""", None)
