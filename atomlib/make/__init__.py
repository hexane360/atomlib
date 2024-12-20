"""
Functions to create structures and cells.
"""

from __future__ import annotations

import logging
import string
import typing as t

from typing_extensions import TypeAlias
import numpy

from ..atomcell import Atoms, AtomCell, HasAtomCellT, IntoAtoms
from ..transform import LinearTransform3D, AffineTransform3D
from ..elem import get_elem, get_elems, get_mass
from ..types import ElemLike, ElemsLike, Num, VecLike
from ..cell import cell_to_ortho, Cell
from ..vec import reduce_vec, split_arr, to_vec3
from ..bbox import BBox3D
from ..util import proc_seed


CellType: TypeAlias = t.Literal['conv', 'prim', 'ortho']


def fcc(elem: ElemLike, a: Num, *, cell: CellType = 'conv', additional: t.Optional[IntoAtoms] = None) -> AtomCell:
    """
    Make a FCC lattice of the specified element, with the given cell.
    If `cell='conv'` (the default), return the conventional cell, four atoms with cubic cell symmetry.
    If `cell='prim'`, return the primitive cell, a single atom with rhombohedral cell symmetry.
    If `cell='ortho'`, return an orthogonal cell, two atoms in a cell of size `[a/sqrt(2), a/sqrt(2), a]`.

    If `additional` is specified, those atoms will be added to the lattice (in fractional coordinates).

    Args:
      elem: Element to add (e.g. `'Al'` or `13`)
      a: Lattice parameter (Angstrom)
      cell: Cell type to return ('conv', 'prim', or 'ortho')
      additional: Additional atoms to add to the structure.

    Returns:
      Periodic FCC unit cell
    """

    elems = [get_elem(elem)]
    cell = t.cast(CellType, str(cell).lower())

    if cell == 'prim':
        xs = ys = zs = [0.]
        ortho = LinearTransform3D(a / 2. * numpy.array([
            [0., 1., 1.],
            [1., 0., 1.],
            [1., 1., 0.],
        ]))
    elif cell == 'ortho':
        elems *= 2
        xs = ys = zs = [0., 0.5]
        b = a / numpy.sqrt(2)
        ortho = LinearTransform3D.scale(b, b, a)
    elif cell == 'conv':
        elems *= 4
        xs = [0., 0., 0.5, 0.5]
        ys = [0., 0.5, 0., 0.5]
        zs = [0., 0.5, 0.5, 0.]
        ortho = LinearTransform3D.scale(all=a)
    else:
        raise ValueError(f"Unknown cell type '{cell}'. Expected 'conv', 'prim', or 'ortho'.")

    frame = Atoms(dict(x=xs, y=ys, z=zs, elem=elems))
    if additional is not None:
        frame = Atoms.concat((frame, additional), how='vertical')

    return AtomCell.from_ortho(frame, ortho, frame='cell_frac')


def wurtzite(elems: ElemsLike, a: Num, c: t.Optional[Num] = None,
             d: t.Optional[Num] = None, *, cell: CellType = 'conv') -> AtomCell:
    r"""
    Create a wurzite lattice of the specified two elements, with the given cell.
    `a` and `c` are the hexagonal cell parameters. `d` is the fractional distance
    between the two sublattices.

    If `cell='prim'` or `cell='conv'` (the default), return a hexagonal unit cell.
    If `cell='ortho'`, return an orthogonal unit cell constructed from the hexagonal unit cell as
    $\hat{\mathbf{a}} = \mathbf{a}$, $\hat{\mathbf{b}} = \mathbf{a} + 2 \mathbf{b}$, $\hat{\mathbf{c}} = \mathbf{c}$.

    Args:
      elems: Elements to add (e.g. `'AlN'` or `('Al', 'N')`)
      a: Lattice parameter (Angstrom)
      c: Vertical lattice parameter (Angstrom)
      d: Vertical distance between the two sublattices (fractional)
      cell: Cell type to return ('conv', 'prim', or 'ortho')

    Returns:
      Periodic wurtzite unit cell
    """
    elems = [t[0] for t in get_elems(elems)]

    if len(elems) != 2:
        raise ValueError("Expected two elements.")

    # default to ideal c/a
    c_a = float(numpy.sqrt(8. / 3.) if c is None else c / a)

    d = 0.25 + 1 / (3 * c_a**2) if d is None else d
    if not 0 < d < 0.5:
        raise ValueError(f"Invalid 'd' parameter: {d}")

    cell = t.cast(CellType, str(cell).lower())
    if cell not in ('prim', 'conv', 'ortho'):
        raise ValueError(f"Unknown cell type '{cell}'. Expected 'conv', 'prim', or 'ortho'.")

    ortho = cell_to_ortho(
        a * numpy.array([1., 1., c_a]), 
        numpy.pi * numpy.array([1/2., 1/2., 2/3.])
    )
    xs = [2/3, 2/3, 1/3, 1/3]
    ys = [1/3, 1/3, 2/3, 2/3]
    #zs = [1. - d, 0., 0.5 - d, 0.5]
    zs = [0.5, 0.5 + d, 0., d]
    elems *= 2

    frame = Atoms(dict(x=xs, y=ys, z=zs, elem=elems))
    atoms = AtomCell.from_ortho(frame, ortho, frame='cell_frac')
    if cell == 'ortho':
        return _ortho_hexagonal(atoms)
    return atoms


def graphite(elem: t.Union[str, ElemLike, None] = None, a: t.Optional[Num] = None,
             c: t.Optional[Num] = None, *, cell: CellType = 'conv'):
    if elem is None:
        elem = 6
    else:
        elem = get_elem(elem)
        if elem != 6 and a is None or c is None:
            raise ValueError("'a' and 'c' must be specified for non-graphite elements.")

    if a is None:
        a = 2.47
    if c is None:
        c = 8.69

    cell = t.cast(CellType, str(cell).lower())
    if cell not in ('prim', 'conv', 'ortho'):
        raise ValueError(f"Unknown cell type '{cell}'. Expected 'conv', 'prim', or 'ortho'.")

    ortho = cell_to_ortho(
        numpy.array([a, a, c]), 
        numpy.pi * numpy.array([1/2., 1/2., 2/3.])
    )
    xs = [0., 2/3, 0., 1/3]
    ys = [0., 1/3, 0., 2/3]
    zs = [0., 0., 1/2, 1/2]
    elems = [elem] * 4

    frame = Atoms(dict(x=xs, y=ys, z=zs, elem=elems))
    atoms = AtomCell.from_ortho(frame, ortho, frame='cell_frac')

    if cell == 'ortho':
        return _ortho_hexagonal(atoms)
    return atoms


def rocksalt(elems: ElemsLike, a: Num, *,
             cell: CellType = 'conv') -> AtomCell:
    """
    Create a rock salt FCC structure AB. Returns the same cell types as `fcc`.

    Args:
      elems: Elements to add (e.g. `'NaCl'` or `('Na', 'Cl')`)
      a: Lattice parameter (Angstrom)
      cell: Cell type to return ('conv', 'prim', or 'ortho'). Returns
            the same cell types as `fcc`.

    Returns:
      Periodic rocksalt unit cell
    """
    elems = [t[0] for t in get_elems(elems)]

    if len(elems) != 2:
        raise ValueError("Expected two elements.")

    if cell == 'prim':
        additional: t.Dict[str, t.Any] = {
            'x': [-0.5],
            'y': [0.5],
            'z': [0.5],
            'elem': [elems[1]],
        }
    elif cell == 'ortho':
        additional: t.Dict[str, t.Any] = {
            'x': [0.5, 0.0],
            'y': [0.5, 0.0],
            'z': [0.0, 0.5],
            'elem': [elems[1]] * 2,
        }
    elif cell == 'conv':
        additional: t.Dict[str, t.Any] = {
            'x': [0.5, 0.0, 0.0, 0.5],
            'y': [0.0, 0.5, 0.0, 0.5],
            'z': [0.0, 0.0, 0.5, 0.5],
            'elem': [elems[1]] * 4,
        }
    else:
        raise ValueError(f"Unknown cell type '{cell}'. Expected 'conv', 'prim', or 'ortho'.")

    return fcc(elems[0], a, cell=cell, additional=additional)


def zincblende(elems: ElemsLike, a: Num, *,
               cell: CellType = 'conv') -> AtomCell:
    """
    Create a zinc-blende FCC structure AB.

    Args:
      elems: Elements to add (e.g. `'ZnS'` or `('Zn', 'S')`)
      a: Lattice parameter (Angstrom)
      cell: Cell type to return ('conv', 'prim', or 'ortho'). Returns
            the same cell types as `fcc`.

    Returns:
      Periodic zinc-blende unit cell
    """
    elems = [t[0] for t in get_elems(elems)]

    if len(elems) != 2:
        raise ValueError("Expected two elements.")

    if cell == 'prim':
        d = [0.25]
        additional: t.Dict[str, t.Any] = {
            'x': d,
            'y': d,
            'z': d,
            'elem': [elems[1]],
        }
    elif cell == 'ortho':
        additional: t.Dict[str, t.Any] = {
            'x': [0.5, 0.0],
            'y': [0.0, 0.5],
            'z': [0.25, 0.75],
            'elem': [elems[1]] * 2,
        }
    elif cell == 'conv':
        additional: t.Dict[str, t.Any] = {
            'x': [0.25, 0.25, 0.75, 0.75],
            'y': [0.25, 0.75, 0.25, 0.75],
            'z': [0.25, 0.75, 0.75, 0.25],
            'elem': [elems[1]] * 4,
        }
    else:
        raise ValueError(f"Unknown cell type '{cell}'. Expected 'conv', 'prim', or 'ortho'.")

    return fcc(elems[0], a, cell=cell, additional=additional)


@t.overload
def diamond(elem: None = None, a: t.Optional[Num] = None, *,
            cell: CellType = 'conv') -> AtomCell:
    ...

@t.overload
def diamond(elem: t.Optional[ElemLike], a: Num, *,
            cell: CellType = 'conv') -> AtomCell:
    ...

def diamond(elem: t.Optional[ElemLike] = None, a: t.Optional[Num] = None, *,
            cell: CellType = 'conv') -> AtomCell:
    """
    Create a diamond cubic FCC structure. `elem` and `a` can be left
    unspecified to return a diamond structure. Otherwise, both
    must be specified.

    Args:
      elem: Element to add (e.g. `'C'`)
      a: Lattice parameter (Angstrom)
      cell: Cell type to return ('conv', 'prim', or 'ortho'). Returns
            the same cell types as `fcc`.

    Returns:
      Periodic diamond cubic unit cell
    """
    if elem is None:
        elems = (6, 6)
    else:
        elem = get_elem(elem)
        elems = (elem, elem)

    if a is None:
        if elems == (6, 6):
            # diamond lattice parameter
            a = 3.567
        else:
            raise ValueError("Must specify lattice parameter 'a'.")

    return zincblende(elems, a, cell=cell)


def fluorite(elems: ElemsLike, a: Num, *,
             cell: CellType = 'conv') -> AtomCell:
    """
    Create a fluorite FCC structure $\\mathrm{AB_2}$. Returns the same cell types as `fcc`.

    Args:
      elems: Elements to add (e.g. `'CaF'` or `('Ca', 'F')`)
      a: Lattice parameter (Angstrom)
      cell: Cell type to return ('conv', 'prim', or 'ortho'). Returns
            the same cell types as `fcc`.

    Returns:
      Periodic fluorite unit cell
    """
    elems = [t[0] for t in get_elems(elems)]

    if len(elems) != 2:
        raise ValueError("Expected two elements.")

    if cell == 'prim':
        d = [0.25, 0.75]
        additional: t.Dict[str, t.Any] = {
            'x': d, 'y': d, 'z': d,
            'elem': [elems[1]] * 2,
        }
    elif cell == 'ortho':
        additional: t.Dict[str, t.Any] = {
            'x': [0.5, 0.5, 0.0, 0.0],
            'y': [0.0, 0.0, 0.5, 0.5],
            'z': [0.25, 0.75, 0.25, 0.75],
            'elem': [elems[1]] * 4,
        }
    elif cell == 'conv':
        additional: t.Dict[str, t.Any] = {
            'x': [0.25] * 4 + [0.75] * 4,
            'y': [0.25, 0.25, 0.75, 0.75, 0.25, 0.25, 0.75, 0.75],
            'z': [0.25, 0.75, 0.25, 0.75, 0.25, 0.75, 0.25, 0.75],
            'elem': [elems[1]] * 8,
        }
    else:
        raise ValueError(f"Unknown cell type '{cell}'. Expected 'conv', 'prim', or 'ortho'.")

    return fcc(elems[0], a, cell=cell, additional=additional)


@t.overload # CsCl
def cesium_chloride(elems: t.Literal['CsCl'] = 'CsCl', a: None = None, *,
                    d: None = None, cell: CellType = 'conv') -> AtomCell:
    ...

@t.overload # general, specify a
def cesium_chloride(elems: ElemsLike, a: Num, *,
                    d: None = None, cell: CellType = 'conv') -> AtomCell:
    ...

@t.overload # general, specify d
def cesium_chloride(elems: ElemsLike = 'CsCl', a: None = None, *,
                    d: Num, cell: CellType = 'conv') -> AtomCell:
    ...

def cesium_chloride(elems: ElemsLike = 'CsCl', a: t.Optional[Num] = None, *,
                    d: t.Optional[Num] = None, cell: CellType = 'conv') -> AtomCell:
    """
    Create a cesium chloride structure $\\mathrm{AB}$.
    CsCl is simple cubic, so all cell types are the same.

    Only one of `a` (lattice parameter) or `d` (bond distance) needs to be specified.

    Args:
      elems: Elements to add (e.g. `'CsCl'` or `('Cs', 'Cl')`)
      a: Lattice parameter (Angstrom)
      d: Nearest-neighbor bond distance (Angstrom)
      cell: Cell type to return ('conv', 'prim', or 'ortho').
            All are identical for this structure

    Returns:
      Periodic cesium chloride unit cell
    """
    elems = [t[0] for t in get_elems(elems)]

    if len(elems) != 2:
        raise ValueError("Expected two elements.")

    if a is not None and d is not None:
        raise ValueError("Both 'a' and 'd' cannot be specified.")

    if a is None:
        if d is not None:
            a_ = d * 2/numpy.sqrt(3)
        elif elems == [55, 17]:
            # CsCl lattice parameter
            a_ = 4.123
        else:
            raise ValueError("Must specify either 'a' or 'd' lattice parameter")
    else:
        a_ = a

    ortho = cell_to_ortho([a_] * 3)

    frame = Atoms(dict(x=[0., 0.5], y=[0., 0.5], z=[0., 0.5], elem=elems))
    return AtomCell.from_ortho(frame, ortho, frame='cell_frac')


def perovskite(elems: ElemsLike, cell_size: VecLike, *,
               cell: CellType = 'conv') -> AtomCell:
    """
    Create a perovskite structure $\\mathrm{ABX_3}$.

    `A` is placed at the origin and `B` is placed at the cell center.
    `cell_size` determines whether a cubic, tetragonal, or orthorhombic
    structure is created. For instance, `cell_size=3.` returns a cubic
    structure, while `cell_size=[3., 5.]` returns a tetragonal structure
    `a=3`, `c=5`.

    All cell types are the same for perovskite, so the `cell` parameter
    has no effect.

    Args:
      elems: Elements to add (e.g. `'CaTiO'` or `('Ca', 'Ti', 'O')`)
      cell_size: Lattice parameters (e.g. `3.0` (cubic), `[3.0, 5.0]`
                 (tetragonal), or `[3.0, 4.0, 5.0]` (orthorhombic)).
      cell: Cell type to return ('conv', 'prim', or 'ortho').
            All are identical for this structure

    Returns:
      Periodic perovskite unit cell.
    """
    elems = [t[0] for t in get_elems(elems)]

    if len(elems) != 3:
        raise ValueError("Expected three elements.")

    cell_size = numpy.atleast_1d(cell_size)
    if cell_size.squeeze().ndim > 1:
        raise ValueError("Expected a 1D vector")
    if len(cell_size) == 2:
        # tetragonal shortcut
        cell_size = numpy.array([cell_size[0], cell_size[0], cell_size[1]])

    if cell not in ('prim', 'ortho', 'conv'):
        raise ValueError(f"Unknown cell type '{cell}'. Expected 'conv', 'prim', or 'ortho'.")

    xs = [0., 0.5, 0., 0.5, 0.5]
    ys = [0., 0.5, 0.5, 0., 0.5]
    zs = [0., 0.5, 0.5, 0.5, 0.]
    elems = [elems[0], elems[1], *([elems[2]] * 3)]

    atoms = Atoms(dict(x=xs, y=ys, z=zs, elem=elems))
    return AtomCell(atoms, Cell.from_unit_cell(cell_size), frame='cell_frac')


def random(cell: t.Union[Cell, VecLike], elems: ElemsLike, density: float,
           seed: t.Optional[object] = None, **extra_cols: t.Any) -> AtomCell:
    """
    Make a random arrangement of atoms inside `cell`
    ([`Cell`][atomlib.cell.Cell] or cell_size vector).

    Args:
      elems: Elements to add (e.g. `'C'`, `6`, or `SiO2`)
      density: Mean mass density to target (g/cm^3)
      seed: Deterministic random seed to add (any object)
      extra_cols: Extra parameters to add to each atom

    Returns:
      A random arrangement of atoms
    """
    if not isinstance(cell, Cell):
        cell = Cell.from_unit_cell(cell, pbc=[True, True, True])

    elems = get_elems(elems)
    # normalize formula unit
    total_num = sum(elem[1] for elem in elems)
    elems = [(elem, num / total_num) for (elem, num) in elems]

    total_mass = sum(get_mass(elem) * num for (elem, num) in elems)
    # g/cm^3 / g/mol * 6.022e23/mol * 1e-24 cm^3/angstrom^3
    total_number_density = density / total_mass * 0.60221408

    rng = numpy.random.RandomState(proc_seed(seed, 'make.random'))
    atoms = []

    for (elem, frac) in elems:
        n = int(numpy.round(numpy.prod(cell.box_size) * total_number_density * frac).astype(int))
        pos = rng.uniform(0., 1., size=(3, n))

        atoms.append(Atoms({
            'x': pos[0], 'y': pos[1], 'z': pos[2],
            'elem': [elem] * n,
            **{k: [v] * n for (k, v) in extra_cols.items()}
        }))

    return AtomCell(Atoms.concat(atoms), cell=cell, frame='cell_box')


def slab(atoms: HasAtomCellT, zone: VecLike = (0., 0., 1.), horz: VecLike = (1., 0., 0.), *,
         max_n: int = 50, tol: float = 0.001) -> HasAtomCellT:
    """
    Create an periodic orthogonal slab of the periodic cell `atoms`.

    `zone` in the original crystal will point along the +z-axis,
    and `horz` (minus the `zone` component) wil point along the +x-axis.

    Finds a periodic orthogonal slab with less than `tol` amount of strain,
    and no more than `max_n` cells on one side.

    Args:
      atoms: Input structure
      zone: Zone to align with the +z-axis
      horz: Zone to align with the +x-axis
      max_n: Maximum number of unit cells to search
      tol: Maximum strain tolerance

    Returns:
      A periodic, orthogonal cell
    """

    # align `zone` with the z-axis, and `horz` with the x-axis
    zone = reduce_vec(to_vec3(zone))  # ensure `zone` is a lattice vector
    # TODO should this go from 'local' or 'ortho'?
    cell_transform = atoms.get_transform('local', 'cell_frac').to_linear()
    align_transform = LinearTransform3D.align(cell_transform @ zone, cell_transform @ horz)
    transform = (align_transform @ cell_transform)
    z = transform @ zone
    numpy.testing.assert_allclose(z / numpy.linalg.norm(z), [0., 0., 1.], atol=1e-6)

    # generate lattice points
    lattice_coords = numpy.stack(numpy.meshgrid(*[numpy.arange(-max_n, max_n)]*3), axis=-1).reshape(-1, 3)
    realspace_coords = transform @ lattice_coords
    realspace_norm = numpy.linalg.norm(realspace_coords, axis=-1)

    # sort coordinates from smallest to largest (TODO this method is slow)
    sorting = realspace_norm.argsort()[1:]
    realspace_norm = realspace_norm[sorting]
    lattice_coords = lattice_coords[sorting]
    realspace_coords = realspace_coords[sorting]
    tols = realspace_norm * tol

    # find lattice points which are acceptablly close to orthogonal
    (x_close, y_close, z_close) = split_arr(numpy.abs(realspace_coords) < tols[:, None], axis=-1)

    try:
        x_i = numpy.argwhere(z_close & ~x_close & y_close)[0, 0],
        y_i = numpy.argwhere(z_close & ~y_close & x_close)[0, 0],

        logging.info(f"x: {lattice_coords[x_i]} transforms to {realspace_coords[x_i]}")
        logging.info(f"y: {lattice_coords[y_i]} transforms to {realspace_coords[y_i]}")
        logging.info(f"z: {zone} transforms to {z}")
    except IndexError:
        raise ValueError("Couldn't find a viable surface along zone {zone}") from None

    # orient vectors correctly
    x = realspace_coords[x_i] * numpy.sign(realspace_coords[x_i][0])
    y = realspace_coords[y_i] * numpy.sign(realspace_coords[y_i][1])

    # repeat original lattice to cover orthogonal lattice
    pts = transform.inverse() @ BBox3D.from_pts([numpy.zeros(3), x, y, z]).corners()
    raw_atoms = atoms._repeat_to_contain(pts).get_atoms('local').transform(align_transform)
    cell = Cell.from_ortho(LinearTransform3D(numpy.stack([x, y, z], axis=0)))

    # strain cell to orthogonal (with atoms in the ``cell`` frame)
    raw_atoms = raw_atoms.transform(cell.get_transform('cell', 'local'))
    cell = cell.strain_orthogonal()
    return atoms.with_cell(cell).with_atoms(raw_atoms, 'cell').crop_to_box()


def stacking_sequence(layer: AtomCell, sequence: str, shift_vector: VecLike = (1, 0, 0), *,
                      n_layers: int = 3) -> AtomCell:
    """
    Create an arbitrary stacking sequence from a single layer `layer`.

    Args:
      layer: Layer to stack into a stacking sequence. Will be stacked along the c axis.
      sequence: Stacking sequence. Each layer should be "A", "B", or "C" (in the common case
                where there are three layers). Example: `"ABCABC"`.
      shift_vector: Shift to apply, in fractional coordinates. The shift between each layer
                    will be `shift_vector/n_layers`. Typically an integer value, to
                    preserve periodicity. Defaults to `[100]`.
      n_layers: Number of layers which corresponds to a shift of a complete lattice vector.
                Defaults to `3`, the case for FCC and HCP.

    Returns:
     An [`AtomCell`][atomlib.atomcell.AtomCell] containing the stacked structure.
    """

    layers = string.ascii_uppercase[:n_layers]

    # TODO generalize this to arbitrary number of layers
    sequence = sequence.upper()
    if any(s not in layers for s in sequence):
        raise ValueError(f"Invalid sequence '{sequence}'. Expected values in '{layers}'")

    # c vector to shift along
    c_vec = layer.to_ortho().transform_vec([0, 0, 1])
    # new cell is original cell tiled by the number of layers
    cell = layer.get_cell().repeat([1, 1, len(sequence)])

    # convert shift_vector to local coordinates
    shift_vector = layer.get_cell().get_transform('local', 'cell_frac').transform_vec(shift_vector)

    atoms = layer.get_atoms('local')
    return AtomCell(Atoms.concat(
        # translate by the shift vector and translate to the correct layer
        atoms.transform(AffineTransform3D.translate(shift_vector * (layers.find(c)) / n_layers + i*c_vec))
        for (i, c) in enumerate(sequence)
    ), cell).wrap()


def _ortho_hexagonal(cell: AtomCell) -> AtomCell:
    a, _, c = cell.cell.cell_size
    cell = cell.repeat((2, 2, 1)).explode()
    frame = cell.get_atoms('local')

    eps = 1e-6
    frame = frame.filter(frame.x() >= -eps, frame.x() < a - eps)

    ortho = cell_to_ortho([a, a * numpy.sqrt(3), c])

    return AtomCell.from_ortho(frame, ortho, frame='local')


__all__ = [
    'fcc', 'wurtzite', 'graphite',
    'zincblende', 'diamond', 'fluorite',
    'cesium_chloride', 'perovskite',
    'random', 'slab',
    'CellType',
]
