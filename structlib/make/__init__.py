"""
Functions to create cells.
"""

import typing as t

import numpy
import polars

from ..core import Atoms, AtomCell, IntoAtoms
from ..transform import LinearTransform3D
from ..elem import get_elem, get_elems
from ..types import ElemLike, Num
from ..cell import cell_to_ortho


CellType = t.Union[t.Literal['conv'], t.Literal['prim'], t.Literal['ortho']]


def fcc(elem: ElemLike, a: Num, *, cell: CellType = 'conv', additional: t.Optional[IntoAtoms] = None) -> AtomCell:
    """
    Make a FCC lattice of the specified element, with the given cell.
    If 'conv' (the default), return the conventional cell, four atoms with cubic cell symmetry.
    If 'prim', return the primitive cell, a single atom with rhombohedral cell symmetry.
    If 'ortho', return an orthogonal cell, two atoms in a cell of size [a/sqrt(2), a/sqrt(2), a].

    If 'additional' is specified, those atoms will be added to the lattice (in fractional coordinates).
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
        frame = frame.concat(Atoms(additional))

    return AtomCell.from_ortho(frame, ortho, frame='cell_frac')


def wurtzite(elems: t.Union[str, t.Sequence[ElemLike]], a: Num, c: t.Optional[Num] = None,
             d: t.Optional[Num] = None, *, cell: CellType = 'conv') -> AtomCell:
    """
    Create a wurzite lattice of the specified two elements, with the given cell.
    `a` and `c` are the hexagonal cell parameters. `d` is the fractional distance
    between the two sublattices.
    """
    if isinstance(elems, str):
        elems = get_elems(elems)
    else:
        elems = list(map(get_elem, elems))

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


def zincblende(elems: t.Union[str, t.Sequence[ElemLike]], a: Num, *,
               cell: CellType = 'conv') -> AtomCell:
    """
    Create a zinc-blende FCC structure AB. Returns the same cell types as `fcc`.
    """
    if isinstance(elems, str):
        elems = get_elems(elems)
    else:
        elems = list(map(get_elem, elems))

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
            'z': [0.25, 0.25],
            'elem': [elems[1]] * 2,
        }
    elif cell == 'conv':
        additional: t.Dict[str, t.Any] = {
            'x': [0.25, 0.25, 0.75, 0.75],
            'y': [0.25, 0.75, 0.25, 0.75],
            'z': [0.25, 0.75, 0.75, 0.25],
            'elem': [elems[1]] * 4,
        }

    return fcc(elems[0], a, cell=cell, additional=additional)



def fluorite(elems: t.Union[str, t.Sequence[ElemLike]], a: Num, *,
             cell: CellType = 'conv') -> AtomCell:
    """
    Create a fluorite FCC structure AB_2. Returns the same cell types as `fcc`.
    """

    if isinstance(elems, str):
        elems = get_elems(elems)
    else:
        elems = list(map(get_elem, elems))

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

    return fcc(elems[0], a, cell=cell, additional=additional)




def _ortho_hexagonal(cell: AtomCell) -> AtomCell:
    a, _, c = cell.cell.cell_size
    cell = cell.repeat((2, 2, 1)).explode()
    frame = cell.get_atoms('local')

    eps = 1e-6
    frame = frame.filter(
        (polars.col('x') >= -eps) & (polars.col('x') < a - eps)
    )

    ortho = cell_to_ortho([a, a * numpy.sqrt(3), c])

    return AtomCell.from_ortho(frame, ortho, frame='local')