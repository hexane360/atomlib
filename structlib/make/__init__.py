"""
Functions to create cells.
"""

import typing as t

import numpy


from ..core import AtomCollection, AtomFrame, AtomCell, SimpleAtoms, IntoAtoms
from ..transform import LinearTransform
from ..vec import Vec3
from ..elem import get_elem, get_elems
from ..types import VecLike, ElemLike, Num
from ..cell import cell_to_ortho


CellType = t.Union[t.Literal['conv'], t.Literal['prim'], t.Literal['ortho']]


def fcc(elem: ElemLike, a: Num, *, cell: CellType = 'conv', additional: t.Optional[IntoAtoms] = None) -> AtomCell:
    """
    Make a FCC lattice of the specified element, with the given cell.
    If 'conv' (the default), return the conventional cell, four atoms with cubic cell symmetry.
    If 'prim', return the primitive cell, a single atom with rhombohedral cell symmetry.
    If 'ortho', return an orthorhombic cell, two atoms in a cell of size [a/sqrt(2), a/sqrt(2), a].

    If 'additional' is specified, those atoms will be added to the lattice (in fractional coordinates).
    """

    elems = [get_elem(elem)]
    cell = t.cast(CellType, str(cell).lower())

    if cell == 'prim':
        xs = ys = zs = [0.]
        ortho = LinearTransform(a / 2. * numpy.array([
            [0., 1., 1.],
            [1., 0., 1.],
            [1., 1., 0.],
        ]))
    elif cell == 'ortho':
        elems *= 2
        xs = ys = zs = [0., 0.5]
        b = a / numpy.sqrt(2)
        ortho = LinearTransform.scale(b, b, a)
    elif cell == 'conv':
        elems *= 4
        xs = [0., 0., 0.5, 0.5]
        ys = [0., 0.5, 0., 0.5]
        zs = [0., 0.5, 0.5, 0.]
        ortho = LinearTransform.scale(all=a)
    else:
        raise ValueError(f"Unknown cell type '{cell}'. Expected 'conv', 'prim', or 'ortho'.")

    frame = AtomFrame(dict(x=xs, y=ys, z=zs, elem=elems))
    if additional is not None:
        frame = frame + AtomFrame(additional)

    return AtomCell(frame, ortho=ortho, frac=True)


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
    c_a: float = numpy.sqrt(8. / 3.) if c is None else c / a

    d = 0.25 + 1 / (3 * c_a**2) if d is None else d
    if not 0 < d < 0.5:  # type: ignore
        raise ValueError(f"Invalid 'd' parameter: {d}")

    if cell in ('prim', 'conv'):
        ortho = cell_to_ortho(
            a * numpy.array([1., 1., c_a]), 
            numpy.pi * numpy.array([1/2., 1/2., 2/3.])
        )
        xs = [2/3, 2/3, 1/3, 1/3]
        ys = [1/3, 1/3, 2/3, 2/3]
        zs = [0., 1. - d, 0.5, 0.5 - d]
        elems *= 2
    elif cell == 'ortho':
        raise NotImplementedError()
    else:
        raise ValueError(f"Unknown cell type '{cell}'. Expected 'conv', 'prim', or 'ortho'.")

    frame = AtomFrame(dict(x=xs, y=ys, z=zs, elem=elems))
    return AtomCell(frame, ortho=ortho, frac=True)