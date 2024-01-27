"""
Functions to modify/distort atomic structures.
"""

import typing as t

import numpy

from .elem import ElemLike
from .cell import Cell
from .atoms import HasAtomsT, AtomSelection, Atoms
from .atomcell import HasAtomCell, HasAtomCellT
from .transform import AffineTransform3D
from .make import random
from .util import proc_seed


def unbunch(structure: HasAtomsT, threshold: float = 0.4, *,
            max_iter: int = 100, max_step: float = 0.5,
            selection: t.Optional[AtomSelection] = None) -> HasAtomsT:
    """
    Iteratively separate closely-spaced atoms in `structure`.
    """
    from scipy.spatial import KDTree

    if isinstance(structure, HasAtomCell):
        cell = structure.get_cell()
        ortho = cell.get_transform('local', 'cell_box')
        coords = structure.coords(selection, frame='local')
    else:
        cell = structure.bbox_atoms()
        ortho = cell.transform_from_unit()
        coords = structure.coords(selection)

    ortho_inv = ortho.inverse()

    updated = False
    for i in range(max_iter):
        tree = KDTree(coords, compact_nodes=False, balanced_tree=False)

        pairs = tree.query_pairs(threshold, output_type='ndarray')
        pairs.sort(axis=0, kind='quicksort')  # required to ensure deterministicity
        for (i, j) in pairs:
            v_ij = coords[j] - coords[i]
            r = numpy.linalg.norm(v_ij)

            if not r < threshold - 1e-6:
                continue

            # seperate to threshold
            step_size = min(float(threshold - r)/2., max_step)
            # step along unit vector
            delta = step_size * v_ij / r
            coords[j] += delta
            coords[i] -= delta
            updated = True

        if not updated:
            # finished
            break
        updated = False

        # clip to box
        coords = ortho @ numpy.clip(ortho_inv @ coords, 0., 1.)

    if isinstance(structure, HasAtomCell):
        return structure.with_coords(coords, selection, frame='local')
    return structure.with_coords(coords, selection)


def contaminate(structure: HasAtomCellT,
                thickness: t.Union[float, t.Tuple[float, float]],
                density: float = 2.0, elem: ElemLike = 'C', *,
                threshold: float = 0.0, max_iter: int = 100,
                seed: t.Optional[object] = None,
                **extra_cols: t.Any) -> HasAtomCellT:
    """
    Add random surface contamination on the c plane of a supercell.

    ``thickness`` is the thickness on top and bottom, in angstroms.
    ``density`` is the average density of contamination to add
    (defaults to density of amorphous carbon).
    ``elem`` is the element to add (defaults to carbon).

    If ``threshold`` is greater than 0., then ``threshold`` and ``max_iter``
    will be passed to ``unbunch`` to separate close atoms.
    """

    cell = structure.get_cell().explode_z()
    c_size = cell.cell_size[2]
    assert cell.cell_size[2] == cell.box_size[2]

    if isinstance(thickness, (int, float)):
        (top_thick, bot_thick) = (thickness, thickness)
    else:
        (top_thick, bot_thick) = thickness

    assert bot_thick >= 0.
    assert top_thick >= 0.

    if top_thick > 0.:
        top_cell = cell.crop(z_max=top_thick, frame='cell').transform_cell(AffineTransform3D.translate(z=c_size))
        top = [random(top_cell, elem=elem, density=density, seed=proc_seed(seed, 'top_cont'), **extra_cols)]
    else:
        top = []

    if bot_thick > 0.:
        bot_cell = cell.crop(z_max=bot_thick, frame='cell').transform_cell(AffineTransform3D.translate(z=-bot_thick))
        bot = [random(bot_cell, elem=elem, density=density, seed=proc_seed(seed, 'bot_cont'), **extra_cols)]
    else:
        bot = []

    if threshold > 0.:
        top = [unbunch(t, threshold=threshold, max_iter=max_iter) for t in top]
        bot = [unbunch(b, threshold=threshold, max_iter=max_iter) for b in bot]

    atoms = Atoms.concat((structure.get_atoms('local'), *bot, *top))
    cell = Cell(
    affine=AffineTransform3D.translate(z=-bot_thick),
        ortho=cell.ortho, cell_size=[*cell.cell_size[:2], c_size + bot_thick + top_thick],
        pbc=cell.pbc, n_cells=cell.n_cells,
    )
    return structure.with_cell(cell).with_atoms(atoms, frame='local')


__all__ = [
    'unbunch', 'contaminate',
]