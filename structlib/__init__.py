from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import logging
import abc
import typing as t

import pandas
import numpy

from .elements import ELEMENTS, ELEMENT_SYMBOLS
from .cif import CIF
from .xyz import XYZ
from .xsf import XSF
from .transform import AffineTransform, LinearTransform, Transform
from .util import map_some, split_arr, FileOrPath
from .vec import Vec3
from .cell import cell_to_ortho, ortho_to_cell

StructureT = t.TypeVar('StructureT', bound='Structure')

CoordinateFrame = t.Union[t.Literal['local'], t.Literal['global'], t.Literal['frac']]
"""
A coordinate frame to use.
There are three main coordinate frames:
 - 'crystal', uses crystallographic axes
 - 'local', orthogonal coordinate system defined by the crystal's bounding box
 - 'global', global coordinate frame

In addition, the 'crystal' and 'local' coordinate frames support fractional
coordinates as well as realspace (in angstrom) coordinates.
"""

Selection = pandas.DataFrame


@dataclass
class Structure:
    atoms: pandas.DataFrame
    """Atoms in the unit cell. Stored in local real-space coordinates."""

    symmetry_sites: t.List[AffineTransform] = field(default_factory=list)
    """List of symmetry sites in the unit cell, stored in crystal coordinates. Separate from translational symmetry."""

    cell_size: t.Optional[Vec3] = None
    """Cell parameters (a, b, c)"""
    cell_angle: Vec3 = field(default_factory=lambda: numpy.pi/2. * numpy.ones((3,)).view(Vec3))
    """Cell angles (alpha, beta, gamma)"""

    n_cells: Vec3 = field(default_factory=lambda: numpy.ones((3,), dtype=int).view(Vec3))
    """Number of cells (n_a, n_b, n_c)"""

    global_transform: AffineTransform = field(default_factory=AffineTransform)
    """Converts local real-space coordinates to global real-space coordinates."""

    ortho: LinearTransform = field(default=None)  # type: ignore (fixed in post_init)
    """Orthogonalization transform. Converts fractional coordinates to local real-space coordinates."""
    ortho_inv: LinearTransform = field(init=False)
    """Fractionalization transform. Converts local real-space coordinates to fractional ones."""
    metric: LinearTransform = field(init=False)
    """Metric tensor. p dot q = p.T @ M @ q forall p, q"""
    metric_inv: LinearTransform = field(init=False)
    """Inverse metric tensor. g dot h = g.T @ M^-1 @ h forall g, h"""

    ortho_global: AffineTransform = field(init=False)
    """Global orthogonalization transform. Converts fractional coordinates to global real-space coordinates."""

    def __post_init__(self):
        if self.ortho is not None:
            if self.cell_size is not None:
                raise ValueError("ortho and cell_size can't both be specified.")
            (self.cell_angle, self.cell_size) = ortho_to_cell(self.ortho)
        else:
            self.ortho = cell_to_ortho(self.cell_angle, self.cell_size)

        self.ortho_inv = self.ortho.inverse()
        self.metric = self.ortho.T @ self.ortho
        self.metric_inv = self.metric.inverse()
        self.ortho_global = self.global_transform @ self.ortho

    def is_orthogonal(self) -> bool:
        return numpy.allclose(self.cell_angle, numpy.pi/2.)

    @classmethod
    def from_cif(cls: t.Type[StructureT], path: t.Union[CIF, FileOrPath]) -> StructureT:
        if isinstance(path, CIF):
            cif = path
        else:
            cif = next(CIF.from_file(path))
 
        df = pandas.DataFrame(cif.stack_tags('atom_site_fract_x', 'atom_site_fract_y', 'atom_site_fract_z',
                                             'atom_site_type_symbol', 'atom_site_occupancy'))
        df.columns = ['a','b','c','symbol','frac_occupancy']

        df['symbol'] = df['symbol'].map(lambda s: ''.join([c for c in s if c.isalpha()]).title())
        df['atomic_number'] = df['symbol'].map(ELEMENTS)

        return cls(df, cell_size=map_some(Vec3.make, cif.cell_size()), symmetry_sites=list(cif.get_symmetry()))

    def write_cif(self, path: t.Union[str, Path, t.TextIO]):
        raise NotImplementedError()

    @classmethod
    def from_xyz(cls: t.Type[StructureT], path: t.Union[XYZ, FileOrPath]) -> StructureT:
        if isinstance(path, XYZ):
            xyz = path
        else:
            xyz = XYZ.from_file(path)

        atoms = xyz.atoms.copy()
        atoms['symbol'] = atoms['symbol'].map(lambda s: ''.join([c for c in s if c.isalpha()]).title())
        atoms['atomic_number'] = atoms['symbol'].map(ELEMENTS)
        atoms['frac_occupancy'] = 1.0
        atoms = atoms[['a', 'b', 'c', 'symbol', 'frac_occupancy', 'atomic_number']]

        return cls(atoms)

    def write_xyz(self, path: FileOrPath, frame: CoordinateFrame = 'global',
                  ext: bool = True, comment: t.Optional[str] = None) -> None:
        params = {}
        if ext:
            matrix = self.ortho_global.inner[:3, :3] if frame == 'global' else self.ortho.inner
            lattice = " ".join(map(str, matrix.flat))
            params['Lattice'] = lattice
            if comment is not None:
                params['Comment'] = comment
        xyz = XYZ(self.atoms, params=params, comment=comment)
        xyz.write(path)

    @classmethod
    def from_xsf(cls: t.Type[StructureT], path: t.Union[XSF, FileOrPath]) -> StructureT:
        if isinstance(path, XSF):
            xsf = path
        else:
            xsf = XSF.from_file(path)

        atoms = xsf.get_atoms().copy()
        atoms['symbol'] = atoms['atomic_number'].map(lambda i: ELEMENT_SYMBOLS[i])
        atoms['frac_occupancy'] = 1.0

        return cls(atoms, ortho=xsf.primitive_cell)  # type: ignore

    def write_mslice(self, path: t.Union[str, Path, t.TextIO]) -> None:
        raise NotImplementedError()

    @classmethod
    def from_file(cls: t.Type[StructureT], path: t.Union[str, Path]) -> StructureT:
        path = Path(path)
        ext = path.suffix.lower()
        if ext == '.cif':
            return cls.from_cif(path)
        if ext == '.xyz':
            return cls.from_xyz(path)
        raise ValueError(f"Unknown file type '{path.suffix}'")

    def write(self, path: t.Union[str, Path]) -> None:
        path = Path(path)
        ext = path.suffix.lower()
        if ext == '.cif':
            return self.write_cif(path)
        if ext == '.xyz':
            return self.write_xyz(path)
        raise ValueError(f"Unknown file type '{path.suffix}'")

    def transform(self: StructureT, transform: Transform) -> StructureT:
        return replace(self, global_transform=self.global_transform @ transform)

    def transform_atoms(self: StructureT, transform: Transform, frame: CoordinateFrame = 'frac') -> StructureT:
        if frame == 'global':
            transform = transform @ self.ortho_global.inverse()  # TODO test these
        elif frame == 'local':
            transform = transform @ self.ortho_inv
        elif not frame == 'frac':
            raise ValueError(f"Unknown coordinate frame '{frame}'. Expected 'global', 'local', or 'frac'.")

        pos = self.atom_positions('frac')
        atoms = self.atoms.copy()
        (atoms['a'], atoms['b'], atoms['c']) = split_arr(pos, axis=-1)
        return replace(self, atoms=atoms)

    def discard_symmetry(self: StructureT, translational=True) -> StructureT:
        if len(self.symmetry_sites) == 0:
            return self
        structs = []
        for site in self.symmetry_sites:
            structs.append(_transform_structure(self.atoms, site))
        new = replace(self, atoms=pandas.concat(structs, ignore_index=True), symmetry_sites=[])
        if translational:
            return new.discard_translational_symmetry()
        return new

    def discard_translational_symmetry(self: StructureT) -> StructureT:
        struct = self.atoms.copy()
        new_struct = struct.copy()
        for (label, n) in zip(('a', 'b', 'c'), self.n_cells):
            if n == 0:
                new_struct.drop(columns='_n', errors='ignore')
                new_struct = new_struct[0:0]
                break
            new_struct['_n'] = (tuple(range(n)),) * new_struct.shape[0]
            new_struct = new_struct.explode('_n', ignore_index=True)
            new_struct[label] += new_struct['_n'].astype(new_struct[label].dtype)

        cell_size = self.cell_size
        if cell_size is not None:
            cell_size *= self.n_cells

        new_struct.drop(columns='_n', inplace=True)
        return replace(self, atoms=new_struct, cell_size=cell_size)

    def repeat(self: StructureT, n_a: int = 1, n_b: int = 1, n_c: int = 1, discard=True) -> StructureT:
        new = replace(self, n_cells=Vec3.make((n_a, n_b, n_c)) * self.n_cells)
        if discard:
            return new.discard_translational_symmetry()
        return new

    def add_atom(self: StructureT, atom: t.Union[str, int], x: float, y: float, z: float,
                 frac_occupancy: float = 1., wobble: float = 0.) -> StructureT:
        if isinstance(atom, str):
            atomic_number = ELEMENTS[atom.title()]
        else:
            atomic_number = atom
        symbol = ELEMENT_SYMBOLS[atomic_number - 1]

        return replace(self, atoms=self.atoms.append({
            'symbol': symbol,
            'atomic_number': atomic_number,
            'x': x,
            'y': y,
            'z': z,
            'wobble': wobble,
            'frac_occupancy': frac_occupancy,
        }, ignore_index=True))

    def atom_positions(self, frame: CoordinateFrame = 'frac') -> numpy.ndarray:
        atoms = self.atoms
        fractional = numpy.stack((atoms['a'], atoms['b'], atoms['c']), axis=-1)
        if frame == 'frac':
            return fractional
        elif frame == 'local':
            return self.ortho @ fractional
        elif frame == 'global':
            return self.ortho_global @ fractional
        else:
            raise ValueError(f"Unknown coordinate frame '{frame}'. Expected 'global', 'local', or 'frac'.")

    def trim(self: StructureT,
             x_min: t.Optional[float] = None, x_max: t.Optional[float] = None,
             y_min: t.Optional[float] = None, y_max: t.Optional[float] = None,
             z_min: t.Optional[float] = None, z_max: t.Optional[float] = None,
             frame: CoordinateFrame = 'local') -> StructureT:
        self = self.discard_symmetry()
        mask = pandas.Series(True, index=self.atoms.index)

        pos = self.atom_positions(frame).T
        if x_min is not None:
            mask &= pos[0] >= x_min
        if x_max is not None:
            mask &= pos[0] <= x_max
        if y_min is not None:
            mask &= pos[1] >= y_min
        if y_max is not None:
            mask &= pos[1] <= y_max
        if z_min is not None:
            mask &= pos[2] >= z_min
        if z_max is not None:
            mask &= pos[2] <= z_max

        logging.info(f"Trimmed {(~mask).sum()} of {len(self.atoms)} atoms")
        return replace(self, atoms=self.atoms.loc[mask])


def _structure_to_positions(df: pandas.DataFrame) -> numpy.ndarray:  # shape: n, 3
	return numpy.stack((df['a'], df['b'], df['c']), axis=-1)


def _transform_structure(df: pandas.DataFrame, transform: Transform) -> pandas.DataFrame:
	pos = transform.transform(_structure_to_positions(df))
	new_df = df.copy()
	(new_df['x'], new_df['y'], new_df['z']) = split_arr(pos, axis=-1)
	return new_df