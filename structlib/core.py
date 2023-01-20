from __future__ import annotations

import abc
from dataclasses import dataclass, fields, replace
import warnings
from pathlib import Path
import copy
import typing as t

import numpy
import polars

from .bbox import BBox
from .types import VecLike, Vec3, to_vec3
from .transform import LinearTransform, AffineTransform, Transform, IntoTransform
from .cell import cell_to_ortho, ortho_to_cell
from .atoms import Atoms, AtomSelection, IntoAtoms


if t.TYPE_CHECKING:
    from .io import CIF, XYZ, XSF, CFG, FileOrPath, FileType  # pragma: no cover
    from .io.mslice import MSliceTemplate                     # pragma: no cover


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


AtomCollectionT = t.TypeVar('AtomCollectionT', bound='AtomCollection')
AtomCellT = t.TypeVar('AtomCellT', bound='AtomCell')


class AtomCollection(abc.ABC):
    """Abstract class representing any (possibly compound) collection of atoms."""

    @abc.abstractmethod
    def transform(self: AtomCollectionT, transform: AffineTransform, frame: CoordinateFrame = 'local') -> AtomCollectionT:
        """
        Transform atoms by `transform`, in the coordinate frame `frame`.
        Transforms cell boxes in addition to atoms.
        """
        ...

    @abc.abstractmethod
    def transform_atoms(self: AtomCollectionT, transform: IntoTransform, frame: CoordinateFrame = 'local') -> AtomCollectionT:
        """
        Transform atoms by `transform`, in the coordinate frame `frame`.
        Never transforms cell boxes.
        """
        ...

    @abc.abstractmethod
    def get_atoms(self, frame: CoordinateFrame = 'local') -> Atoms:
        ...

    @abc.abstractmethod
    def bbox(self) -> BBox:
        """Return the bounding box of self, in global coordinates."""
        ...

    @abc.abstractmethod
    def clone(self: AtomCollectionT) -> AtomCollectionT:
        ...

    @abc.abstractmethod
    def _str_parts(self) -> t.Iterable[t.Any]:
        ...

    @abc.abstractmethod
    def _replace_atoms(self: AtomCollectionT, atoms: Atoms, frame: CoordinateFrame = 'local') -> AtomCollectionT:
        ...

    def __repr__(self) -> str:
        return "\n".join(map(str, self._str_parts()))

    __str__ = __repr__

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @t.overload
    @staticmethod
    def read(path: FileOrPath, ty: FileType) -> AtomCollection:
        ...

    @t.overload
    @staticmethod
    def read(path: t.Union[str, Path, t.TextIO], ty: t.Literal[None] = None) -> AtomCollection:
        ...

    @staticmethod
    def read(path, ty=None) -> AtomCollection:
        """
        Read a structure from a file.

        Currently, supported file types are 'cif', 'xyz', and 'xsf'.
        If no `ty` is specified, it is inferred from the file's extension.
        """
        return io.read(path, ty)

    @staticmethod
    def read_cif(f: t.Union[FileOrPath, CIF]) -> AtomCollection:
        """Read a structure from a CIF file."""
        return io.read_cif(f)

    @staticmethod
    def read_xyz(f: t.Union[FileOrPath, XYZ]) -> AtomCollection:
        """Read a structure from an XYZ file."""
        return io.read_xyz(f)

    @staticmethod
    def read_xsf(f: t.Union[FileOrPath, XSF]) -> AtomCollection:
        """Read a structure from an XSF file."""
        return io.read_xsf(f)

    @staticmethod
    def read_cfg(f: t.Union[FileOrPath, CFG]) -> AtomCell:
        """Read a structure from a CFG file."""
        return io.read_cfg(f)

    @t.overload
    def write(self, path: FileOrPath, ty: FileType):
        ...

    @t.overload
    def write(self, path: t.Union[str, Path, t.TextIO], ty: t.Literal[None] = None):
        ...

    def write(self, path, ty=None):
        """
        Write this structure to a file.

        A file type may be specified using `ty`.
        If no `ty` is specified, it is inferred from the path's extension.
        """
        io.write(self, path, ty)


@dataclass(init=False, repr=False, frozen=True)
class SimpleAtoms(AtomCollection):
    """
    Cell of atoms with no known structure or periodicity.
    """

    atoms: Atoms
    """Atoms in the cell. Stored in 'local' coordinates (i.e. relative to the enclosing group but not relative to box dimensions)."""

    def __init__(self, atoms: IntoAtoms):
        object.__setattr__(self, 'atoms', Atoms(atoms))

    def bbox(self) -> BBox:
        """Get this structure's bounding box."""
        return self.atoms.bbox

    def with_bounds(self, cell_size: t.Optional[VecLike] = None) -> AtomCell:
        """
        Return a periodic cell with the given orthogonal cell dimensions.

        If cell_size is not specified, it will be assumed (and may be incorrect).
        """
        if cell_size is None:
            warnings.warn("Cell boundary unknown. Defaulting to cell BBox")
            cell_size = self.atoms.bbox.size

        # TODO be careful with the origin here
        return AtomCell(self.atoms, cell_size=cell_size)

    def transform(self, transform: IntoTransform, frame: CoordinateFrame = 'local') -> SimpleAtoms:
        if frame.lower() == 'frac':
            raise ValueError("Can't use 'frac' coordinate frame when box is unknown.")

        return replace(self, atoms=self.atoms.transform(transform))

    transform_atoms = transform

    def get_atoms(self, frame: CoordinateFrame = 'local') -> Atoms:
        if frame.lower() == 'frac':
            raise ValueError("Can't use 'frac' coordinate frame when box is unknown.")

        return self.atoms

    def _replace_atoms(self: AtomCollectionT, atoms: Atoms, frame: CoordinateFrame = 'local') -> AtomCollectionT:
        if frame.lower() == 'frac':
            raise ValueError("Can't use 'frac' coordinate frame when box is unknown.")
        
        return replace(self, frame=atoms)

    def clone(self) -> SimpleAtoms:
        return self.__class__(**{field.name: copy.deepcopy(getattr(self, field.name)) for field in fields(self)})

    def _str_parts(self) -> t.Iterable[t.Any]:
        return (self.atoms,)

    def __len__(self) -> int:
        return self.atoms.__len__()


@dataclass(init=False, repr=False, frozen=True)
class AtomCell(AtomCollection):
    """
    Cell of atoms with known size and periodic boundary conditions.
    """

    atoms: Atoms
    """Atoms in the cell. Stored in 'local' coordinates (i.e. relative to the enclosing group but not relative to box dimensions)."""

    cell_size: Vec3
    """Cell size (a, b, c), neglecting n_cells."""

    cell_angle: Vec3
    """Cell angles (alpha, beta, gamma)"""

    n_cells: Vec3
    """Number of cells (n_a, n_b, n_c)"""

    ortho: LinearTransform
    """Orthogonalization transform. Converts fractional coordinates to local real-space coordinates."""
    #ortho_inv: LinearTransform
    #"""Inverse orthogonalization transform. Converts local real-space coordinates to fractional coordinates."""
    metric: LinearTransform
    """Metric tensor. p dot q = p.T @ M @ q forall p, q"""

    @t.overload
    def __init__(self, atoms: IntoAtoms, cell_size: VecLike,
                 cell_angle: t.Optional[VecLike] = None, *,
                 n_cells: t.Optional[VecLike] = None,
                 ortho: t.Literal[None] = None,
                 frac: bool = False):
        ...

    @t.overload
    def __init__(self, atoms: IntoAtoms, *,
                 n_cells: t.Optional[VecLike] = None,
                 ortho: LinearTransform,
                 frac: bool = False):
        ...

    def __init__(self, atoms: IntoAtoms,
                 cell_size: t.Optional[VecLike] = None,
                 cell_angle: t.Optional[VecLike] = None, *,
                 n_cells: t.Optional[VecLike] = None,
                 ortho: t.Optional[LinearTransform] = None,
                 frac: bool = False):
        if n_cells is None:
            n_cells = to_vec3(numpy.ones((3,), dtype=int), numpy.int_)
        else:
            n_cells = to_vec3(n_cells, numpy.int_)
            if not numpy.issubdtype(n_cells.dtype, numpy.integer):
                raise TypeError(f"n_cells must be an integer dtype. Instead got dtype '{n_cells.dtype}'")
        object.__setattr__(self, 'n_cells', n_cells)

        if ortho is not None:
            if cell_size is not None or cell_angle is not None:
                raise ValueError("Crystal: 'ortho' cannot be specified with 'cell_size' or 'cell_angle'.")

            (cell_size, cell_angle) = ortho_to_cell(ortho)
        else:
            if cell_size is None:
                raise ValueError("Crystal: Either 'cell_size' or 'ortho' must be specified.")

            cell_size = to_vec3(cell_size)
            cell_angle = to_vec3(cell_angle if cell_angle is not None else numpy.full(3, numpy.pi/2.))
            ortho = cell_to_ortho(cell_size, cell_angle)

        object.__setattr__(self, 'ortho', ortho)
        object.__setattr__(self, 'cell_size', cell_size)
        object.__setattr__(self, 'cell_angle', cell_angle)
        object.__setattr__(self, 'metric', self.ortho.T @ self.ortho)

        atoms = Atoms(atoms)
        if frac:
            atoms = atoms.transform(ortho).round_near_zero()

        object.__setattr__(self, 'atoms', atoms)

        self.__post_init__()

    def __post_init__(self):
        pass

    def _str_parts(self) -> t.Iterable[t.Any]:
        return (
            f"Cell size:  {self.cell_size!s}",
            f"Cell angle: {self.cell_angle!s}",
            f"# Cells: {self.n_cells!s}",
            self.atoms,
        )

    def __len__(self) -> int:
        return self.atoms.__len__()

    def transform(self, transform: AffineTransform, frame: CoordinateFrame = 'local') -> AtomCell:
        if isinstance(transform, Transform) and not isinstance(transform, AffineTransform):
            raise ValueError("Non-affine transforms cannot change the box dimensions. Use `transform_atoms` instead.")

        if frame.lower() == 'frac':
            # coordinate change the transform
            transform = self.ortho @ transform @ self.ortho.inverse()

        ortho = transform.to_linear() @ self.ortho
        return AtomCell(self.atoms.transform(transform), ortho=ortho)

    def permute_axes(self, shift: t.Optional[int] = None) -> AtomCell:
        """Permutes the axes of the crystal, such that 'a' is closest to the 'x' axis."""

        if shift is None:
            dots = numpy.array([numpy.dot([1, 0, 0], vec / numpy.linalg.norm(vec)) for vec in self.ortho.inner.T])
            print(f"dots: {dots}")
            new_i = numpy.argmax(dots)
        else:
            new_i = shift
        if new_i == 0:
            return self

        print(f"new_i: {shift}")
        atoms = self.get_atoms('frac')
        atoms = atoms.with_coords(numpy.roll(atoms.coords(), -new_i, axis=1))
        ortho = LinearTransform(numpy.roll(self.ortho.inner, -new_i, axis=1))
        print(f"old ortho:\n{self.ortho.inner}")
        print(f"new ortho:\n{ortho}")

        return AtomCell(atoms, ortho=ortho, frac=True)

    def transform_axes(self, transform: AffineTransform) -> AtomCell:
        """Transform the local axes of the crystal, without affecting the atom positions."""
        if isinstance(transform, Transform) and not isinstance(transform, AffineTransform):
            raise ValueError("Non-affine transforms cannot change the box dimensions. Use `transform_atoms` instead.")

        ortho = transform.to_linear() @ self.ortho
        return AtomCell(self.atoms, ortho=ortho)

    def transform_atoms(self, transform: IntoTransform, frame: CoordinateFrame = 'local') -> AtomCell:
        if frame.lower() == 'frac':
            # coordinate change the transform
            transform = self.ortho @ Transform.make(transform) @ self.ortho.inverse()

        return AtomCell(self.atoms.transform(transform), ortho=self.ortho)

    def crop(self, x_min: float = -numpy.inf, x_max: float = numpy.inf,
             y_min: float = -numpy.inf, y_max: float = numpy.inf,
             z_min: float = -numpy.inf, z_max: float = numpy.inf, *,
             frame: CoordinateFrame = 'local') -> AtomCell:
        self = self.explode()

        mins = to_vec3([x_min, y_min, z_min])
        maxs = to_vec3([x_max, y_max, z_max])

        if frame.lower() != 'frac':
            if not self.is_orthogonal():
                raise ValueError("Cannot crop a non-orthogonal cell in orthogonal coordinates. Use crop_atoms instead.")
            [mins, maxs] = self.ortho.inverse().transform([mins, maxs])

        frac_atoms = self.get_atoms('frac')
        new_atoms = frac_atoms.filter(
            (polars.col('x') >= mins[0]) & (polars.col('x') <= maxs[0]) &
            (polars.col('y') >= mins[1]) & (polars.col('y') <= maxs[1]) &
            (polars.col('z') >= mins[2]) & (polars.col('z') <= maxs[2])
        )
        # TODO this is broken for shifted cell minimums
        new_ortho = LinearTransform(numpy.diag([min(v_max, 1.) for v_max in maxs])).compose(self.ortho)
        return AtomCell(new_atoms, ortho=new_ortho, frac=True)

    def crop_atoms(self, x_min: float = -numpy.inf, x_max: float = numpy.inf,
                   y_min: float = -numpy.inf, y_max: float = numpy.inf,
                   z_min: float = -numpy.inf, z_max: float = numpy.inf, *,
                   frame: CoordinateFrame = 'local') -> AtomCell:
        self = self.explode()
        min = to_vec3([x_min, y_min, z_min])
        max = to_vec3([x_max, y_max, z_max])

        if frame.lower() == 'frac':
            [min, max] = self.ortho.transform([min, max])

        new_atoms = self.get_atoms('local').filter(
            (polars.col('x') >= min[0]) & (polars.col('x') <= max[0]) &
            (polars.col('y') >= min[1]) & (polars.col('y') <= max[1]) &
            (polars.col('z') >= min[2]) & (polars.col('z') <= max[2])
        )
        return self._replace_atoms(new_atoms, frame='local')

    def wrap(self, eps: float = 1e-5):
        atoms = self.get_atoms('frac')
        coords = atoms.coords()
        coords = (coords + eps) % 1. - eps
        return self._replace_atoms(atoms.with_coords(coords), frame='frac')

    def get_atoms(self, frame: CoordinateFrame = 'local') -> Atoms:
        if frame.lower() == 'frac':
            return self.atoms.transform(self.ortho.inverse())

        return self.atoms

    def _replace_atoms(self, atoms: Atoms, frame: CoordinateFrame = 'local') -> AtomCell:
        if frame.lower() == 'frac':
            atoms = atoms.transform(self.ortho)
        return AtomCell(atoms, n_cells=self.n_cells, ortho=self.ortho)

    def bbox(self) -> BBox:
        return self.atoms.bbox | self.cell_bbox('global')

    def cell_corners(self, frame: CoordinateFrame = 'global') -> numpy.ndarray:
        """Return a (8, 3) ndarray containing the corners of self."""
        n = [(0., n) for n in self.n_cells]
        corners = numpy.stack(list(map(numpy.ravel, numpy.meshgrid(*n, indexing='ij'))), axis=-1)
        if frame.lower() in ('local', 'global'):
            return self.ortho @ corners
        return corners

    def cell_bbox(self, frame: CoordinateFrame = 'global') -> BBox:
        return BBox.from_pts(self.cell_corners(frame))

    def is_orthogonal(self) -> bool:
        return self.ortho.is_diagonal()

    def orthogonalize(self) -> OrthoCell:
        if self.is_orthogonal:
            return OrthoCell(self.atoms, ortho=self.ortho, n_cells=self.n_cells)
        raise NotImplementedError()

    def repeat(self, n: t.Union[int, t.Tuple[int, int, int]], /, *, explode: bool = False) -> AtomCell:
        """Tile the cell"""

        n_cells = (self.n_cells * numpy.broadcast_to(n, 3))
        new = self.__class__(self.atoms, ortho=self.ortho, n_cells=n_cells)
        return new.explode() if explode else new

    def explode(self) -> AtomCell:
        """Turn a tiled cell into one large cell, duplicating atoms along the way."""
        if numpy.prod(self.n_cells.view(numpy.ndarray)) == 1:
            return self

        a, b, c = map(numpy.arange, self.n_cells)

        # Nx3 ndarray of cell offsets
        cells = numpy.stack(numpy.meshgrid(a, b, c)).reshape(3, -1).T.astype(float)

        ortho_inv = self.ortho.inverse()
        atoms = Atoms.concat([
            self.atoms.transform(self.ortho @ AffineTransform.translate(cell) @ ortho_inv)
            for cell in cells
        ])

        ortho = self.ortho @ LinearTransform.scale(self.n_cells)
        new = self.__class__(atoms, ortho=ortho)
        return new

    def clone(self: AtomCellT) -> AtomCellT:
        return self.__class__(**{field.name: copy.deepcopy(getattr(self, field.name)) for field in fields(self)})

    def write_mslice(self, f: FileOrPath, template: t.Optional[MSliceTemplate] = None):
        """Read this structure to an mslice file."""
        return io.write_mslice(self, f, template)

    __mul__ = repeat

    def assert_equal(self, other):
        assert isinstance(other, AtomCell)
        numpy.testing.assert_array_almost_equal(self.ortho.inner, other.ortho.inner, 6)
        numpy.testing.assert_array_equal(self.n_cells, other.n_cells)
        self.atoms.assert_equal(other.atoms)


class OrthoCell(AtomCell):
    def __post_init__(self):
        if not numpy.allclose(self.cell_angle, numpy.pi/2.):
            raise ValueError(f"OrthoCell constructed with non-orthogonal angles: {self.cell_angle}")

    def is_orthogonal(self) -> t.Literal[True]:
        return True


from . import io


__ALL__ = [
    'CoordinateFrame', 'Atoms', 'IntoAtoms', 'AtomSelection', 'AtomCollection', 'Cell', 'PeriodicCell', 'Lattice',
]