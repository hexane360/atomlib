from __future__ import annotations

import abc
from pathlib import Path
import typing as t

from numpy.typing import ArrayLike

if t.TYPE_CHECKING:  # pragma: no cover
    from .atoms import HasAtoms as _HasAtoms
    from .atoms import HasAtomsT
    from .atomcell import HasAtomCell as _HasAtomCell

    from .io import CIF, CIFDataBlock, XYZ, XYZFormat, XSF, CFG, LMP, FileType, FileOrPath
    from .io.mslice import MSliceFile, BinaryFileOrPath

else:
    class _HasAtoms: ...
    class _HasAtomCell: ...


def _cast_atoms(atoms: _HasAtoms, ty: t.Type[HasAtomsT]) -> HasAtomsT:
    """
    Ensure `atoms` is constructed as the correct type.

    If `ty` is `HasAtoms` or `HasAtomCell`, any subclass will be returned.
    But if `ty` is concrete, `atoms` will be converted to exactly that type
    (even if it throws away information).
    """
    from .atoms import HasAtoms, Atoms
    from .atomcell import HasAtomCell, AtomCell

    if isinstance(atoms, ty):
        return atoms
    if issubclass(ty, HasAtomCell) and not isinstance(atoms, HasAtomCell):
        raise TypeError("File contains no cell information.")

    if ty is AtomCell and isinstance(atoms, HasAtomCell):
        return atoms.get_atomcell()  # type: ignore
    if ty is Atoms and isinstance(atoms, HasAtoms):
        return atoms.get_atoms()  # type: ignore
    raise TypeError(f"Can't convert read atoms type '{type(atoms)}' to requested type '{ty}'")


class AtomsIOMixin(_HasAtoms, abc.ABC):
    """
    Mix-in to add IO methods to [`HasAtoms`][atomlib.atoms.HasAtoms].

    All concrete subclasses of [`HasAtoms`][atomlib.atoms.HasAtoms] should also subclass this.
    """

    @t.overload
    @classmethod
    def read(cls: t.Type[HasAtomsT], path: FileOrPath, ty: FileType) -> HasAtomsT:
        ...

    @t.overload
    @classmethod
    def read(cls: t.Type[HasAtomsT], path: t.Union[str, Path, t.TextIO], ty: t.Literal[None] = None) -> HasAtomsT:
        ...

    @classmethod
    def read(cls: t.Type[HasAtomsT], path: FileOrPath, ty: t.Optional[FileType] = None) -> HasAtomsT:
        """
        Read a structure from a file.

        Supported types can be found in the [io][atomlib.io] module.
        If no `ty` is specified, it is inferred from the file's extension.
        """
        from .io import read
        return _cast_atoms(read(path, ty), cls)  # type: ignore

    @classmethod
    def read_cif(cls: t.Type[HasAtomsT], f: t.Union[FileOrPath, CIF, CIFDataBlock], block: t.Union[int, str, None] = None) -> HasAtomsT:
        """
        Read a structure from a CIF file.

        If `block` is specified, read data from the given block of the CIF file (index or name).
        """
        from .io import read_cif
        return _cast_atoms(read_cif(f, block), cls)

    @classmethod
    def read_xyz(cls: t.Type[HasAtomsT], f: t.Union[FileOrPath, XYZ]) -> HasAtomsT:
        """Read a structure from an XYZ file."""
        from .io import read_xyz
        return _cast_atoms(read_xyz(f), cls)

    @classmethod
    def read_xsf(cls: t.Type[HasAtomsT], f: t.Union[FileOrPath, XSF]) -> HasAtomsT:
        """Read a structure from an XSF file."""
        from .io import read_xsf
        return _cast_atoms(read_xsf(f), cls)

    @classmethod
    def read_cfg(cls: t.Type[HasAtomsT], f: t.Union[FileOrPath, CFG]) -> HasAtomsT:
        """Read a structure from a CFG file."""
        from .io import read_cfg
        return _cast_atoms(read_cfg(f), cls)

    @classmethod
    def read_lmp(cls: t.Type[HasAtomsT], f: t.Union[FileOrPath, LMP], type_map: t.Optional[t.Dict[int, t.Union[str, int]]] = None) -> HasAtomsT:
        """Read a structure from a LAAMPS data file."""
        from .io import read_lmp
        return _cast_atoms(read_lmp(f, type_map=type_map), cls)

    def write_cif(self, f: FileOrPath):
        from .io import write_cif
        write_cif(self, f)

    def write_xyz(self, f: FileOrPath, fmt: XYZFormat = 'exyz'):
        from .io import write_xyz
        write_xyz(self, f, fmt)

    def write_xsf(self, f: FileOrPath):
        from .io import write_xsf
        write_xsf(self, f)

    def write_cfg(self, f: FileOrPath):
        from .io import write_cfg
        write_cfg(self, f)

    def write_lmp(self, f: FileOrPath):
        from .io import write_lmp
        write_lmp(self, f)

    @t.overload
    def write(self, path: FileOrPath, ty: FileType):
        ...

    @t.overload
    def write(self, path: t.Union[str, Path, t.TextIO], ty: t.Literal[None] = None):
        ...

    def write(self, path: FileOrPath, ty: t.Optional[FileType] = None):
        """
        Write this structure to a file.

        A file type may be specified using `ty`.
        If no `ty` is specified, it is inferred from the path's extension.
        """
        from .io import write
        write(self, path, ty)  # type: ignore


class AtomCellIOMixin(_HasAtomCell, AtomsIOMixin):
    """
    Mix-in to add IO methods to [`HasAtomCell`][atomlib.atomcell.HasAtomCell].

    All concrete subclasses of [`HasAtomCell`][atomlib.atomcell.HasAtomCell] should also subclass this.
    """

    def write_mslice(self, f: BinaryFileOrPath, template: t.Optional[MSliceFile] = None, *,
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
        Write a structure to an mslice file.

        `template` may be a file, path, or `ElementTree` containing an existing mslice file.
        Its structure will be modified to make the final output. If not specified, a default
        template will be used.

        Additional options modify simulation properties. If an option is not specified, the
        template's properties are used.
        """
        from .io import write_mslice
        return write_mslice(self, f, template, slice_thickness=slice_thickness,
                            scan_points=scan_points, scan_extent=scan_extent,
                            conv_angle=conv_angle, energy=energy, defocus=defocus,
                            noise_sigma=noise_sigma, tilt=tilt, tds=tds, n_cells=n_cells)

    def write_qe(self, f: FileOrPath, pseudo: t.Optional[t.Mapping[str, str]] = None):
        """
        Write a structure to a Quantum Espresso pw.x file.

        Args:
          f: File or path to write to
          pseudo: Mapping from atom symbol
        """
        from .io import write_qe
        write_qe(self, f, pseudo)