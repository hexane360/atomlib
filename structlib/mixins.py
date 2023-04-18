
import abc
from pathlib import Path
import typing as t

from numpy.typing import ArrayLike

from .atoms import HasAtoms, HasAtomsT
from .atomcell import HasAtomCell
from .util import FileOrPath

if t.TYPE_CHECKING:
    # pyright: reportImportCycles=false
    from .io import CIF, XYZ, XSF, CFG, FileOrPath, FileType  # pragma: no cover
    from .io.mslice import MSliceTemplate                     # pragma: no cover


class AtomsIOMixin(HasAtoms, abc.ABC):
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

        Currently, supported file types are 'cif', 'xyz', and 'xsf'.
        If no `ty` is specified, it is inferred from the file's extension.
        """
        from .io import _cast_atoms, read
        return _cast_atoms(read(path, ty), cls)  # type: ignore

    @classmethod
    def read_cif(cls: t.Type[HasAtomsT], f: t.Union[FileOrPath, CIF]) -> HasAtomsT:
        """Read a structure from a CIF file."""
        from .io import _cast_atoms, read_cif
        return _cast_atoms(read_cif(f), cls)

    @classmethod
    def read_xyz(cls: t.Type[HasAtomsT], f: t.Union[FileOrPath, XYZ]) -> HasAtomsT:
        """Read a structure from an XYZ file."""
        from .io import _cast_atoms, read_xyz
        return _cast_atoms(read_xyz(f), cls)

    @classmethod
    def read_xsf(cls: t.Type[HasAtomsT], f: t.Union[FileOrPath, XSF]) -> HasAtomsT:
        """Read a structure from an XSF file."""
        from .io import _cast_atoms, read_xsf
        return _cast_atoms(read_xsf(f), cls)

    @classmethod
    def read_cfg(cls: t.Type[HasAtomsT], f: t.Union[FileOrPath, CFG]) -> HasAtomsT:
        """Read a structure from a CFG file."""
        from .io import _cast_atoms, read_cfg
        return _cast_atoms(read_cfg(f), cls)

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


class AtomCellIOMixin(HasAtomCell, AtomsIOMixin):
    def write_mslice(self, f: FileOrPath, template: t.Optional[MSliceTemplate] = None, *,
                 slice_thickness: t.Optional[float] = None,
                 scan_points: t.Optional[ArrayLike] = None,
                 scan_extent: t.Optional[ArrayLike] = None):
        """Write this structure to an mslice file."""
        from .io import write_mslice
        return write_mslice(self, f, template, slice_thickness=slice_thickness,
                            scan_points=scan_points, scan_extent=scan_extent)

    def write_qe(self, f: FileOrPath, pseudo: t.Optional[t.Mapping[str, str]] = None):
        from .io import write_qe
        write_qe(self, f, pseudo)