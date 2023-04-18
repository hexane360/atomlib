"""
IO for XCrySDen's XSF format. (http://www.xcrysden.org/doc/XSF.html)
"""

from __future__ import annotations

from dataclasses import dataclass
from io import TextIOBase
import logging
import typing as t

import numpy
from numpy.typing import NDArray
import polars

from ..transform import LinearTransform3D
from ..util import open_file, FileOrPath

Periodicity = t.Literal['crystal', 'slab', 'polymer', 'molecule']

if t.TYPE_CHECKING:
    from ..atoms import HasAtoms
    from ..atomcell import HasAtomCell


_PBCS: t.Dict[Periodicity, NDArray[numpy.bool_]] = {
    'molecule': numpy.array([0, 0, 0], dtype=numpy.bool_),
    'polymer': numpy.array([1, 0, 0], dtype=numpy.bool_),
    'slab': numpy.array([1, 1, 0], dtype=numpy.bool_),
    'crystal': numpy.array([1, 1, 1], dtype=numpy.bool_),
}


def _periodicity_to_pbc(periodicity: Periodicity) -> NDArray[numpy.bool_]:
    try:
        return _PBCS[periodicity]
    except KeyError:
        raise ValueError(f"Unknown XSF periodicity '{periodicity}'") from None


def _pbc_to_periodicity(pbc: NDArray[numpy.bool_]) -> Periodicity:
    n = numpy.count_nonzero(pbc)
    if n == 0:
        return 'molecule'
    if n == 3:
        return 'crystal'
    # only return 'polymer' for [1, 0, 0]
    # and 'slab' for [1, 1, 0]
    if n == 1 and pbc[0]:
        return 'polymer'
    if n == 2 and ~pbc[2]:
        return 'slab'
    return 'molecule'


@dataclass
class XSF:
    periodicity: Periodicity = 'crystal'
    primitive_cell: t.Optional[LinearTransform3D] = None
    conventional_cell: t.Optional[LinearTransform3D] = None

    prim_coords: t.Optional[polars.DataFrame] = None
    conv_coords: t.Optional[polars.DataFrame] = None
    atoms: t.Optional[polars.DataFrame] = None

    def get_atoms(self) -> polars.DataFrame:
        if self.prim_coords is not None:
            return self.prim_coords
        if self.atoms is not None:
            return self.atoms
        if self.conv_coords is not None:
            raise NotImplementedError()  # TODO untransform conv_coords by conventional_cell?
        raise ValueError("No coordinates specified in XSF file.")

    def get_pbc(self) -> NDArray[numpy.bool_]:
        return _periodicity_to_pbc(self.periodicity)

    @staticmethod
    def from_cell(cell: HasAtomCell) -> XSF:
        ortho = cell.to_ortho().to_linear()
        return XSF(
            primitive_cell=ortho,
            conventional_cell=ortho,
            prim_coords=cell.get_atoms('local').inner,
            periodicity=_pbc_to_periodicity(cell.pbc)
        )

    @staticmethod
    def from_atoms(atoms: HasAtoms) -> XSF:
        return XSF(
            periodicity='molecule',
            atoms=atoms.get_atoms('local').inner
        )

    @staticmethod
    def from_file(file: FileOrPath) -> XSF:
        logging.info(f"Loading XSF {file.name if hasattr(file, 'name') else file!r}...")  # type: ignore
        with open_file(file) as f:
            return XSFParser(f).parse()

    def __post_init__(self):
        if self.prim_coords is None and self.conv_coords is None and self.atoms is None:
            raise ValueError("Error: No coordinates are specified (atoms, primitive, or conventional).")

        if self.prim_coords is not None and self.conv_coords is not None:
            logging.warn("Warning: Both 'primcoord' and 'convcoord' are specified. 'convcoord' will be ignored.")
        elif self.conv_coords is not None and self.conventional_cell is None:
            raise ValueError("If 'convcoord' is specified, 'convvec' must be specified as well.")

        if self.periodicity == 'molecule':
            if self.atoms is None:
                raise ValueError("'atoms' must be specified for molecules.")

    def write(self, path: FileOrPath):
        with open_file(path, 'w') as f:
            print(self.periodicity.upper(), file=f)
            if self.primitive_cell is not None:
                print('PRIMVEC', file=f)
                self._write_cell(f, self.primitive_cell)
            if self.conventional_cell is not None:
                print('CONVVEC', file=f)
                self._write_cell(f, self.conventional_cell)
            print(file=f)

            if self.prim_coords is not None:
                print("PRIMCOORD", file=f)
                print(f"{len(self.prim_coords)} 1", file=f)
                self._write_coords(f, self.prim_coords)
            if self.conv_coords is not None:
                print("CONVCOORD", file=f)
                print(f"{len(self.conv_coords)} 1", file=f)
                self._write_coords(f, self.conv_coords)
            if self.atoms is not None:
                print("ATOMS", file=f)
                self._write_coords(f, self.atoms)

    def _write_cell(self, f: TextIOBase, cell: LinearTransform3D):
        for row in cell.inner.T:
            for val in row:
                f.write(f"{val:12.7f}")
            f.write('\n')

    def _write_coords(self, f: TextIOBase, coords: polars.DataFrame):
        for (elem, x, y, z) in coords.select(['elem', 'x', 'y', 'z']).rows():
            print(f"{elem:2d} {x:11.6f} {y:11.6f} {z:11.6f}", file=f)
        print(file=f)


class XSFParser:
    def __init__(self, file: TextIOBase):
        self._file: TextIOBase = file
        self._peek_line: t.Optional[str] = None
        self.lineno = 0

    def skip_line(self, line: t.Optional[str]) -> bool:
        return line is None or line.isspace() or line.lstrip().startswith('#')

    def peek_line(self) -> t.Optional[str]:
        try:
            while self.skip_line(self._peek_line):
                self._peek_line = next(self._file)
                self.lineno += 1
            return self._peek_line
        except StopIteration:
            return None

    def next_line(self) -> t.Optional[str]:
        line = self.peek_line()
        self._peek_line = None
        return line

    def parse_atoms(self, expected_length: t.Optional[int] = None) -> polars.DataFrame:
        zs = []
        coords = []
        words = None

        while (line := self.peek_line()):
            words = line.split()
            if len(words) == 0:
                continue
            if words[0].isalpha():
                break
            self.next_line()
            try:
                z = int(words[0])
                if z < 0 or z > 118:
                    raise ValueError()
            except (ValueError, TypeError):
                raise ValueError(f"Invalid atomic number '{words[0]}'") from None

            try:
                coords.append(numpy.array(list(map(float, words[1:]))))
                zs.append(z)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid atomic coordinates '{' '.join(words[1:])}'") from None

        if expected_length is not None:
            if not expected_length == len(zs):
                logging.warn(f"Warning: List length {len(zs)} doesn't match declared length {expected_length}")
        elif len(zs) == 0:
            raise ValueError(f"Expected atom list after keyword 'ATOMS'. Got '{line or 'EOF'}' instead.")

        if len(zs) == 0:
            return polars.DataFrame({}, schema=['elem', 'x', 'y', 'z'])  # type: ignore

        coord_lens = list(map(len, coords))
        if not all(l == coord_lens[0] for l in coord_lens[1:]):
            raise ValueError("Mismatched atom dimensions.")
        if coord_lens[0] < 3:
            raise ValueError("Expected at least 3 coordinates per atom.")

        coords = numpy.stack(coords, axis=0)[:, :3]
        (x, y, z) = map(lambda a: a[:, 0], numpy.split(coords, 3, axis=1))

        return polars.DataFrame({'elem': zs, 'x': x, 'y': y, 'z': z})

    def parse_coords(self) -> polars.DataFrame:
        line = self.next_line()
        if line is None:
            raise ValueError("Unexpected EOF before atom list")
        words = line.split()
        try:
            if not len(words) == 2:
                raise ValueError()
            (n, _) = map(int, words)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid atom list length: {line}") from None

        return self.parse_atoms(n)

    def parse_lattice(self) -> LinearTransform3D:
        rows = []
        for _ in range(3):
            line = self.next_line()
            if line is None:
                raise ValueError("Unexpected EOF in vector section.")
            words = line.split()
            try:
                if not len(words) == 3:
                    raise ValueError()
                row = numpy.array(list(map(float, words)))
                rows.append(row)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid lattice vector: {line}") from None

        matrix = numpy.stack(rows, axis=-1)
        return LinearTransform3D(matrix)

    def eat_sandwich(self, keyword: str):
        begin_keyword = 'begin_' + keyword
        end_keyword = 'end_' + keyword
        lineno = self.lineno

        while (line := self.next_line()):
            keyword = line.lstrip().split(maxsplit=1)[0].lower()
            if keyword.lower() == begin_keyword:
                # recurse to inner (identical) section
                self.eat_sandwich(keyword)
                continue
            if keyword.lower() == end_keyword:
                break
        else:
            raise ValueError(f"Unclosed section '{keyword}' opened at line {lineno}")

    def parse(self) -> XSF:
        data: t.Dict[str, t.Any] = {}
        periodicity: Periodicity = 'molecule'

        while (line := self.next_line()):
            keyword = line.lstrip().split(maxsplit=1)[0].lower()
            logging.debug(f"Parsing keyword {keyword}")

            if keyword == 'animsteps':
                raise ValueError("Animated XSF files are not supported.")
            elif keyword == 'atoms':
                data['atoms'] = self.parse_atoms()
            elif keyword in ('primcoord', 'convcoord'):
                data[keyword] = self.parse_coords()
            elif keyword in ('primvec', 'convvec'):
                data[keyword] = self.parse_lattice()
            elif keyword in ('crystal', 'slab', 'polymer', 'molecule'):
                periodicity = t.cast(Periodicity, keyword)
            elif keyword.startswith('begin_'):
                self.eat_sandwich(keyword.removeprefix('begin_'))
            elif keyword.startswith('end_'):
                raise ValueError(f"Unopened section close keyword '{keyword}'")
            else:
                raise ValueError(f"Unexpected keyword '{keyword.upper()}'.")

        if len(data) == 0:
            raise ValueError("Unexpected EOF while parsing XSF file.")

        # most validation is performed in XSF
        return XSF(
            periodicity, atoms=data.get('atoms'),
            prim_coords=data.get('primcoord'),
            conv_coords=data.get('convcoord'),
            primitive_cell=data.get('primvec'),
            conventional_cell=data.get('convvec'),
        )
