"""
IO for AtomEye's CFG file format, described here: http://li.mit.edu/Archive/Graphics/A/index.html#standard_CFG
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from io import TextIOBase
import re
import typing as t

import numpy
import polars

from ..transform import LinearTransform3D
from ..util import FileOrPath, open_file, map_some
from ..elem import get_elem
from ..atoms import HasAtoms
from ..atomcell import HasAtomCell

@dataclass
class CFG:
    atoms: polars.DataFrame

    cell: LinearTransform3D
    transform: t.Optional[LinearTransform3D] = None
    eta: t.Optional[LinearTransform3D] = None

    length_scale: t.Optional[float] = None
    length_unit: t.Optional[str] = None
    rate_scale: t.Optional[float] = None
    rate_unit: t.Optional[str] = None

    @staticmethod
    def from_file(file: FileOrPath) -> CFG:
        with open_file(file, 'r') as f:
            return CFGParser(f).parse()

    @staticmethod
    def from_atoms(atoms: HasAtoms) -> CFG:
        if isinstance(atoms, HasAtomCell):
            cell = atoms.get_transform('cell_box').inverse().to_linear()
            atoms = atoms.get_atoms('cell_box')
        else:
            cell = LinearTransform3D.identity()

        # ensure we have masses and velocities
        atoms = atoms.with_mass().with_velocity()
        return CFG(atoms._get_frame(), cell, length_scale=1.0, length_unit="Angstrom")

    def write(self, file: FileOrPath):
        with open_file(file, 'w', newline='\r\n') as f:
            f.write(f"Number of particles = {len(self.atoms)}\n")

            if self.length_scale is not None:
                unit = f" [{self.length_unit}]" if self.length_unit is not None else ""
                f.write(f"\nA = {self.length_scale:.8}{unit}\n\n")

            cell = self.cell.inner
            for (i, j) in product(range(3), repeat=2):
                f.write(f"H0({i+1},{j+1}) = {cell[j,i]:.8} A\n")

            if self.transform is not None:
                f.write("\n")
                transform = self.transform.inner
                for (i, j) in product(range(3), repeat=2):
                    f.write(f"Transform({i+1},{j+1}) = {transform[j,i]:.8}\n")

            if self.eta is not None:
                f.write("\n")
                eta = self.eta.inner
                for i in range(3):
                    for j in range(i, 3):
                        f.write(f"eta({i+1},{j+1}) = {eta[j,i]:.8}\n")

            if self.rate_scale is not None:
                unit = f" [{self.rate_unit}]" if self.rate_unit is not None else ""
                f.write(f"\nR = {self.rate_scale:.8}{unit}\n")

            f.write("\n")
            for row in self.atoms.select(('mass', 'symbol', 'x', 'y', 'z', 'v_x', 'v_y', 'v_z')).rows():
                (mass, sym, x, y, z, v_x, v_y, v_z) = row
                f.write(f"{mass:.4f} {sym:>2} {x:.8} {y:.8} {z:.8} {v_x:.8} {v_y:.8} {v_z:.8}\n")


TAGS: t.FrozenSet[str] = frozenset(map(lambda s: s.lower(), (
    'Number of particles',
    'A',
    'R',
)))

ARRAY_TAGS: t.FrozenSet[str] = frozenset(map(lambda s: s.lower(), (
    'H0',
    'Transform',
    'eta',
)))


class CFGParser:
    def __init__(self, f: TextIOBase):
        self.buf = LineBuffer(f)

    def parse(self) -> CFG:
        (n, value_tags, array_tags) = self.parse_tags()
        atoms = self.parse_atoms(n)

        try:
            cell = array_tags['h0']
        except KeyError:
            raise ValueError("CFG file missing required tag 'H0'") from None

        length = value_tags.get('a')
        length_scale = map_some(lambda t: t[0], length)
        length_unit = map_some(lambda t: t[1], length)

        rate = value_tags.get('r')
        rate_scale = map_some(lambda t: t[0], rate)
        rate_unit = map_some(lambda t: t[1], rate)

        return CFG(
            atoms=atoms,
            cell=LinearTransform3D(cell),
            transform=map_some(LinearTransform3D, array_tags.get('transform')),
            eta=map_some(LinearTransform3D, array_tags.get('eta')),
            length_scale=length_scale,
            length_unit=length_unit,
            rate_scale=rate_scale,
            rate_unit=rate_unit,
        )

    def parse_tags(self) -> t.Tuple[int, t.Dict[str, t.Tuple[float, t.Optional[str]]], t.Dict[str, numpy.ndarray]]:
        first = True

        # tag, (value, unit)
        n: t.Optional[int] = None
        value_tags: t.Dict[str, t.Tuple[float, t.Optional[str]]] = {}
        array_tags: t.Dict[str, t.List[t.List[t.Optional[float]]]] = {}

        while (line := self.buf.peek()) is not None:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                # skip comments and blank lines
                self.buf.next()
                continue

            if first:
                if not line.lower().startswith('number of particles'):
                    raise ValueError("File does not start with Number of particles."
                                    " Is this an AtomEye CFG file?")

            try:
                tag, value = line.split('=')
            except ValueError:
                try:
                    float(line.split(' ', 1)[0])
                    # started list of atoms
                    break
                except ValueError:
                    raise ValueError(f"Expected a tag-value pair at line {self.buf.line}: '{line}'")

            tag = tag.strip()
            value = value.strip()
            if first:
                try:
                    value = int(value)
                except ValueError:
                    raise ValueError(f"Invalid # of elements '{value}' at line {self.buf.line}") from None
                n = value
                first = False
                self.buf.next()
                continue

            if tag.lower() in TAGS:
                try:
                    value_tags[tag.lower()] = self.parse_value_with_unit(value)
                except ValueError:
                    raise ValueError(f"Invalid value '{value}' at line {self.buf.line}") from None
            elif (match := re.match(r'(.+)\((\d+),(\d+)\)', tag)):
                try:
                    (tag, i, j) = (match[1].lower(), int(match[2]), int(match[3]))
                    if not (0 < i <= 3 and 0 < j <= 3):
                        raise ValueError(f"Invalid index ({i},{j}) for tag '{tag}' at line {self.buf.line}")
                    if tag not in array_tags:
                        array_tags[tag] = [[None] * 3, [None] * 3, [None] * 3]
                    try:
                        val = self.parse_value_with_unit(value)[0]
                        array_tags[tag][j-1][i-1] = val
                        if tag == 'eta':
                            array_tags[tag][i-1][j-1] = val
                    except ValueError:
                        raise ValueError(f"Invalid value '{value}' at line {self.buf.line}") from None
                except ValueError:
                    raise ValueError(f"Invalid indexes in tag '{tag}' at line {self.buf.line}") from None
                if tag.lower() not in ARRAY_TAGS:
                    raise ValueError(f"Unknown array tag '{tag}'")
            elif tag.lower() in ARRAY_TAGS:
                raise ValueError(f"Missing indexes for tag '{tag}' at line {self.buf.line}")
            else:
                raise ValueError(f"Unknown tag '{tag}' at line {self.buf.line}")

            self.buf.next()

        if n is None:
            raise ValueError("Empty CFG file")

        ndarray_tags: t.Dict[str, numpy.ndarray] = {}

        for (tag, value) in array_tags.items():
            for i in range(3):
                for j in range(3):
                    if value[j][i] is None:
                        raise ValueError(f"Tag '{tag}' missing value for index ({i+1},{j+1})")
            ndarray_tags[tag] = numpy.array(value)

        return (n, value_tags, ndarray_tags)

    def parse_value_with_unit(self, value: str) -> t.Tuple[float, t.Optional[str]]:
        segments = value.split(maxsplit=1)
        if len(segments) == 1:
            return (float(value), None)
        value, unit = map(lambda s: s.strip(), segments)

        if (match := re.match(r'\[(.+)\]', unit)):
            unit = str(match[1])
        else:
            unit = unit.split(maxsplit=1)[0]

        return (float(value), unit)

    def parse_atoms(self, n: int) -> polars.DataFrame:
        rows = []
        columns = (
            ('elem', polars.Int8), ('symbol', polars.Utf8),
            ('x', polars.Float64), ('y', polars.Float64), ('z', polars.Float64),
            ('v_x', polars.Float64), ('v_y', polars.Float64), ('v_z', polars.Float64),
            ('mass', polars.Float64),
        )

        while (line := self.buf.peek()) is not None:
            self.buf.next()
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                # skip comments and blank lines
                continue

            cols = line.split()
            if len(cols) != 8:
                raise ValueError(f"Misformatted row '{line}' at line {self.buf.line}")

            (mass, sym, x, y, z, v_x, v_y, v_z) = cols
            try:
                (mass, x, y, z, v_x, v_y, v_z) = map(float, (mass, x, y, z, v_x, v_y, v_z))
            except ValueError:
                raise ValueError(f"Invalid values at line {self.buf.line}")
            try:
                elem = get_elem(sym)
            except ValueError:
                raise ValueError(f"Invalid atomic symbol '{sym}' at line {self.buf.line}")

            rows.append((elem, sym, x, y, z, v_x, v_y, v_z, mass))

        if n != len(rows):
            raise ValueError(f"# of atom rows doesn't match declared number ({len(rows)} vs. {n})")

        return polars.DataFrame(rows, schema=columns, orient='row')  # type: ignore


class LineBuffer:
    def __init__(self, f: TextIOBase):
        self.inner = iter(f)
        self.line: int = 0
        self._peek: t.Optional[str] = None

    def next(self):
        self._peek = None

    def peek(self) -> t.Optional[str]:
        if self._peek is None:
            try:
                self._peek = next(self.inner)
                self.line += 1
            except StopIteration:
                pass
        return self._peek