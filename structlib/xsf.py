from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import TextIOBase
from multiprocessing.sharedctypes import Value
from pathlib import Path
import re
import logging
import typing as t

import numpy
from numpy.typing import NDArray
import pandas

from .transform import LinearTransform
from .util import open_file, FileOrPath

Periodicity = t.Union[t.Literal['crystal'], t.Literal['slab'], t.Literal['polymer'], t.Literal['molecule']]


@dataclass
class XSF:
    periodicity: Periodicity = 'crystal'
    primitive_cell: t.Optional[LinearTransform] = None
    conventional_cell: t.Optional[LinearTransform] = None

    prim_coords: t.Optional[pandas.DataFrame] = None
    conv_coords: t.Optional[pandas.DataFrame] = None
    atoms: t.Optional[pandas.DataFrame] = None

    def get_atoms(self) -> pandas.DataFrame:
        if self.prim_coords is not None:
            return self.prim_coords
        if self.atoms is not None:
            return self.atoms
        if self.conv_coords is not None:
            raise NotImplementedError()  # TODO untransform conv_coords by conventional_cell?
        raise ValueError("No coordinates specified in XSF file.")

    @staticmethod
    def from_file(file: FileOrPath) -> XSF:
        logging.info(f"Loading XSF {file.name if hasattr(file, 'name') else file!r}...")  # type: ignore
        file = open_file(file)
        return XSFParser(file).parse()

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

    def parse_atoms(self, expected_length: t.Optional[int] = None) -> pandas.DataFrame:
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
            return pandas.DataFrame({}, columns=['atomic_number', 'a', 'b', 'c'])

        coord_lens = list(map(len, coords))
        if not all(l == coord_lens[0] for l in coord_lens[1:]):
            raise ValueError("Mismatched atom dimensions.")
        if coord_lens[0] < 3:
            raise ValueError("Expected at least 3 coordinates per atom.")

        coords = numpy.stack(coords, axis=0)[:, :3]
        (x, y, z) = map(lambda a: a[:, 0], numpy.split(coords, 3, axis=1))

        return pandas.DataFrame({'atomic_number': zs, 'a': x, 'b': y, 'c': z})

    def parse_coords(self) -> pandas.DataFrame:
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

    def parse_lattice(self) -> LinearTransform:
        rows = []
        for i in range(3):
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

        matrix = numpy.stack(rows, axis=0)
        return LinearTransform(matrix)

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
