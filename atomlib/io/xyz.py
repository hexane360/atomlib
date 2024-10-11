"""
IO for the informal XYZ file format.

Supports the extended XYZ format, described [here](https://atomsk.univ-lille.fr/doc/en/format_xyz.html).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice
import warnings
import re
import logging
import typing as t

from typing_extensions import TypeAlias
import numpy
from numpy.typing import NDArray
import polars

from ..util import open_file, FileOrPath
from ..elem import get_sym, get_elem
from ..atoms import HasAtoms
from ..atomcell import HasAtomCell
from .util import parse_whitespace_separated

_EXT_TOKEN_RE = re.compile(r"(=|\s+|\")")
_PROP_TYPE_MAP: t.Dict[str, t.Type[polars.DataType]] = {
    's': polars.Utf8,
    'r': polars.Float64,
    'i': polars.Int64,
}
# map OVITO property names into our standard names
# see: https://www.ovito.org/manual/reference/file_formats/input/xyz.html
_PROP_NAME_MAP: t.Dict[str, str] = {
    'pos': 'coords',
    'species': 'symbol', 'element': 'symbol',
    'vel': 'velocity', 'velo': 'velocity',
    'atom_types': 'type',
    'id': 'i',
    'transparency': 'frac_occupancy',
}
_PROP_NAME_UNMAP: t.Dict[str, str] = {
    'frac_occupancy': 'transparency',
    'symbol': 'species',
    'i': 'id',
}


XYZFormat: TypeAlias = t.Literal['xyz', 'exyz']

T = t.TypeVar('T')


def _flatten(it: t.Iterable[t.Union[T, t.Iterable[T]]]) -> t.Iterator[T]:
    for val in it:
        if isinstance(val, (list, tuple)):
            yield from t.cast(t.Iterable[T], val)
        else:
            yield t.cast(T, val)


@dataclass
class XYZ:
    atoms: polars.DataFrame
    comment: t.Optional[str] = None
    params: t.Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_atoms(atoms: HasAtoms) -> XYZ:
        params = {}
        if isinstance(atoms, HasAtomCell):
            coords = atoms.get_cell().to_ortho().to_linear().inner.ravel()
            lattice_str = " ".join((f"{c:.8f}" for c in coords))
            params['Lattice'] = lattice_str

            pbc_str = " ".join(str(int(v)) for v in atoms.get_cell().pbc)
            params['pbc'] = pbc_str

        return XYZ(
            atoms.get_atoms('local')._get_frame(),
            params=params
        )

    @staticmethod
    def from_file(file: FileOrPath) -> XYZ:
        logging.info(f"Loading XYZ {file.name if hasattr(file, 'name') else file!r}...")  # type: ignore

        with open_file(file, 'r') as f:
            try:
                # TODO be more gracious about whitespace here
                length = int(f.readline())
            except ValueError:
                raise ValueError("Error parsing XYZ file: Invalid length") from None
            except IOError as e:
                raise IOError(f"Error parsing XYZ file: {e}") from None

            comment = f.readline().rstrip('\n')
            # TODO handle if there's not a gap here

            try:
                params = ExtXYZParser(comment).parse()
            except ValueError:
                params = None

            schema = _get_columns_from_params(params)

            df = parse_whitespace_separated(f, schema, start_line=1)

            # map atomic numbers -> symbols (on columns which are Int8)
            df = df.with_columns(
                get_sym(df.select(polars.col('symbol').cast(polars.Int8, strict=False)).to_series())
                    .fill_null(df['symbol']).alias('symbol')
            )
            # ensure all symbols are recognizable (this will raise ValueError if not)
            get_elem(df['symbol'])

            if length < len(df):
                warnings.warn(f"Warning: truncating structure of length {len(df)} "
                            f"to match declared length of {length}")
                df = df[:length]
            elif length > len(df):
                warnings.warn(f"Warning: structure length {len(df)} doesn't match "
                            f"declared length {length}.\nData could be corrupted.")

            try:
                params = ExtXYZParser(comment).parse()
                return XYZ(df, comment, params)
            except ValueError:
                pass

            return XYZ(df, comment)

    def write(self, file: FileOrPath, fmt: XYZFormat = 'exyz'):
        with open_file(file, 'w', newline='\r\n') as f:

            f.write(f"{len(self.atoms)}\n")
            if len(self.params) > 0 and fmt == 'exyz':
                f.write(" ".join(_param_strings(self.params)))
            else:
                f.write(self.comment or "")
            f.write("\n")

            # not my best work
            col_space = (3, 12, 12, 12)
            f.writelines(
                "".join(
                    f"{val:< {space}.8f}" if isinstance(val, float) else f"{val:<{space}}" for (val, space) in zip(_flatten(row), col_space)
                ).strip() + '\n' for row in self.atoms.select(('symbol', 'coords')).rows()
            )

    def cell_matrix(self) -> t.Optional[NDArray[numpy.float64]]:
        if (s := self.params.get('Lattice')) is None:
            return None

        try:
            items = list(map(float, s.split()))
            if not len(items) == 9:
                raise ValueError("Invalid length")
            return numpy.array(items, dtype=numpy.float64).reshape((3, 3)).T
        except ValueError:
            warnings.warn(f"Warning: Invalid format for key 'Lattice=\"{s}\"'")
        return None

    def pbc(self) -> t.Optional[NDArray[numpy.bool_]]:
        if (s := self.params.get('pbc')) is None:
            return None

        val_map = {'0': False, 'f': False, '1': True, 't': True}
        try:
            items = [val_map[v.lower()] for v in s.split()]
            if not len(items) == 3:
                raise ValueError("Invalid length")
            return numpy.array(items, dtype=numpy.bool_)
        except ValueError:
            warnings.warn(f"Warning: Invalid format for key 'pbc=\"{s}\"'")
        return None


def batched(iterable: t.Iterable[T], n: int) -> t.Iterator[t.Tuple[T, ...]]:
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def _param_strings(params: t.Dict[str, str]) -> t.Iterator[str]:
    for (k, v) in params.items():
        if any(c in k for c in (' ', '\t', '\n')):
            k = f'"{k}"'
        if any(c in v for c in (' ', '\t', '\n')):
            v = f'"{v}"'
        yield f"{k}={v}"


def _get_columns_from_params(params: t.Optional[t.Dict[str, str]]) -> t.Dict[str, t.Union[polars.DataType, t.Type[polars.DataType]]]:
    if params is None or (s := params.get('Properties')) is None:
        return {
            'symbol': polars.Utf8,
            'coords': polars.Array(polars.Float64, 3),
        }

    try:
        segs = s.split(':')
        if len(segs) % 3:
            raise ValueError()

        d = {}
        for (name, ty, num) in batched(segs, 3):
            num = int(num)
            if num < 1:
                raise ValueError()

            name = _PROP_NAME_MAP.get(name, name)
            ty = _PROP_TYPE_MAP[ty.lower()]

            d[name] = ty if num == 1 else polars.Array(ty, num)

        if not all(c in d.keys() for c in ('symbol', 'coords')):
            raise ValueError("Properties string missing required columns.")

        return d

    except ValueError:
        raise ValueError(f"Improperly formatted 'Properties' parameter: {s}")


class ExtXYZParser:
    def __init__(self, comment: str):
        self._tokens = list(filter(len, _EXT_TOKEN_RE.split(comment)))
        self._tokens.reverse()

    def peek(self) -> t.Optional[str]:
        return None if len(self._tokens) == 0 else self._tokens[-1]

    def next(self) -> str:
        return self._tokens.pop()

    def skip_wspace(self):
        word = self.peek()
        while word is not None and word.isspace():
            self.next()
            word = self.peek()

    def parse(self) -> t.Dict[str, str]:
        self.skip_wspace()
        d = {}
        while len(self._tokens) > 0:
            key = self.parse_val()
            eq = self.next()
            if not eq == "=":
                raise ValueError(f"Expected key-value separator, instead got '{eq}'")
            val = self.parse_val()
            d[key] = val
            self.skip_wspace()
        return d

    def parse_val(self) -> str:
        token = self.peek()
        if token == "=":
            raise ValueError("Expected value, instead got '='")
        if not token == "\"":
            return self.next()

        # quoted string
        self.next()
        words = []
        while not (word := self.peek()) == "\"":
            if word is None:
                raise ValueError("EOF while parsing string value")
            words += self.next()
        self.next()
        return "".join(words)
