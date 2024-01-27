"""
IO for the informal XYZ file format.

Partially supports the extended XYZ format, described here: https://atomsk.univ-lille.fr/doc/en/format_xyz.html
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice
import warnings
import re
import io
import logging
import typing as t

import numpy
from numpy.typing import NDArray
import polars
from polars.exceptions import PolarsPanicError  # type: ignore

from ..util import open_file, open_file_binary, BinaryFileOrPath, FileOrPath
from ..elem import get_sym, get_elem
from ..atoms import HasAtoms
from ..atomcell import HasAtomCell

_EXT_TOKEN_RE = re.compile(r"(=|\s+|\")")
_PROP_TYPE_MAP: t.Dict[str, t.Type[polars.DataType]] = {
    's': polars.Utf8,
    'r': polars.Float64,
    'i': polars.Int64,
}
# map OVITO property names into our standard names
# see: https://www.ovito.org/manual/reference/file_formats/input/xyz.html
_PROP_NAME_MAP: t.Dict[str, str] = {
    'pos': '',  # turns into 'x', 'y', 'z'
    'species': 'symbol', 'element': 'symbol',
    'vel': 'v', 'velo': 'v',
    'atom_types': 'type',
    'id': 'i',
    'transparency': 'frac_occupancy',
}
_PROP_NAME_UNMAP: t.Dict[str, str] = {
    'frac_occupancy': 'transparency',
    'symbol': 'species',
    'i': 'id',
}


XYZFormat = t.Literal['xyz', 'exyz']

T = t.TypeVar('T')


class XYZToCSVReader(io.IOBase):
    def __init__(self, inner: io.IOBase):
        self.inner = inner

    def read(self, n: int) -> bytes:  # type: ignore
        buf = self.inner.read(n)
        buf = re.sub(rb'[ \t]+', rb'\t', buf)
        return buf

    def seek(self, offset: int, whence: int = 0, /) -> int:  # pragma: nocover
        return self.inner.seek(offset, whence)

    def __getattr__(self, name: str):
        if name in ('name', 'getvalue'):
            # don't let polars steal the buffer from us
            raise AttributeError()
        return getattr(self.inner, name)


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
    def from_file(file: BinaryFileOrPath) -> XYZ:
        logging.info(f"Loading XYZ {file.name if hasattr(file, 'name') else file!r}...")  # type: ignore

        with open_file_binary(file, 'r') as f:
            try:
                # TODO be more gracious about whitespace here
                length = int(f.readline())
            except ValueError:
                raise ValueError(f"Error parsing XYZ file: Invalid length") from None
            except IOError as e:
                raise IOError(f"Error parsing XYZ file: {e}") from None

            comment = f.readline().rstrip(b'\n').decode('utf-8')
            # TODO handle if there's not a gap here

            try:
                params = ExtXYZParser(comment).parse()
            except ValueError:
                params = None

            column_names, column_types = _get_columns_from_params(params)

            f = XYZToCSVReader(f)
            df: polars.DataFrame = polars.read_csv(
                f, separator='\t',  # type: ignore
                new_columns=column_names,
                dtypes=column_types,
                has_header=False, use_pyarrow=False
            )
            if len(df.columns) > 4:
                raise ValueError("Error parsing XYZ file: Extra columns in at least one row.")

            #TODO .fill_null(Series) seems to be broken on polars 0.14.11
            # map atomic numbers -> symbols (on columns whichare UInt8)
            sym = get_sym(df.select(polars.col('symbol').cast(polars.UInt8, strict=False)).to_series())
            df = df.with_columns(
                polars.when(sym.is_null())  # type: ignore
                .then(polars.col('symbol'))
                .otherwise(sym)  # type: ignore
                .alias('symbol'))

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
                    f"{val:< {space}.8f}" if isinstance(val, float) else f"{val:<{space}}" for (val, space) in zip(row, col_space)
                ).strip() + '\n' for row in self.atoms.select(('symbol', 'x', 'y', 'z')).rows()
            )

    def cell_matrix(self) -> t.Optional[NDArray[numpy.float_]]:
        if (s := self.params.get('Lattice')) is None:
            return None

        try:
            items = list(map(float, s.split()))
            if not len(items) == 9:
                raise ValueError("Invalid length")
            return numpy.array(items, dtype=numpy.float_).reshape((3, 3)).T
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


def _get_columns_from_params(params: t.Optional[t.Dict[str, str]]) -> t.Tuple[t.List[str], t.List[t.Type[polars.DataType]]]:
    if params is None or (s := params.get('Properties')) is None:
        return (
            ['symbol', 'x', 'y', 'z'],
            [polars.Utf8, polars.Float64, polars.Float64, polars.Float64]
        )

    try:
        segs = s.split(':')
        if len(segs) % 3:
            raise ValueError()

        col_names = []
        col_types = []
        for (name, ty, num) in batched(segs, 3):
            num = int(num)
            if num < 1:
                raise ValueError()

            name = _PROP_NAME_REMAP.get(name, name)
            ty = _PROP_TYPE_MAP[ty.lower()]

            if num == 1:
                col_names.append(name)
                col_types.append(ty)
                continue

            suffixes = ('x', 'y', 'z') if num == 3 else range(num)
            col_names.extend(f"{name}_{c}".lstrip('_') for c in suffixes)
            col_types.extend(ty for _ in suffixes)

        if not all(c in col_names for c in ('symbol', 'x', 'y', 'z')):
            raise ValueError("Properties string missing required columns.")

        return (col_names, col_types)

    except ValueError:
        raise ValueError(f"Improperly formmated 'Properties' parameter: {s}")


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
