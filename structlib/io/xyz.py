"""
IO for the informal XYZ file format.

Partially supports the extended XYZ format, described here: https://atomsk.univ-lille.fr/doc/en/format_xyz.html
"""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings
import re
import io
import logging
import typing as t

import numpy
import polars
from polars.exceptions import PolarsPanicError

from ..util import open_file, open_file_binary, BinaryFileOrPath, FileOrPath
from ..elem import get_sym
from ..atoms import HasAtoms
from ..atomcell import HasAtomCell

_COMMENT_RE = re.compile(r"(=|\s+|\")")


XYZFormat = t.Literal['xyz', 'exyz']


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

            f= XYZToCSVReader(f)
            df = polars.read_csv(f, separator='\t',  # type: ignore
                                new_columns=['symbol', 'x', 'y', 'z'],
                                dtypes=[polars.Utf8, polars.Float64, polars.Float64, polars.Float64],
                                has_header=False, use_pyarrow=False)
            if len(df.columns) > 4:
                raise ValueError("Error parsing XYZ file: Extra columns in at least one row.")

            try:
                #TODO .fill_null(Series) seems to be broken on polars 0.14.11
                sym = df.select(polars.col('symbol')
                                      .cast(polars.UInt8, strict=False)
                                      .map(get_sym, return_dtype=polars.Utf8)).to_series()
                df = df.with_columns(
                    polars.when(sym.is_null())
                    .then(polars.col('symbol'))
                    .otherwise(sym)
                    .alias('symbol'))
            except PolarsPanicError:
                invalid = (polars.col('symbol').cast(polars.UInt8) > 118).first()
                raise ValueError(f"Invalid atomic number {invalid}") from None

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
                f.write(" ".join(param_strings(self.params)))
            else:
                f.write(self.comment or "")
            f.write("\n")

            # not my best work
            col_space = (3, 12, 12, 12)
            f.writelines(
                "".join(
                    f"{val:< {space}.8f}" if isinstance(val, float) else f"{val:<{space}}" for (val, space) in zip(row, col_space)
                    ) + '\n' for row in self.atoms.select(('symbol', 'x', 'y', 'z')).rows()
            )

    def cell_matrix(self) -> t.Optional[numpy.ndarray]:
        if 'Lattice' not in self.params:
            return None
        s = self.params['Lattice']
        try:
            items = list(map(float, s.split()))
            if not len(items) == 9:
                raise ValueError("Invalid length")
            return numpy.array(items).reshape((3, 3)).T
        except ValueError:
            warnings.warn(f"Warning: Invalid format for key 'Lattice=\"{s}\"'")


def param_strings(params: t.Dict[str, str]) -> t.Iterator[str]:
    for (k, v) in params.items():
        if any(c in k for c in (' ', '\t', '\n')):
            k = f'"{k}"'
        if any(c in v for c in (' ', '\t', '\n')):
            v = f'"{v}"'
        yield f"{k}={v}"


class ExtXYZParser:
    def __init__(self, comment: str):
        self._tokens = list(filter(len, _COMMENT_RE.split(comment)))
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
