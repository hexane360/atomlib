from __future__ import annotations

from dataclasses import dataclass, field
import warnings
import re
import logging
import typing as t

import numpy
import polars
import pandas

from ..util import open_file, FileOrPath

_COMMENT_RE = re.compile(r"(=|\s+|\")")

@dataclass
class XYZ:
    atoms: polars.DataFrame
    comment: t.Optional[str] = None
    params: t.Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_file(file: FileOrPath) -> XYZ:
        logging.info(f"Loading XYZ {file.name if hasattr(file, 'name') else file!r}...")  # type: ignore
        file = open_file(file, 'r')

        try:
            # TODO be more gracious about whitespace here
            length = int(file.readline())
        except ValueError:
            raise ValueError(f"Error parsing XYZ file: Invalid length") from None
        except IOError as e:
            raise IOError(f"Error parsing XYZ file: {e}") from None

        comment = file.readline().rstrip('\n')  # TODO handle if there's not a gap here

        #df = polars.read_csv(file, sep=r' ', has_header=False, new_columns=['symbol', 'x', 'y', 'z'])
        df = pandas.read_table(file, sep=r'\s+', header=None,  # type: ignore
                               names=['symbol', 'x', 'y', 'z'])
        df = polars.from_pandas(df)

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

    def write(self, file: FileOrPath):
        file = open_file(file, 'w', newline='\r\n')

        file.write(f"{len(self.atoms)}\n")
        if len(self.params) > 0:
            file.write(" ".join(param_strings(self.params)))
        else:
            file.write(self.comment or "")
        file.write("\n")

        # not my best work
        col_space = (3, 12, 12, 12)
        file.writelines(
            "".join(
                f"{val:< {space}.8f}" if isinstance(val, float) else f"{val:<{space}}" for (val, space) in zip(row, col_space)
                ) + '\n' for row in self.atoms.select(('symbol', 'x', 'y', 'z')).rows()
        )
        #self.atoms.to_string(file, columns=['symbol', 'x', 'y', 'z'], col_space=[4, 12, 12, 12],
        #                     header=False, index=False, justify='left', float_format='{:.8f}'.format)

    def cell_matrix(self) -> t.Optional[numpy.ndarray]:
        if self.params is None or 'Lattice' not in self.params:
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