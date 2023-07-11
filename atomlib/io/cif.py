"""
IO for the CIF1.1 file format, specified here: https://www.iucr.org/resources/cif/spec/version1.1
"""

from __future__ import annotations

from dataclasses import dataclass
from io import TextIOBase, StringIO
from itertools import repeat
import operator
import re
import logging
import typing as t

import numpy
import polars
from numpy.typing import NDArray

from ..transform import AffineTransform3D
from ..expr import Parser, BinaryOp, BinaryOrUnaryOp, sub
from ..util import open_file, FileOrPath


Value = t.Union[int, float, str, None]
_INT_RE = re.compile(r'[-+]?\d+')
# float regex with uncertainty
_FLOAT_RE = re.compile(r'([-+]?\d*(\.\d*)?(e[-+]?\d+)?)(\(\d+\))?', re.I)


@dataclass
class CIF:
    name: t.Optional[str]
    data: t.Dict[str, t.Union[t.List[Value], Value]]

    @staticmethod
    def from_file(file: FileOrPath) -> t.Iterator[CIF]:
        with open_file(file) as f:
            yield from CifReader(f).parse()

    def stack_tags(self, *tags: str, dtype: t.Union[str, numpy.dtype, t.Iterable[t.Union[str, numpy.dtype]], None] = None,
                   rename: t.Optional[t.Iterable[t.Optional[str]]] = None, required: t.Union[bool, t.Iterable[bool]] = True) -> polars.DataFrame:
        dtypes: t.Iterable[t.Optional[numpy.dtype]]
        if dtype is None:
            dtypes = repeat(None)
        elif isinstance(dtype, (numpy.dtype, str)):
            dtypes = (numpy.dtype(dtype),) * len(tags)
        else:
            dtypes = tuple(map(lambda ty: numpy.dtype(ty), dtype))
            if len(dtypes) != len(tags):
                raise ValueError(f"dtype list of invalid length")

        if isinstance(required, bool):
            required = repeat(required)

        if rename is None:
            rename = repeat(None)

        d = {}
        for (tag, ty, req, name) in zip(tags, dtypes, required, rename):
            if tag not in self.data:
                if req:
                    raise ValueError(f"Tag '{tag}' missing from CIF file")
                continue
            try:
                arr = numpy.array(self.data[tag], dtype=ty)
                d[name or tag] = arr
            except TypeError:
                raise TypeError(f"Tag '{tag}' of invalid or heterogeneous type.")

        if len(d) == 0:
            return polars.DataFrame({})

        l = len(next(iter(d.values())))
        if any(len(arr) != l for arr in d.values()):
            raise ValueError(f"Tags of mismatching lengths: {tuple(map(len, d.values()))}")

        return polars.DataFrame(d)

    def cell_size(self) -> t.Optional[t.Tuple[float, float, float]]:
        """Return cell size (in angstroms)."""
        try:
            a = float(self['cell_length_a'])  # type: ignore
            b = float(self['cell_length_b'])  # type: ignore
            c = float(self['cell_length_c'])  # type: ignore
            return (a, b, c)
        except (ValueError, TypeError, KeyError):
            return None

    def cell_angle(self) -> t.Optional[t.Tuple[float, float, float]]:
        """Return cell angle (in degrees)."""
        try:
            a = float(self['cell_angle_alpha'])  # type: ignore
            b = float(self['cell_angle_beta'])   # type: ignore
            g = float(self['cell_angle_gamma'])  # type: ignore
            return (a, b, g)
        except (ValueError, TypeError, KeyError):
            return None

    def get_symmetry(self) -> t.Iterator[AffineTransform3D]:
        syms = self.data.get('symmetry_equiv_pos_as_xyz', None)
        if syms is None:
            syms = ()
        if not hasattr(syms, '__iter__'):
            syms = (syms,)
        return map(parse_symmetry, map(str, syms))  # type: ignore

    def __getitem__(self, key: str) -> t.Union[Value, t.List[Value]]:
        return self.data.__getitem__(key)


class SymmetryVec:
    @classmethod
    def parse(cls, s: str) -> SymmetryVec:
        if s[0] in ('x', 'y', 'z'):
            a = numpy.zeros((4,))
            a[('x', 'y', 'z').index(s[0])] += 1.
            return cls(a)
        return cls(float(s))

    def __init__(self, val: t.Union[float, NDArray[numpy.floating]]):
       self.inner: t.Union[float, NDArray[numpy.floating]] = val

    def is_scalar(self) -> bool:
        return isinstance(self.inner, float)

    def to_vec(self) -> NDArray[numpy.floating]:
        if isinstance(self.inner, float):
            vec = numpy.zeros((4,))
            vec[3] = self.inner
            return vec
        return self.inner

    def __add__(self, rhs: SymmetryVec) -> SymmetryVec:
        if self.is_scalar() and rhs.is_scalar():
            return SymmetryVec(self.inner + rhs.inner)
        return SymmetryVec(rhs.to_vec() + self.to_vec())

    def __neg__(self) -> SymmetryVec:
        return SymmetryVec(-self.inner)

    def __sub__(self, rhs: SymmetryVec) -> SymmetryVec:
        if self.is_scalar() and rhs.is_scalar():
            return SymmetryVec(self.inner - rhs.inner)
        return SymmetryVec(rhs.to_vec() - self.to_vec())

    def __mul__(self, rhs: SymmetryVec) -> SymmetryVec:
        if not self.is_scalar() and not rhs.is_scalar():
            raise ValueError("Can't multiply two symmetry directions")
        return SymmetryVec(rhs.inner * self.inner)

    def __truediv__(self, rhs: SymmetryVec) -> SymmetryVec:
        if not self.is_scalar() and not rhs.is_scalar():
            raise ValueError("Can't divide two symmetry directions")
        return SymmetryVec(rhs.inner / self.inner)


SYMMETRY_PARSER: Parser[SymmetryVec, SymmetryVec] = Parser([
    BinaryOrUnaryOp(['-'], sub, False, 5),
    BinaryOp(['+'], operator.add, 5),
    BinaryOp(['*'], operator.mul, 6),
    BinaryOp(['/'], operator.truediv, 6),
], SymmetryVec.parse)


def parse_symmetry(s: str) -> AffineTransform3D:
    axes = s.split(',')
    if not len(axes) == 3:
        raise ValueError(f"Error parsing symmetry expression '{s}': Expected 3 values, got {len(axes)}")

    axes = [SYMMETRY_PARSER.parse(StringIO(ax)).eval(lambda v: v).to_vec() for ax in axes]
    axes.append(numpy.array([0., 0., 0., 1.]))
    return AffineTransform3D(numpy.stack(axes, axis=0))


class CifReader:
    def __init__(self, file: TextIOBase):
        self.line = 0
        self._file: TextIOBase = file
        self._buf: t.Optional[str] = None
        self._after_eol = True
        self._eof = False

    def parse(self) -> t.Iterator[CIF]:
        while True:
            line = self.line
            word = self.peek_word()
            if word is None:
                return
            if word.lower().startswith('data_'):
                self.next_word()
                name = word[len('data_'):]
            elif word.startswith('_'):
                name = None
            else:
                raise ValueError(f"While parsing line {line}: Unexpected token {word}")

            yield self.parse_datablock(name)

    def after_eol(self) -> bool:
        """
        Returns whether the current token (the one that will be returned
        by the next peek() or next()) is after a newline.
        """
        return self._after_eol

    def peek_line(self) -> t.Optional[str]:
        buf = self._try_fill_buf()
        return buf

    def next_line(self) -> t.Optional[str]:
        line = self.peek_line()
        self._buf = None
        return line

    def next_until(self, marker: str) -> t.Optional[str]:
        """
        Collect words until `marker`. Because of the weirdness of CIF,
        `marker` must occur immediately before a whitespace boundary.
        """
        s = ""
        buf = self._try_fill_buf()
        if buf is None:
            return None
        while not (match := re.search(re.escape(marker) + r'(?=\s|$)', buf)):
            s += buf
            buf = self._try_fill_buf(True)
            if buf is None:
                return None
        s += buf[:match.end()]
        self._buf = buf[match.end():]
        if len(self._buf) == 0 or self._buf.isspace():
            self._buf = None
        return s

    def peek_word(self) -> t.Optional[str]:
        while True:
            buf = self._try_fill_buf()
            if buf is None:
                return None
            buf = buf.lstrip()
            if len(buf) == 0 or buf.isspace() or buf.startswith('#'):
                # eat comment or blank line
                self._buf = None
                continue
            break

        #print(f"buf: '{buf}'")
        return buf.split(maxsplit=1)[0]

    def next_word(self) -> t.Optional[str]:
        w = self.peek_word()
        if w is None:
            return None
        assert self._buf is not None
        self._buf = self._buf.lstrip()[len(w)+1:].lstrip()
        if len(self._buf) == 0 or self._buf.isspace():
            # eat whitespace at end of line
            self._buf = None
            self._after_eol = True
        else:
            self._after_eol = False
        return w

    def _try_fill_buf(self, force: bool = False) -> t.Optional[str]:
        if force:
            self._buf = None
        if self._buf is None:
            try:
                self._buf = next(self._file)
                self.line += 1
            except StopIteration:
                pass
        return self._buf
    
    def parse_bare(self) -> t.Union[int, float, str]:
        w = self.next_word()
        if w is None:
            raise ValueError("Unexpected EOF while parsing value.")
        if _INT_RE.fullmatch(w):
            return int(w)  # may raise
        if (m := _FLOAT_RE.fullmatch(w)):
            if m[1] != '.':
                return float(m[1])  # may raise
        return w

    def parse_datablock(self, name: t.Optional[str] = None) -> CIF:
        logging.debug(f"parse datablock '{name}'")
        data: t.Dict[str, t.Union[t.List[Value], Value]] = {}

        while True:
            word = self.peek_word()
            if word is None:
                break
            if word.lower() == 'loop_':
                self.next_word()
                data.update(self.parse_loop())
            elif word.startswith('_'):
                self.next_word()
                data[word[1:]] = self.parse_value()
                logging.debug(f"{word[1:]} = {data[word[1:]]}")
            else:
                break

        return CIF(name, data)

    def eat_saveframe(self):
        line = self.line
        while True:
            w = self.next_word()
            if w is None:
                raise ValueError(f"EOF before end of save frame starting at line {line}")
            if w.lower() == 'save_':
                break

    def parse_loop(self) -> t.Dict[str, t.Any]:
        line = self.line
        tags = []
        while True:
            w = self.peek_word()
            if w is None:
                raise ValueError(f"EOF before loop values at line {line}")
            if w.startswith('_'):
                self.next_word()
                tags.append(w[1:])
            else:
                break

        vals = tuple([] for _ in tags)
        i = 0

        while True:
            w = self.peek_word()
            if w is None or w.startswith('_') or w.endswith('_'):
                break
            vals[i].append(self.parse_value())
            i = (i + 1) % len(tags)

        if i != 0:
            n_vals = sum(map(len, vals))
            raise ValueError(f"While parsing loop at line {line}: "
                            f"Got {n_vals} vals, expected a multiple of {len(tags)}")

        return dict(zip(tags, vals))

    def parse_value(self) -> Value:
        logging.debug(f"parse_value")
        w = self.peek_word()
        assert w is not None
        if w in ('.', '?'):
            self.next_word()
            return None

        if self.after_eol() and w == ';':
            return self.parse_text_field()

        if w[0] in ('"', "'"):
            return self.parse_quoted()

        return self.parse_bare()

    def parse_text_field(self) -> str:
        line = self.line
        l = self.next_line()
        assert l is not None
        s = l.lstrip().removeprefix(';')
        while True:
            l = self.next_line()
            if l is None:
                raise ValueError(f"While parsing text field at line {line}: Unexpected EOF")
            if l.strip() == ';':
                break
            s += l
        return s

    def parse_quoted(self) -> str:
        line = self.line
        w = self.peek_word()
        assert w is not None
        quote = w[0]
        if quote not in ('"', "'"):
            raise ValueError(f"While parsing string at line {line}: Invalid quote char {quote}")

        s = self.next_until(quote)
        if s is None:
            raise ValueError(f"While parsing string {w}... at line {line}: Unexpected EOF")
        return s.lstrip()[1:-1]
