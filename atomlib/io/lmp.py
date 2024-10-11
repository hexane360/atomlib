"""
IO for LAAMPS data files, as described [here](https://docs.lammps.org/read_data.html#format-of-a-data-file).
"""

from __future__ import annotations

from dataclasses import dataclass
from io import TextIOBase
import re
import typing as t

import numpy
import polars

from ..atomcell import HasAtomCell, HasAtoms, Atoms, Cell, AtomCell
from ..elem import get_elem, get_sym
from ..util import open_file, FileOrPath, localtime, checked_left_join, CheckedJoinError
from ..transform import AffineTransform3D, LinearTransform3D
from .util import parse_whitespace_separated


@dataclass
class LMP:
    comment: t.Optional[str]
    headers: t.Dict[str, t.Any]
    sections: t.Tuple[LMPSection, ...]

    def get_cell(self) -> Cell:
        dims = numpy.array([
            self.headers.get(f"{c}lo {c}hi", (-0.5, 0.5))
            for c in "xyz"
        ])
        origin = dims[:, 0]
        tilts = self.headers.get("xy xz yz", (0., 0., 0.))

        ortho = numpy.diag(dims[:, 1] - dims[:, 0])
        (ortho[0, 1], ortho[0, 2], ortho[1, 2]) = tilts

        return Cell.from_ortho(LinearTransform3D(ortho).translate(origin))

    def get_atoms(self, type_map: t.Optional[t.Dict[int, t.Union[str, int]]] = None) -> AtomCell:
        if type_map is not None:
            try:
                type_map_df = polars.DataFrame({
                    'type': polars.Series(type_map.keys(), dtype=polars.Int32),
                    'elem': polars.Series(list(map(get_elem, type_map.values())), dtype=polars.UInt8),
                    'symbol': polars.Series([get_sym(v) if isinstance(v, int) else v for v in type_map.values()], dtype=polars.Utf8),
                })
            except ValueError as e:
                raise ValueError("Invalid type map") from e
        else:
            type_map_df = None

        cell = self.get_cell()

        def _apply_type_labels(df: polars.DataFrame, section_name: str, labels: t.Optional[polars.DataFrame] = None) -> polars.DataFrame:
            if labels is not None:
                #df = df.with_columns(polars.col('type').replace(d, default=polars.col('type').cast(polars.Int32, strict=False), return_dtype=polars.Int32))
                df = df.with_columns(polars.col('type').replace_strict(labels['symbol'], labels['type'], default=polars.col('type').cast(polars.Int32, strict=False), return_dtype=polars.Int32))
                if df['type'].is_null().any():
                    raise ValueError(f"While parsing section {section_name}: Unknown atom label or invalid atom type")
            try:
                return df.with_columns(polars.col('type').cast(polars.Int32))
            except polars.ComputeError:
                raise ValueError(f"While parsing section {section_name}: Invalid atom type(s)")

        atoms: t.Optional[polars.DataFrame] = None
        labels: t.Optional[polars.DataFrame] = None
        masses: t.Optional[polars.DataFrame] = None
        velocities = None

        for section in self.sections:
            start_line = section.start_line + 1

            if section.name == 'Atoms':
                if section.style not in (None, 'atomic'):
                    # TODO support other styles
                    raise ValueError(f"Only 'atomic' atom_style is supported, instead got '{section.style}'")

                atoms = parse_whitespace_separated(section.body, {
                    'i': polars.Int64, 'type': polars.Utf8,
                    'coords': polars.Array(polars.Float64, 3),
                }, start_line=start_line)
                atoms = _apply_type_labels(atoms, 'Atoms', labels)
            elif section.name == 'Atom Type Labels':
                labels = parse_whitespace_separated(section.body, {'type': polars.Int32, 'symbol': polars.Utf8}, start_line=start_line)
            elif section.name == 'Masses':
                masses = parse_whitespace_separated(section.body, {'type': polars.Utf8, 'mass': polars.Float64}, start_line=start_line)
                masses = _apply_type_labels(masses, 'Masses', labels)
            elif section.name == 'Velocities':
                velocities = parse_whitespace_separated(section.body, {
                    'i': polars.Int64, 'velocity': polars.Array(polars.Float64, 3),
                }, start_line=start_line)

        # now all 'type's should be in Int32

        if atoms is None:
            if self.headers['atoms'] > 0:
                raise ValueError("Missing required section 'Atoms'")
            return AtomCell(Atoms.empty(), cell=cell, frame='local')

        # next we need to assign element symbols
        # first, if type_map is specified, use that:
        #if type_map_elem is not None and type_map_sym is not None:
        if type_map_df is not None:
            try:
                atoms = checked_left_join(atoms, type_map_df, on='type')
            except CheckedJoinError as e:
                raise ValueError(f"Missing type_map specification for atom type(s): {', '.join(map(repr, e.missing_keys))}")
        elif labels is not None:
            try:
                labels = labels.with_columns(get_elem(labels['symbol']))
            except ValueError as e:
                raise ValueError("Failed to auto-detect elements from type labels. Please pass 'type_map' explicitly") from e
            try:
                atoms = checked_left_join(atoms, labels, 'type')
            except CheckedJoinError as e:
                raise ValueError(f"Missing labels for atom type(s): {', '.join(map(repr, e.missing_keys))}")
        # otherwise we have no way
        else:
            raise ValueError("Failed to auto-detect elements from type labels. Please pass 'type_map' explicitly")

        if velocities is not None:
            # join velocities
            try:
                # TODO use join_asof here?
                atoms = checked_left_join(atoms, velocities, 'i')
            except CheckedJoinError as e:
                raise ValueError(f"Missing velocities for {len(e.missing_keys)}/{len(atoms)} atoms")

        if masses is not None:
            # join masses
            try:
                atoms = checked_left_join(atoms, masses, 'type')
            except CheckedJoinError as e:
                raise ValueError(f"Missing masses for atom type(s): {', '.join(map(repr, e.missing_keys))}")

        return AtomCell(atoms, cell=cell, frame='local')

    @staticmethod
    def from_atoms(atoms: HasAtoms) -> LMP:
        if isinstance(atoms, HasAtomCell):
            # we're basically converting everything to the ortho frame, but including the affine shift

            # transform affine shift into ortho frame
            origin = atoms.get_transform('ortho', 'local').to_linear().round_near_zero() \
                .transform(atoms.get_cell().affine.translation())

            # get the orthogonalization transform only, without affine
            ortho = atoms.get_transform('ortho', 'cell_box').to_linear().round_near_zero().inner

            # get atoms in ortho frame, and then add the affine shift
            frame = atoms.get_atoms('ortho').transform_atoms(AffineTransform3D.translate(origin)) \
                .round_near_zero().with_type()
        else:
            bbox = atoms.bbox_atoms()
            ortho = numpy.diag(bbox.size)
            origin = bbox.min

            frame = atoms.get_atoms('local').with_type()

        types = frame.unique(subset='type')
        types = types.with_mass().sort('type')

        now = localtime()
        comment = f"# Generated by atomlib on {now.isoformat(' ', 'seconds')}"

        headers = {}
        sections = []

        headers['atoms'] = len(frame)
        headers['atom types'] = len(types)

        for (s, low, diff) in zip(('x', 'y', 'z'), origin, ortho.diagonal()):
            headers[f"{s}lo {s}hi"] = (low, low + diff)

        headers['xy xz yz'] = (ortho[0, 1], ortho[0, 2], ortho[1, 2])

        body = [
            f" {ty:8} {sym:>4}\n"
            for (ty, sym) in types.select('type', 'symbol').rows()
        ]
        sections.append(LMPSection("Atom Type Labels", tuple(body)))

        if 'mass' in types:
            body = [
                f" {ty:8} {mass:14.7f}  # {sym}\n"
                for (ty, sym, mass) in types.select(('type', 'symbol', 'mass')).rows()
            ]
            sections.append(LMPSection("Masses", tuple(body)))

        body = [
            f" {i+1:8} {ty:4} {x:14.7f} {y:14.7f} {z:14.7f}\n"
            for (i, (ty, (x, y, z))) in enumerate(frame.select(('type', 'coords')).rows())
        ]
        sections.append(LMPSection("Atoms", tuple(body), 'atomic'))

        if (velocities := frame.velocities()) is not None:
            body = [
                f" {i+1:8} {v_x:14.7f} {v_y:14.7f} {v_z:14.7f}\n"
                for (i, (v_x, v_y, v_z)) in enumerate(velocities)
            ]
            sections.append(LMPSection("Velocities", tuple(body)))

        return LMP(comment, headers, tuple(sections))

    @staticmethod
    def from_file(file: FileOrPath) -> LMP:
        with open_file(file, 'r') as f:
            return LMPReader(f).parse()

    def write(self, file: FileOrPath):
        with open_file(file, 'w') as f:
            print((self.comment or "") + '\n', file=f)

            # print headers
            for (name, val) in self.headers.items():
                val = _HEADER_FMT.get(name, lambda s: f"{s:8}")(val)
                print(f" {val} {name}", file=f)

            # print sections
            for section in self.sections:
                line = section.name
                if section.style is not None:
                    line += f'  # {section.style}'
                print(f"\n{line}\n", file=f)

                f.writelines(section.body)


@dataclass
class LMPSection:
    name: str
    body: t.Tuple[str, ...]
    style: t.Optional[str] = None
    start_line: int = 0


class LMPReader:
    def __init__(self, f: TextIOBase):
        self.line = 0
        self._file: TextIOBase = f
        self._buf: t.Optional[str] = None

    def _split_comment(self, line: str) -> t.Tuple[str, t.Optional[str]]:
        split = _COMMENT_RE.split(line, maxsplit=1)
        return (split[0], split[1] if len(split) > 1 else None)

    def parse(self) -> LMP:
        # parse comment
        comment = self.next_line(skip_blank=False)
        if comment is None:
            raise ValueError("Unexpected EOF (file is blank)")
        if comment.isspace():
            comment = None
        else:
            comment = comment[:-1]

        headers = self.parse_headers()
        sections = self.parse_sections(headers)

        return LMP(comment, headers, sections)

    def parse_headers(self) -> t.Dict[str, t.Any]:
        headers: t.Dict[str, t.Any] = {}
        while True:
            line = self.peek_line()
            if line is None:
                break
            body = self._split_comment(line)[0]

            if (match := _HEADER_KW_RE.search(body)) is None:
                # probably a body
                break
            self.next_line()

            name = match[0]
            value = body[:match.start(0)].strip()

            try:
                if name in _HEADER_PARSE:
                    value = _HEADER_PARSE[name](value)
            except Exception as e:
                raise ValueError(f"While parsing header '{name}' at line {self.line}: Failed to parse value '{value}") from e

            #print(f"header {name} => {value} (type {type(value)})")
            headers[name] = value

        return headers

    def parse_sections(self, headers: t.Dict[str, t.Any]) -> t.Tuple[LMPSection, ...]:
        first = True

        sections: t.List[LMPSection] = []

        while True:
            start_line = self.line
            line = self.next_line()
            if line is None:
                break
            name, comment = self._split_comment(line)
            name = name.strip()

            try:
                n_lines_header = _SECTION_KWS[name]
            except KeyError:
                if first:
                    raise ValueError(f"While parsing line {self.line}: Unknown header or section keyword '{line}'") from None
                else:
                    raise ValueError(f"While parsing line {self.line}: Unknown section keyword '{line}'") from None

            try:
                if n_lines_header is None:
                    # special case for PairIJ Coeffs:
                    n = int(headers['atom types'])
                    n_lines = (n * (n + 1)) // 2
                else:
                    n_lines = int(headers[n_lines_header])
            except KeyError:
                raise ValueError(f"While parsing body section '{name}' at line {self.line}: "
                                 f"Missing required header '{n_lines_header or 'atom types'}'") from None

            style = comment if name in _SECTION_STYLE_KWS else None
            if style is not None:
                style = style.strip()

            #print(f"section '{name}' @ {self.line}, {n_lines} lines, style {style}")

            lines = self.collect_lines(n_lines)
            if lines is None:
                raise ValueError(f"While parsing body section '{name}' starting at line {self.line}: "
                                 f"Unexpected EOF before {n_lines} lines were read")

            sections.append(LMPSection(
                name, tuple(lines), style, start_line
            ))
            first = False

        return tuple(sections)

    def _try_fill_buf(self, skip_blank: bool = True) -> t.Optional[str]:
        if self._buf is None:
            try:
                # skip blank lines
                while True:
                    self._buf = next(self._file)
                    self.line += 1
                    if not (skip_blank and self._buf.isspace()):
                        break
            except StopIteration:
                pass
        return self._buf

    def peek_line(self, skip_blank: bool = True) -> t.Optional[str]:
        return self._try_fill_buf(skip_blank)

    def next_line(self, skip_blank: bool = True) -> t.Optional[str]:
        line = self._try_fill_buf(skip_blank)
        self._buf = None
        return line

    def collect_lines(self, n: int) -> t.Optional[t.List[str]]:
        assert self._buf is None
        lines = []
        try:
            for _ in range(n):
                while True:
                    line = next(self._file)
                    if not line.isspace():
                        lines.append(line)
                        break
        except StopIteration:
            return None
        self.line += n
        return lines


def write_lmp(atoms: HasAtoms, f: FileOrPath):
    LMP.from_atoms(atoms).write(f)
    return


def _parse_seq(f: t.Callable[[str], t.Any], n: int) -> t.Callable[[str], t.Tuple[t.Any, ...]]:
    def inner(s: str) -> t.Tuple[t.Any, ...]:
        vals = s.strip().split()
        if len(vals) != n:
            raise ValueError(f"Expected {n} values, instead got {len(vals)}")
        return tuple(map(f, vals))

    return inner

_parse_2float = _parse_seq(float, 2)
_parse_3float = _parse_seq(float, 3)

def _fmt_2float(vals: t.Sequence[float]):
    return f"{vals[0]:16.7f} {vals[1]:14.7f}"

def _fmt_3float(vals: t.Sequence[float]):
    return f"{vals[0]:16.7f} {vals[1]:14.7f} {vals[2]:14.7f}"


_HEADER_KWS = {
    "xlo xhi", "ylo yhi", "zlo zhi", "xy xz yz",
    "atoms", "bonds", "angles", "dihedrals", "impropers",
    "atom types", "bond types", "angle types", "dihedral types", "improper types",
    "extra bond per atom", "extra angle per atom", "extra dihedral per atom",
    "extra improper per atom", "extra special per atom",
    "ellipsoids", "lines",
    "triangles", "bodies",
}

_HEADER_PARSE: t.Dict[str, t.Callable[[str], t.Any]] = {
    'atoms': int, 'bonds': int, 'angles': int, 'dihedrals': int, 'impropers': int,
    'ellipsoids': int, 'lines': int, 'triangles': int, 'bodies': int,
    'atom types': int, 'bond types': int, 'angle types': int, 'dihedral types': int, 'improper types': int,
    'extra bond per atom': int, 'extra angle per atom': int, 'extra dihedral per atom': int,
    'extra improper per atom': int, 'extra special per atom': int,
    'xlo xhi': _parse_2float, 'ylo yhi': _parse_2float, 'zlo zhi': _parse_2float,
    'xy xz yz': _parse_3float,
}

_HEADER_FMT: t.Dict[str, t.Callable[[t.Any], str]] = {
    'xlo xhi': _fmt_2float, 'ylo yhi': _fmt_2float, 'zlo zhi': _fmt_2float,
    'xy xz yz': _fmt_3float,
}

_SECTION_KWS: t.Dict[str, t.Optional[str]] = {
    # atom property sections
    "Atoms": "atoms", "Velocities": "atoms", "Masses": "atom types",
    "Ellipsoids": "ellipsoids", "Lines": "lines", "Triangles": "triangles", "Bodies": "bodies",
    # molecular topology sections
    "Bonds": "bonds", "Angles": "angles", "Dihedrals": "dihedrals", "Impropers": "impropers",
    # type label maps
    "Atom Type Labels": "atom types", "Bond Type Labels": "bond types", "Angle Type Labels": "angle types",
    "Dihedral Type Labels": "dihedral types", "Improper Type Labels": "improper types",
    # force field sections
    "Pair Coeffs": "atom types", "PairIJ Coeffs": None, "Bond Coeffs": "bond types", "Angle Coeffs": "angle types",
    "Dihedral Coeffs": "dihedral types", "Improper Coeffs": "improper types",
    # class 2 force field sections
    "BondBond Coeffs": "angle types", "BondAngle Coeffs": "angle types", "MiddleBondTorision Coeffs": "dihedral types",
    "EndBondTorsion Coeffs": "dihedral types", "AngleTorsion Coeffs": "dihedral types", "AngleAngleTorsion Coeffs": "dihedral types",
    "BondBond13 Coeffs": "dihedral types", "AngleAngle Coeffs": "improper types",
}
_SECTION_STYLE_KWS: t.Set[str] = {
    "Atoms", "Pair Coeffs", "PairIJ Coeffs", "Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs",
}

_HEADER_KW_RE = re.compile("{}$".format('|'.join(map(re.escape, _HEADER_KWS))))
_COMMENT_RE = re.compile(r"\s#")