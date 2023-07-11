
import re
import typing as t

from importlib_resources import files
import polars
import numpy

from .types import ElemLike

ELEMENTS = {
    'h': 1, 'he': 2, 'li': 3, 'be': 4, 'b': 5, 'c': 6, 'n': 7, 'o': 8,
    'f': 9, 'ne': 10, 'na': 11, 'mg': 12, 'al': 13, 'si': 14, 'p': 15, 's': 16,
    'cl': 17, 'ar': 18, 'k': 19, 'ca': 20, 'sc': 21, 'ti': 22, 'v': 23, 'cr': 24,
    'mn': 25, 'fe': 26, 'co': 27, 'ni': 28, 'cu': 29, 'zn': 30, 'ga': 31, 'ge': 32,
    'as': 33, 'se': 34, 'br': 35, 'kr': 36, 'rb': 37, 'sr': 38, 'y': 39, 'zr': 40,
    'nb': 41, 'mo': 42, 'tc': 43, 'ru': 44, 'rh': 45, 'pd': 46, 'ag': 47, 'cd': 48,
    'in': 49, 'sn': 50, 'sb': 51, 'te': 52, 'i': 53, 'xe': 54, 'cs': 55, 'ba': 56,
    'la': 57, 'ce': 58, 'pr': 59, 'nd': 60, 'pm': 61, 'sm': 62, 'eu': 63, 'gd': 64,
    'tb': 65, 'dy': 66, 'ho': 67, 'er': 68, 'tm': 69, 'yb': 70, 'lu': 71, 'hf': 72,
    'ta': 73, 'w': 74, 're': 75, 'os': 76, 'ir': 77, 'pt': 78, 'au': 79, 'hg': 80,
    'tl': 81, 'pb': 82, 'bi': 83, 'po': 84, 'at': 85, 'rn': 86, 'fr': 87, 'ra': 88,
    'ac': 89, 'th': 90, 'pa': 91, 'u': 92, 'np': 93, 'pu': 94, 'am': 95, 'cm': 96,
    'bk': 97, 'cf': 98, 'es': 99, 'fm': 100, 'md': 101, 'no': 102, 'lr': 103, 'rf': 104,
    'db': 105, 'sg': 106, 'bh': 107, 'hs': 108, 'mt': 109, 'ds': 110, 'rg': 111, 'cn': 112,
    'uut': 113, 'fl': 114, 'uup': 115, 'lv': 116, 'uus': 117, 'uuo': 118,
}

ELEMENT_SYMBOLS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S',
    'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
    'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
    'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
    'Uut', 'Fl', 'Uup', 'Lv', 'Uus', 'Uuo',
]
assert len(ELEMENTS) == len(ELEMENT_SYMBOLS)

DATA_PATH = files('atomlib.data')
_ELEMENT_MASSES: t.Optional[numpy.ndarray] = None
_ION_RADII: t.Optional[t.Dict[str, float]] = None
_ELEMENT_RADII: t.Optional[numpy.ndarray] = None


def _get_sym(elem: int) -> str:
    try:
        return ELEMENT_SYMBOLS[elem-1]
    except IndexError:
        raise ValueError(f"Invalid atomic number {elem}") from None


_SYM_RE = r'[a-zA-Z]{1,3}'


@t.overload
def get_elem(sym: ElemLike) -> int:
    ...

@t.overload
def get_elem(sym: polars.Series) -> polars.Series:
    ...

def get_elem(sym: t.Union[int, str, polars.Series]):
    if isinstance(sym, int):
        if not 0 < sym < len(ELEMENTS):
            raise ValueError(f"Invalid atomic number {sym}")
        return sym

    if isinstance(sym, polars.Series):
        return sym.str.extract(_SYM_RE, 0) \
            .str.to_lowercase() \
            .apply(lambda e: ELEMENTS[e], return_dtype=polars.UInt8) \
            .alias('elem')

    sym_s = re.search(_SYM_RE, str(sym))
    try:
        return ELEMENTS[sym_s[0].lower()]  # type: ignore
    except (KeyError, IndexError):
        raise ValueError(f"Invalid element symbol '{sym}'")


def get_elems(sym: str) -> t.List[int]:
    if len(sym) > 0:
        sym = sym[0].upper() + sym[1:]
    segments = [
        match[0] for match in re.finditer(r'[A-Z][a-z]?', str(sym))
    ]
    if len(segments) == 0:
        raise ValueError(f"Invalid compound '{sym}'")

    elems = [ELEMENTS.get(seg.lower()) for seg in segments]
    for (seg, v) in zip(segments, elems):
        if v is None:
            raise ValueError(f"Unknown element '{seg}' in '{sym}'. Compounds are case-sensitive.")
    return t.cast(t.List[int], elems)


@t.overload
def get_sym(elem: int) -> str:
    ...

@t.overload
def get_sym(elem: polars.Series) -> polars.Series:
    ...

def get_sym(elem: t.Union[int, polars.Series]):
    if isinstance(elem, polars.Series):
        return elem.apply(_get_sym, return_dtype=polars.Utf8) \
                   .alias('symbol')

    return ELEMENT_SYMBOLS[elem-1]


@t.overload
def get_mass(elem: int) -> float:
    ...

@t.overload
def get_mass(elem: polars.Series) -> polars.Series:
    ...

@t.overload
def get_mass(elem: t.Union[numpy.ndarray, t.Sequence[int]]) -> numpy.ndarray:
    ...

def get_mass(elem: t.Union[int, t.Sequence[int], numpy.ndarray, polars.Series]):
    """
    Get the standard atomic mass for the given element.

    Follows the `2021 table of the IUPAC Commission on Isotopic Abundances and Atomic Weights <https://doi.org/10.1515/pac-2019-0603>`_.
    """
    global _ELEMENT_MASSES

    if _ELEMENT_MASSES is None:
        with (DATA_PATH / 'masses.npy').open('rb') as f:
            _ELEMENT_MASSES = numpy.load(f, allow_pickle=False)

    if isinstance(elem, polars.Series):
        return polars.Series(values=_ELEMENT_MASSES)[elem-1]

    if isinstance(elem, (int, numpy.ndarray)):
        return _ELEMENT_MASSES[elem-1]  # type: ignore
    return _ELEMENT_MASSES[[e-1 for e in elem]]  # type: ignore


def get_ionic_radius(elem: int, charge: int) -> float:
    """
    Get crystal ionic radius in angstroms for ``elem`` in charge state ``charge``.

    Follows `R.D. Shannon, Acta Cryst. A32 (1976) <https://doi.org/10.1107/S0567739476001551>`.
    """
    global _ION_RADII

    import json

    if _ION_RADII is None:
        with (DATA_PATH / 'ion_radii.json').open('r') as f:
            _ION_RADII = json.load(f)
        assert _ION_RADII is not None

    s = f"{get_sym(elem)}{charge:+d}"

    try:
        return _ION_RADII[s]
    except KeyError:
        raise ValueError(f"Unknown radius for ion '{s}'") from None


@t.overload
def get_radius(elem: int) -> float:
    ...

@t.overload
def get_radius(elem: polars.Series) -> polars.Series:
    ...

@t.overload
def get_radius(elem: t.Union[numpy.ndarray, t.Sequence[int]]) -> numpy.ndarray:
    ...

def get_radius(elem: t.Union[int, t.Sequence[int], numpy.ndarray, polars.Series]):
    """
    Get the neutral atomic radius for the given element(s), in angstroms.

    Follows the calculated values in `E. Clementi et. al, J. Chem. Phys. 47 (1967) <https://doi.org/10.1063/1.1712084>`_.
    """
    global _ELEMENT_RADII

    if _ELEMENT_RADII is None:
        with (DATA_PATH / 'radii.npy').open('rb') as f:
            _ELEMENT_RADII = numpy.load(f, allow_pickle=False)

    if isinstance(elem, polars.Series):
        return polars.Series(values=_ELEMENT_RADII)[elem-1]

    if isinstance(elem, (int, numpy.ndarray)):
        return _ELEMENT_RADII[elem-1]  # type: ignore
    return _ELEMENT_RADII[[e-1 for e in elem]]  # type: ignore
