# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'atomlib'
#copyright = ''
author = 'Colin Gilgenbach'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

nitpicky = True
nitpick_ignore = [
    ('py:class', 'type'),
    ('py:class', 'T'),
    ('py:class', 'U'),
    ('py:class', 'P'),
    ('py:class', 'U_co'),
]

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'h5py': ('https://docs.h5py.org/en/latest/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pyarrow': ('https://arrow.apache.org/docs/', None),
    'polars': ('https://pola-rs.github.io/polars/py-polars/html/', None),
    'h5py': ('https://docs.h5py.org/en/latest/', None),
}

autodoc_default_options = {
    'member-order': 'bysource',
}

autodoc_type_aliases = {
    'ArrayLike': 'numpy.typing.ArrayLike',
    'NDArray': 'numpy.typing.NDArray',

    'Vec3': 'atomlib.types.Vec3',
    'VecLike': 'atomlib.types.VecLike',
    'Pts3DLike': 'atomlib.types.Pts3DLike',
    'Num': 'atomlib.types.Num',
    'ElemLike': 'atomlib.types.ElemLike',

    'SchemaDict': 'atomlib.atoms.SchemaDict',
    'IntoAtoms': 'atomlib.atoms.IntoAtoms',
    'AtomSelection': 'atomlib.atoms.AtomSelection',
    'AtomValues': 'atomlib.atoms.AtomValues',

    'CoordinateFrame': 'atomlib.cell.CoordinateFrame',
    'IntoTransform3D': 'atomlib.transform.IntoTransform3D',
    'CellType': 'atomlib.make.CellType',
}

autodoc_typehints_format = 'short'

templates_path = ['templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
#html_static_path = ['static']


def setup(app):
    ...

