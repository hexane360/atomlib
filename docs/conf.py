# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'structlib'
#copyright = ''
author = 'Colin Gilgenbach'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
]

autodoc_default_options = {
    'member-order': 'bysource',
}

autodoc_type_aliases = {
    'ArrayLike': 'numpy.typing.ArrayLike',
    'NDArray': 'numpy.typing.NDArray',

    'Vec3': 'structlib.types.Vec3',
    'VecLike': 'structlib.types.VecLike',
    'Pts3DLike': 'structlib.types.Pts3DLike',
    'Num': 'structlib.types.Num',
    'ElemLike': 'ElemLike',

    'SchemaDict': 'structlib.atoms.SchemaDict',
    'IntoAtoms': 'structlib.atoms.IntoAtoms',
    'AtomSelection': 'structlib.atoms.AtomSelection',
    'AtomValues': 'structlib.atoms.AtomValues',

    'IntoTransform3D': 'structlib.transform.IntoTransform3D',

    'CellType': 'structlib.make.CellType',
}
autodoc_typehints_format = 'fully-qualified'

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

templates_path = ['templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
#html_static_path = ['static']

nitpicky = True
nitpick_ignore = [
    ('py:class', 'type'),
]