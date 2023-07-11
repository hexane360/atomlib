
from os import environ
import requests
import logging
import typing as t

import numpy

from ..atomcell import AtomCell
from ..atoms import Atoms
from ..elem import get_elem
from ..transform import LinearTransform3D


def get_api_key(key: t.Optional[str] = None) -> str:
    try:
        return environ['MP_API_KEY']
    except KeyError:
        raise RuntimeError("No materials project API key specified. "
                           "Either pass `api_key` or set `MP_API_KEY` in your environment.") from None


def get_api_endpoint() -> str:
    return environ.get("MP_API_ENDPOINT") or "https://api.materialsproject.org/"


def resolve_id(id: t.Union[str, int]) -> str:
    if isinstance(id, int):
        return str(id)
    if id.lower().startswith('mp-'):
        return id[3:]
    return id


def load_materials_project(id: t.Union[str, int], *, api_key: t.Optional[str] = None,
                           api_endpoint: t.Optional[str] = None) -> AtomCell:
    id = resolve_id(id)
    api_key = api_key or get_api_key()
    if len(api_key) != 32:
        raise RuntimeError("Materials project API key must be a 32-character alphanumeric string. "
                           f"Instead got '{api_key}' of length '{len(api_key)}'.")
    api_endpoint = (api_endpoint or get_api_endpoint()).rstrip('/')

    logging.info(f"Fetching structure mp-{id} from materials project...")
    response = requests.get(
        api_endpoint + f'/materials/mp-{id}/',
        headers={'X-Api-Key': api_key},
        params={'_fields': 'structure'}
    )
    data = response.json()['data'][0]
    structure = data['structure']

    ortho = LinearTransform3D(numpy.array(structure['lattice']['matrix']).T)
    sites = structure['sites']

    rows = []
    for site in sites:
        (x, y, z) = site['xyz']
        for species in site['species']:
            rows.append({
                'symbol': site['label'],
                'elem': get_elem(species['element']),
                'x': x, 'y': y, 'z': z,
                'frac': species['occu'],
                **site['properties'],
            })

    frame = Atoms(rows, orient='row')
    return AtomCell.from_ortho(frame, ortho)