import json
import pathlib


def _load_schema() -> dict:
    with open(pathlib.Path(__file__).parent / 'schema.json') as f:
        return json.load(f)


SCITIFF_METADATA_SCHEMA = _load_schema()
