# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)
import json
import pathlib


def _load_schema(schema_name: str) -> dict:
    with open(pathlib.Path(__file__).parent / schema_name) as f:
        return json.load(f)


SCITIFF_METADATA_CONTAINER_SCHEMA = _load_schema('metadata-schema.json')
