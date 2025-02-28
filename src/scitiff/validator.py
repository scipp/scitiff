# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)
from jsonschema import validate

from ._resources import SCITIFF_METADATA_CONTAINER_SCHEMA


def validate_scitiff_metadata_container(instance: dict) -> None:
    validate(instance, SCITIFF_METADATA_CONTAINER_SCHEMA)
