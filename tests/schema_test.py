# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)
from scitiff._resources import SCITIFF_METADATA_CONTAINER_SCHEMA, SCITIFF_SCHEMA
from scitiff._schema import SciTiff, SciTiffCompatibleMetadata


def test_schema_files_up_to_date() -> None:
    assert (
        SciTiffCompatibleMetadata.model_json_schema()
        == SCITIFF_METADATA_CONTAINER_SCHEMA
    )
    assert SciTiff.model_json_schema() == SCITIFF_SCHEMA
