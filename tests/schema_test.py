# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)
from scitiff._resources import SCITIFF_METADATA_CONTAINER_SCHEMA, SCITIFF_SCHEMA
from scitiff._schema import SciTiff, SciTiffMetadataContainer


def test_metadata_container_schema_file_up_to_date() -> None:
    try:
        assert (
            SciTiffMetadataContainer.model_json_schema()
            == SCITIFF_METADATA_CONTAINER_SCHEMA
        )
    except AssertionError as e:
        raise AssertionError(
            "The metadata container schema file is not up to date. "
            "Please update the schema file. "
            "You can do this by running `scitiff-dev-dump-schemas`."
        ) from e


def test_image_schema_file_up_to_date() -> None:
    try:
        assert SciTiff.model_json_schema() == SCITIFF_SCHEMA
    except AssertionError as e:
        raise AssertionError(
            "The metadata container schema file is not up to date. "
            "Please update the schema file. "
            "You can do this by running `scitiff-dev-dump-schemas`."
        ) from e
