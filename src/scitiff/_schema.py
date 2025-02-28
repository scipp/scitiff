# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)

from pydantic import BaseModel, Field


class ScippVariableMetadata(BaseModel):
    """Scipp Variable Metadata without the values."""

    dims: list[str]
    shape: list[int]
    unit: str
    dtype: str
    variance: list[float] | None = None


class ScippVariable(ScippVariableMetadata):
    """Scipp Variable Metadata with the values."""

    values: list[float]


class ScippDataArrayMetadata(BaseModel):
    """Scipp DataArray Metadata without values(image)."""

    data: ScippVariableMetadata
    masks: dict[str, ScippVariable] = Field(default_factory=dict)
    coords: dict[str, ScippVariable] = Field(default_factory=dict)
    name: str | None = None


class ScippDataArray(ScippDataArrayMetadata):
    """Scipp DataArray Metadata with values(image)."""

    data: ScippVariable


class SciTiffMetadata(BaseModel):
    """SCITIFF Metadata."""

    image: ScippDataArrayMetadata
    schema_version: str = "{VERSION_PLACEHOLDER}"


class SciTiffCompatibleMetadata(BaseModel):
    """SCITIFF Compatible Metadata."""

    scitiffmeta: SciTiffMetadata


class SciTiff(BaseModel):
    """SCITIFF object."""

    metadata: SciTiffMetadata
    data: list
    """The image data in the order of the dimensions specified in the metadata."""


def dump_schemas():
    """Dump all schemas to JSON files."""
    import json
    import pathlib

    # Dump metadata schema
    scitiff_metadata_schema = SciTiffCompatibleMetadata.model_json_schema()
    metadata_json_path = (
        pathlib.Path(__file__).parent / "_resources/metadata-schema.json"
    )
    with open(metadata_json_path, "w") as f:
        json.dump(scitiff_metadata_schema, f, indent=2)

    # Dump scitiff schema
    scitiff_schema_path = (
        pathlib.Path(__file__).parent / "_resources/scitiff-schema.json"
    )
    scitiff_schema = SciTiff.model_json_schema()
    with open(scitiff_schema_path, "w") as f:
        json.dump(scitiff_schema, f, indent=2)
