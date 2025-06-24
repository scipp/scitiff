# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)

from typing import Any, Literal

from pydantic import BaseModel, Field

from ._json_helpers import beautify_json

TIME_DIMENSION_AND_COORDINATE_NAME = "t"
"""The name of the time dimension and coordinate."""
ZAXIS_DIMENSION_AND_COORDINATE_NAME = "z"
"""The name of the z-axis dimension and coordinate."""
CHANNEL_DIMENSION_AND_COORDINATE_NAME = "c"
"""The name of the channel dimension and coordinate."""
YAXIS_DIMENSION_AND_COORDINATE_NAME = "y"
"""The name of the y-axis dimension and coordinate."""
XAXIS_DIMENSION_AND_COORDINATE_NAME = "x"
"""The name of the x-axis dimension and coordinate."""

SCITIFF_IMAGE_STACK_DIMENSIONS = (
    TIME_DIMENSION_AND_COORDINATE_NAME,  # t
    ZAXIS_DIMENSION_AND_COORDINATE_NAME,  # z
    CHANNEL_DIMENSION_AND_COORDINATE_NAME,  # c
    YAXIS_DIMENSION_AND_COORDINATE_NAME,  # y
    XAXIS_DIMENSION_AND_COORDINATE_NAME,  # x
)
"""The order of the dimensions in the image stack.

The order is from the outermost dimension to the innermost dimension.
i.e. image[0] is image stack of the first channel.
i.e.2. image[1][0] is the first frame(t) of the second channel.
It is ImageJ Hyperstack default dimension order.

"""


class ScippVariableMetadata(BaseModel):
    """Scipp Variable Metadata without the values."""

    dims: tuple[str, ...]
    shape: tuple[int, ...]
    unit: str | None
    dtype: str


class ScippVariable(ScippVariableMetadata):
    """Scipp Variable Metadata with the values.

    Only 1D variable is allowed for metadata.
    """

    values: float | str | list[float] | list[str]
    """The values of the variable."""


class ImageVariableMetadata(ScippVariableMetadata):
    """Image Metadata."""

    dims: tuple[Literal["t"], Literal["z"], Literal["c"], Literal["y"], Literal["x"]]
    """Scitiff image stack has the fixed number and order of dimensions."""
    shape: tuple[int, int, int, int, int]
    """The shape of the image data."""


class ScippDataArrayMetadata(BaseModel):
    """Scipp DataArray Metadata without values(image)."""

    data: ScippVariableMetadata
    masks: dict[str, ScippVariable] = Field(default_factory=dict)
    """Only 1-dimensional masks are supported for metadata."""
    coords: dict[str, ScippVariable] = Field(default_factory=dict)
    """Only 1-dimensional coordinates are supported for metadata."""
    name: str | None = None


class ImageDataArrayMetadata(ScippDataArrayMetadata):
    """Image DataArray Metadata without values(image)."""

    data: ImageVariableMetadata


class ScippDataArray(ScippDataArrayMetadata):
    """Scipp DataArray Metadata with values(image)."""

    data: ScippVariable


class DAQMetadata(BaseModel):
    facility: str = Field(default="Unknown", description="Facility name")
    instrument: str = Field(default="Unknown", description="Instrument name")
    detector_type: str = Field(default="Unknown", description="Detector type")


class SciTiffMetadata(BaseModel):
    """SCITIFF Metadata."""

    image: ImageDataArrayMetadata
    daq: DAQMetadata = Field(default_factory=DAQMetadata)
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata that is not part of the schema.",
    )
    schema_version: str = "{VERSION_PLACEHOLDER}"


class SciTiffMetadataContainer(BaseModel, extra="allow"):
    """SCITIFF Compatible Metadata."""

    scitiffmeta: SciTiffMetadata


def dump_schemas():
    """Dump all schemas to JSON files."""
    import argparse
    import json
    import pathlib

    parser = argparse.ArgumentParser(
        description="Dump metadata example into a json file."
    )

    # Dump metadata schema
    default_metadata_json_path = (
        pathlib.Path(__file__).parent / "_resources/metadata-schema.json"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=default_metadata_json_path.as_posix(),
        help="Output file name.",
    )
    args = parser.parse_args()
    output_path = pathlib.Path(args.output)
    output_path.write_text(
        beautify_json(
            json.dumps(SciTiffMetadataContainer.model_json_schema(), indent=2)
        )
    )
