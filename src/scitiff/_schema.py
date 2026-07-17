# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)
"""
Most fields should not be strictly validated to allow loading files with wrong metadata.
"""

import re
import warnings
from enum import Enum, StrEnum
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, Field, model_validator

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

    unit: str | None
    dtype: str


class ScippVariable0D(ScippVariableMetadata):
    """Scipp Variable Metadata with scalar value."""

    dims: tuple[()] = Field(default=())
    shape: tuple[()] = Field(default=())
    values: float | str
    """The value of the scalar variable"""


class ScippVariable1D(ScippVariableMetadata):
    """Scipp Variable Metadata with 1D array values."""

    dims: tuple[str]
    shape: tuple[int]
    values: list[float] | list[str]
    """The 1D values list of the variable."""


class ScippVariable2D(ScippVariableMetadata):
    """Scipp Variable Metadata with 2D array values.

    For 2D array, only numbers(float/int) are allowed."""

    dims: tuple[str, str]
    shape: tuple[int, int]
    values: list[list[float]] = Field(strict=True)
    """The 2D values list of the variable."""


ScippVariable = ScippVariable0D | ScippVariable1D | ScippVariable2D


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


class SourceType(Enum):
    """Enum for probe types."""

    NEUTRON = "neutron"
    X_RAY = "x-ray"
    ELECTRON = "electron"


class NeutronSourceType(Enum):
    CONTINUOUS = "continuous"
    LONG_PULSE = "long-pulse"
    SHORT_PULSE = "short-pulse"


class NeutronMetadata(BaseModel):
    neutron_type: NeutronSourceType
    wavelength_range: tuple[ScippVariable0D, ScippVariable0D]


class XRayMetadata(BaseModel): ...


class ElectronMetadata(BaseModel): ...


SourceMetaType = NeutronMetadata | XRayMetadata | ElectronMetadata | None


def complain_if_not_email(value: str) -> str:
    email_re = re.compile(r"(^[\w\-\.]+)@([\w-]+\.+[\w-]{2,})$")
    if email_re.match(value) is None:
        warnings.warn(
            category=UserWarning,
            message=f"Given email {value} does not match a valid email format.",
            stacklevel=1,
        )

    return value


def complain_if_not_orcid(value: str) -> str:
    orcid_re = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[0-9X]{1}$")
    if orcid_re.match(value) is None:
        warnings.warn(
            category=UserWarning,
            message=f"Given orcid {value} does not match a valid orcid format.",
            stacklevel=1,
        )
    return value


class Person(BaseModel):
    name: str = Field(description="Name of the person.")
    affiliation: str | None = Field(
        default=None,
        description="Affiliation of the person at the time of the data acquisition.",
    )
    email: Annotated[str, AfterValidator(complain_if_not_email)] | None = Field(
        default=None, description="Email address of the person."
    )
    orcid: Annotated[str, AfterValidator(complain_if_not_orcid)] | None = Field(
        default=None,
        description="ORCID of the person. "
        "See https://orcid.org/ for more details about ORCID.",
    )


class ExperimentIdentifierType(StrEnum):
    PROPOSAL_ID = "PROPOSAL_ID"
    """Proposal ID under which the data acquisition occurred."""
    RUN_NUMBER = "RUN_NUMBER"
    """Unique number of an individual data acquisition round. e.g. one file written"""
    CUSTOM = "CUSTOM"
    """Custom identifier defined by the team."""


class ExperimentIdentifier(BaseModel):
    type: ExperimentIdentifierType = Field(
        description="Type of experiment identifier. "
        "e.g. proposal_id, run_number or custom. "
        "Custom identifier should have helpful description."
    )
    value: str
    description: str = ""


class DAQMetadata(BaseModel):
    """DAQ information related to the image.

    For example, if a raw image is directly extracted by one acquisition, it should
    """

    facility: str | list[str] = Field(default_factory=list, description="Facility name")
    instrument: str | list[str] = Field(
        default_factory=list, description="Instrument name"
    )
    detector_type: str | list[str] = Field(
        default_factory=list, description="Detector type"
    )
    source_type: str | SourceType | None = Field(
        default=None,
        description="Type of source(probe). i.e. neutron, x-ray, etc.",
    )
    source: SourceMetaType = Field(default=None, description="Source metadata.")
    simulated: bool | None = Field(
        default=None, description="Flag indicating if the data is simulated."
    )
    principal_investigators: list[Person] = Field(
        default_factory=list,
        description="Principal Investigator(s) of the data acquisition.",
    )
    team: list[Person] = Field(
        default_factory=list,
        description="Anyone who participated the data acquisition.",
    )
    local_contacts: list[Person] = Field(
        default_factory=list,
        description="Local contact(s) of the data acquisition.",
    )
    experiment_identifiers: list[ExperimentIdentifier] = Field(
        default_factory=list,
        description="Related experiment identifiers. e.g. Proposal IDs or run numbers.",
    )


class ImageResultType(StrEnum):
    NORMALIZED = "NORMALIZED"
    SAMPLE = "SAMPLE"
    OPENBEAM = "OPENBEAM"
    DARKCURRENT = "DARKCURRENT"


class ProcessIdentifier(BaseModel):
    type: str = ""
    value: str
    description: str = ""


class ImageProcessMetadata(BaseModel):
    """Metadata about how the image was derived
    and what the image represents as a result.

    """

    result_type: ImageResultType | str | None = Field(
        default=None, description="The type of image as a result of the image process. "
    )
    processing_steps: list[str] = Field(default_factory=list)
    parameters: dict[str, str | float] = Field(default_factory=dict)
    process_identifiers: list[ProcessIdentifier] = Field(
        default_factory=list,
        description="Unique ID of the process. e.g. job id in catalogue, "
        "package version of the processing workflow.",
    )
    coordinate_descriptions: dict[str, str] = Field(
        default_factory=dict,
        description="Details of what each coordinate of the image means."
        "Names are often not descriptive enough.",
    )


class SciTiffMetadata(BaseModel):
    """SCITIFF Metadata."""

    image: ImageDataArrayMetadata = Field(
        description="Physical Properties of the Image such as coordinates."
    )
    daq: DAQMetadata = Field(default_factory=DAQMetadata)
    process: ImageProcessMetadata = Field(default_factory=ImageProcessMetadata)
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata that is not part of the schema.",
    )
    schema_version: str = "{VERSION_PLACEHOLDER}"

    @model_validator(mode='after')
    def check_coordinate_names(self) -> "SciTiffMetadata":
        # check if the process metadata coordinate
        # name matches the ones with image metadata.
        process_coordinates = set(self.process.coordinate_descriptions.keys())
        image_coordinates = set(self.image.coords.keys())
        if any(unmatching := process_coordinates - image_coordinates):
            names = ",".join(unmatching)
            warnings.warn(
                "Coordinate names in `process.coordinate_descriptions` "
                f"not found in the image metadata: [{names}]. "
                f"Coordinates should be one of: [{','.join(image_coordinates)}]",
                stacklevel=1,
            )

        return self


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
