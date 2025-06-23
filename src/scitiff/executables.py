# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)
from pydantic import BaseModel

from . import __version__
from ._json_helpers import beautify_json
from ._schema import (
    SCITIFF_IMAGE_STACK_DIMENSIONS,
    DAQMetadata,
    ImageDataArrayMetadata,
    ImageVariableMetadata,
    ScippVariable,
    SciTiffMetadata,
    SciTiffMetadataContainer,
)


def dump_beautified_json(model: BaseModel) -> str:
    json_model = model.model_dump_json(
        indent=2, exclude_none=True, exclude_defaults=True
    )

    return beautify_json(json_model)


def _build_dummy_metadata() -> SciTiffMetadataContainer:
    image_meta = ImageDataArrayMetadata(
        data=ImageVariableMetadata(
            dims=SCITIFF_IMAGE_STACK_DIMENSIONS,
            shape=(1, 1, 1, 1, 1),
            dtype="float32",
            unit="counts",
        ),
        coords={
            't': ScippVariable(
                dims=('t',), shape=(1,), dtype='int', unit='s', values=[0]
            ),
            'z': ScippVariable(
                dims=('z',), shape=(1,), dtype='int', unit='m', values=[0]
            ),
            'y': ScippVariable(
                dims=('y',), shape=(1,), dtype='int', unit='m', values=[0]
            ),
            'x': ScippVariable(
                dims=('x',), shape=(1,), dtype='int', unit='m', values=[0]
            ),
        },
    )
    return SciTiffMetadataContainer(
        scitiffmeta=SciTiffMetadata(
            image=image_meta,
            daq=DAQMetadata(instrument='ess'),
            schema_version=__version__,
        )
    )


def dump_metadata_example():
    """
    Dump metadata example into a json file.
    """
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(
        description="Dump metadata example into a json file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scitiff_metadata_example.json",
        help="Output file name.",
    )
    args = parser.parse_args()
    file_path = pathlib.Path(args.output)

    metadata = _build_dummy_metadata()
    file_path.write_text(dump_beautified_json(metadata))
