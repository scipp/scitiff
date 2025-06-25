# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)
import pathlib

import tifffile as tf
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
            daq=DAQMetadata(facility='ess', instrument='odin'),
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


def load_metadata(file_path: pathlib.Path) -> dict | None:
    import json

    with tf.TiffFile(file_path) as tif:
        if tif.imagej_metadata is None:
            return None
        else:
            return {
                key: json.loads(value)
                if isinstance(value, str) and value.startswith('{')
                else value
                for key, value in tif.imagej_metadata.items()
            }


class _DotDotDot:
    """
    Placeholder for shortened values in metadata.
    """

    def __repr__(self):
        return "..."


def shorten_values(meta: dict) -> dict:
    """
    Shorten values in the metadata dictionary.
    """

    return {
        k: [v[0], _DotDotDot(), v[-1]]  # Show first and last value
        if (isinstance(v, list) and len(v) > 2 and k == 'values')
        else (shorten_values(v) if isinstance(v, dict) else v)
        for k, v in meta.items()
    }


def show_metadata(
    file_path: pathlib.Path | str,
    *,
    max_depth: int | None = 4,
    show_all: bool = False,
):
    """
    Show all (ImageJ) metadata of a tiff file in a jupyter notebook.

    Parameters
    ----------
    file_path:
        Path to the tiff file to read metadata from.
    max_depth:
        Maximum depth of nested metadata to display.
        Set to `None` to show all metadata.
    show_all:
        If `True`, show all ImageJ metadata, not just SCITIFF metadata.
        If `False`, only show SCITIFF metadata.

    """
    try:
        from rich.pretty import Pretty
    except ImportError as e:
        raise ImportError("Please install python package 'rich'.") from e

    if (meta := load_metadata(pathlib.Path(file_path))) is None:
        return Pretty(f"{file_path} does not contain metadata.")
    elif not show_all:
        scitiff_meta_keys = SciTiffMetadataContainer.model_fields.keys()
        meta = {k: v for k, v in meta.items() if k in scitiff_meta_keys}
    else:
        ...

    return Pretty(shorten_values(meta), max_depth=max_depth)


def print_metadata():
    """
    Show all (ImageJ) metadata of a tiff file in a console.
    """
    import argparse
    import pathlib

    try:
        from rich.pretty import pprint
    except ImportError as e:
        raise ImportError(
            "Please install python package 'rich' to use this command."
        ) from e

    parser = argparse.ArgumentParser(
        description="Quickly show metadata of a tiff file."
    )
    parser.add_argument(
        type=str,
        default="scitiff_metadata_example.json",
        dest="file_name",
        help="Output file name.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum depth of nested metadata to display.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all ImageJ metadata, not just SCITIFF metadata.",
        default=False,
    )

    args = parser.parse_args()
    file_path = pathlib.Path(args.file_name)

    meta = load_metadata(file_path)
    if meta is None:
        pprint(f"{file_path} does not contain metadata.")
    else:
        if not args.show_all:
            scitiff_meta_keys = SciTiffMetadataContainer.model_fields.keys()
            meta = {k: v for k, v in meta.items() if k in scitiff_meta_keys}

        pprint(shorten_values(meta), max_depth=args.max_depth)
