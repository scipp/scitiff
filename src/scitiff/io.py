# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)
import json
import pathlib

import scipp as sc
import tifffile as tf
from scipp.compat.dict import from_dict

from ._schema import (
    ImageDataArrayMetadata,
    ImageVariableMetadata,
    ScippVariable,
    SciTiffMetadata,
    SciTiffMetadataContainer,
)


def _ensure_dimension_order(sizes: dict) -> dict:
    # Order of the dimensions is according to the HyperStacks tiff format.
    order = ['c', 't', 'z', 'y', 'x']
    return {key: sizes[key] for key in order if key in sizes}


def _scipp_variable_to_model(var: sc.Variable) -> ScippVariable:
    return ScippVariable(
        dims=tuple(var.dims),
        dtype=str(var.dtype),
        shape=tuple(var.shape),
        unit=str(var.unit),
        values=var.values.tolist(),
    )


def _extract_metadata_from_dataarray(da: sc.DataArray) -> ImageDataArrayMetadata:
    default_sizes = {'x': 1, 'y': 1, 'z': 1, 't': 1, 'c': 1}
    final_sizes = _ensure_dimension_order({**default_sizes, **da.sizes})
    return ImageDataArrayMetadata(
        data=ImageVariableMetadata(
            dims=tuple(final_sizes.keys()),
            shape=tuple(final_sizes.values()),
            unit=str(da.unit),
            dtype=str(da.dtype),
        ),
        masks={key: _scipp_variable_to_model(mask) for key, mask in da.masks.items()},
        coords={
            key: _scipp_variable_to_model(coord) for key, coord in da.coords.items()
        },
    )


def extract_metadata(dg: sc.DataGroup | sc.DataArray) -> SciTiffMetadataContainer:
    if isinstance(dg, sc.DataArray):
        _metadata = _extract_metadata_from_dataarray(dg)
    else:
        raise NotImplementedError(
            "Extracting metadata from DataGroup to SCITIFF is not yet implemented."
        )

    return SciTiffMetadataContainer(scitiffmeta=SciTiffMetadata(image=_metadata))


def _export_data_array(da: sc.DataArray, file_path: str | pathlib.Path) -> None:
    metadata = extract_metadata(da)
    # Make sure the data is consistent with the metadata
    # It is because ``z`` dimension and ``c`` dimension are often not present
    # but it is require by the HyperStacks and scitiffmeta schema.
    # Also, HyperStacks require specific order of dimensions.
    dims = metadata.scitiffmeta.image.data.dims
    shape = metadata.scitiffmeta.image.data.shape
    sizes: dict[str, int] = dict(zip(dims, shape, strict=True))
    final_image = sc.broadcast(da, sizes=sizes)
    tf.imwrite(
        file_path,
        final_image.values,
        imagej=True,
        metadata={
            key: json.dumps(value)
            for key, value in metadata.model_dump(mode='json').items()
            # Tiff metadata will automatically be translated to json-like-string
            # so we need to convert the metadata to string in advance
            # to catch the error early and make sure it can be loaded back.
        },
        dtype=str(final_image.dtype),
    )


def _export_data_group(dg: sc.DataGroup, file_path: str | pathlib.Path) -> None:
    raise NotImplementedError('Exporting DataGroup to SCITIFF is not yet implemented.')


def export_scitiff(
    dg: sc.DataGroup | sc.DataArray, file_path: str | pathlib.Path
) -> None:
    if isinstance(dg, sc.DataArray):
        _export_data_array(dg, file_path)
    else:
        _export_data_group(dg, file_path)


def load_scitiff(file_path: str | pathlib.Path) -> sc.DataArray | sc.DataGroup:
    with tf.TiffFile(file_path) as tif:
        metadata = json.loads(tif.imagej_metadata['scitiffmeta'])

    img = tf.imread(file_path, squeeze=True)
    metadata = SciTiffMetadata(**metadata)
    # # imread squeezes the dimensions, so we need to keep the original sizes
    # # if not squeezed, the sizes have 1 extra dimension that I do not understand...
    metadata_dict = metadata.model_dump(mode='json', exclude_none=True)
    image_as_dict = metadata_dict['image']
    sizes = dict(zip(metadata.image.data.dims, metadata.image.data.shape, strict=True))
    sizes = {key: value for key, value in sizes.items() if value > 1}
    image_as_dict['data']['values'] = img
    image_as_dict['data']['dims'] = tuple(sizes.keys())
    image_as_dict['data']['shape'] = tuple(sizes.values())
    return from_dict(image_as_dict)  # TODO: Return datagroup including other metadata
