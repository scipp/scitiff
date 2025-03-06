# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)
import json
import pathlib

import scipp as sc
import tifffile as tf
from scipp.compat.dict import from_dict

from ._schema import (
    SCITIFF_IMAGE_STACK_DIMENSIONS,
    ImageDataArrayMetadata,
    ImageVariableMetadata,
    ScippVariable,
    SciTiffMetadata,
    SciTiffMetadataContainer,
)


def _scipp_variable_to_model(var: sc.Variable) -> ScippVariable:
    import warnings

    if var.ndim > 1:
        warnings.warn(
            "The variable has more than 1 dimension. "
            "Variables for metadata should not take too much space. "
            "Consider using single dimension variable for metadata.",
            ResourceWarning,
            stacklevel=2,
        )
    return ScippVariable(
        dims=tuple(var.dims),
        dtype=str(var.dtype),
        shape=tuple(var.shape),
        unit=str(var.unit),
        values=var.values.tolist(),
    )


def _scipp_variable_to_metadata_model(var: sc.Variable) -> ImageVariableMetadata:
    # Image data variable should have more than 1 dimension
    # So we do not warn the user about the multi-dimensional variable
    return ImageVariableMetadata(
        dims=tuple(var.dims),
        dtype=str(var.dtype),
        shape=tuple(var.shape),
        unit=str(var.unit),
    )


def _extract_metadata_from_dataarray(da: sc.DataArray) -> ImageDataArrayMetadata:
    return ImageDataArrayMetadata(
        data=_scipp_variable_to_metadata_model(da.data),
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


def to_scitiff_image(da: sc.DataArray) -> sc.DataArray:
    default_sizes = {"x": 1, "y": 1, "z": 1, "t": 1, "c": 1}
    final_sizes = {**default_sizes, **da.sizes}
    # Order of the dimensions is according to the HyperStacks tiff format.
    order = SCITIFF_IMAGE_STACK_DIMENSIONS  # ("c", "t", "z", "y", "x")
    final_sizes = {key: final_sizes[key] for key in order if key in final_sizes}
    dims = tuple(final_sizes.keys())
    shape = tuple(final_sizes.values())
    sizes: dict[str, int] = dict(zip(dims, shape, strict=True))
    # It is because ``z`` dimension and ``c`` dimension are often not present
    # but it is require by the HyperStacks and scitiffmeta schema.
    # Also, HyperStacks require specific order of dimensions.
    return sc.broadcast(da, sizes=sizes)


def _export_data_array(da: sc.DataArray, file_path: str | pathlib.Path) -> None:
    final_image = to_scitiff_image(da)
    metadata = extract_metadata(final_image)
    tf.imwrite(
        file_path,
        final_image.values,
        imagej=True,
        metadata={
            key: json.dumps(value)
            for key, value in metadata.model_dump(mode="json").items()
            # Tiff metadata will automatically be translated to json-like-string
            # so we need to convert the metadata to string in advance
            # to catch the error early and make sure it can be loaded back.
        },
        dtype=str(final_image.dtype),
    )


def _export_data_group(dg: sc.DataGroup, file_path: str | pathlib.Path) -> None:
    raise NotImplementedError("Exporting DataGroup to SCITIFF is not yet implemented.")


def export_scitiff(
    dg: sc.DataGroup | sc.DataArray, file_path: str | pathlib.Path
) -> None:
    if isinstance(dg, sc.DataArray):
        _export_data_array(dg, file_path)
    else:
        _export_data_group(dg, file_path)


def load_scitiff(file_path: str | pathlib.Path, squeeze: bool = True) -> sc.DataGroup:
    with tf.TiffFile(file_path) as tif:
        container = SciTiffMetadataContainer(
            scitiffmeta=json.loads(tif.imagej_metadata["scitiffmeta"])
        )

    img_metadata = container.scitiffmeta.image
    image = sc.zeros(
        dims=[*img_metadata.data.dims],
        shape=[*img_metadata.data.shape],
        unit=img_metadata.data.unit,
        dtype=img_metadata.data.dtype,
    )
    tf.imread(file_path, squeeze=False, out=image.values)
    coords = {
        key: from_dict(value.model_dump()) for key, value in img_metadata.coords.items()
    }
    masks = {
        key: from_dict(value.model_dump()) for key, value in img_metadata.masks.items()
    }
    image_da = sc.DataArray(
        data=image, coords=coords, masks=masks, name=img_metadata.name or ''
    )
    return (
        sc.DataGroup(image=image_da.squeeze())
        if squeeze
        else sc.DataGroup(image=image_da)
    )
