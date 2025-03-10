# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)
import json
import pathlib

import pydantic
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


def _wrap_unit(unit: str | None) -> str | None:
    # str(None), which is `None` is interpreted as `N` (neuton) when
    # it is loaded back from the json file.
    return str(unit) if unit is not None else None


def _scipp_variable_to_model(var: sc.Variable) -> ScippVariable:
    if var.ndim > 1:
        raise ValueError(
            "Only 1-dimensional variable is allowed for metadata. "
            "The variable has more than 1 dimension."
        )
    # string values does not have `tolist` method
    if hasattr(var, "values") and hasattr(var.values, "tolist"):
        values = var.values.tolist()
    else:
        values = list(var.values)

    return ScippVariable(
        dims=var.dims,
        dtype=str(var.dtype),
        shape=var.shape,
        unit=_wrap_unit(var.unit),
        values=values,
    )


def _scipp_variable_to_metadata_model(var: sc.Variable) -> ImageVariableMetadata:
    # Image data variable should have more than 1 dimension
    # So we do not warn the user about the multi-dimensional variable
    return ImageVariableMetadata(
        dims=var.dims, dtype=str(var.dtype), shape=var.shape, unit=_wrap_unit(var.unit)
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


def _validate_dimensions(da: sc.DataArray) -> None:
    if illegal_dim := [
        dim for dim in da.dims if dim not in SCITIFF_IMAGE_STACK_DIMENSIONS
    ]:
        raise ValueError(
            f"DataArray has unexpected dimensions: {','.join(illegal_dim)}. "
            f"Allowed dimensions are: {SCITIFF_IMAGE_STACK_DIMENSIONS} "
            "Use `scipp.DataArray.rename_dims` to rename the dimensions."
        )


def _ensure_hyperstack_sizes_default_order(sizes: dict) -> dict:
    # Order of the dimensions is according to the HyperStacks tiff format.
    order = SCITIFF_IMAGE_STACK_DIMENSIONS
    default_sizes = {"x": 1, "y": 1, "z": 1, "t": 1, "c": 1}
    final_sizes = {**default_sizes, **sizes}
    return {key: final_sizes[key] for key in order if key in final_sizes}


def to_scitiff_image(da: sc.DataArray) -> sc.DataArray:
    _validate_dimensions(da)
    final_sizes = _ensure_hyperstack_sizes_default_order(da.sizes)
    dims = tuple(final_sizes.keys())
    shape = tuple(final_sizes.values())
    sizes: dict[str, int] = dict(zip(dims, shape, strict=True))
    # It is because ``z`` dimension and ``c`` dimension are often not present
    # but it is require by the HyperStacks and scitiffmeta schema.
    # Also, HyperStacks require specific order of dimensions.
    return sc.broadcast(da, sizes=sizes)


def _save_data_array(da: sc.DataArray, file_path: str | pathlib.Path) -> None:
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


def save_scitiff(
    dg: sc.DataGroup | sc.DataArray, file_path: str | pathlib.Path
) -> None:
    """Save an image in scipp data structure to a SCITIFF format including metadata.

    The image is transposed to the default HyperStack order
    before saving the image,
    which is ``x``, ``y``, ``c``, ``z``, ``t``.
    (From the innermost dimension to the outermost dimension)

    .. note::
        Before the image is saved, it is broadcasted to match the HyperStack
        even if part of dimensions are not present.
        For example, if the image has only ``x`` and ``y`` dimensions,
        the image will be broadcasted to ``x``, ``y``, ``c``, ``z``, ``t`` dimensions
        with size of 1 for each dimension that is not present.

        .. warning::
            :func:`load_scitiff` function will squeeze the dimensions
            with size 1 by default.
            Other image tools also may squeeze the image
            and drop the dimensions with size 1 by default.

    .. note::

        .. csv-table::  Dimensions of the image stack
            :header: "Name", "Description"

                     "x",    "x-axis (width)"
                     "y",    "y-axis (height)"
                     "c",    "channel-axis"
                     "z",    "z-axis"
                     "t",    "time-axis(time-of-flight or other time-like dimension)"

        .. warning::

            For neutron imaging, ``c`` dimension may not represent color channels.


    Parameters
    ----------
    dg:
        The image data to save.

    file_path:
        The path to save the image data.

    Raises
    ------
    ValueError
        If the image data has unexpected dimensions.
        The function does not understand any other names for the dimensions
        except ``x``, ``y``, ``c``, ``z``, ``t``.

    """
    if isinstance(dg, sc.DataArray):
        _save_data_array(dg, file_path)
    else:
        raise NotImplementedError("Saving DataGroup to SCITIFF is not yet implemented.")


def _is_nested_value(
    model_fields: dict[str, pydantic.fields.FieldInfo], key: str
) -> bool:
    return (
        key in model_fields
        and (field_type := model_fields[key].annotation) is not None
        and issubclass(field_type, pydantic.BaseModel)
    )


def _read_image_as_dataarray(
    image_metadata: ImageDataArrayMetadata, file_path: str | pathlib.Path
) -> sc.DataArray:
    image = sc.zeros(
        dims=[*image_metadata.data.dims],
        shape=[*image_metadata.data.shape],
        unit=image_metadata.data.unit,
        dtype=image_metadata.data.dtype,
    )
    tf.imread(file_path, squeeze=False, out=image.values)
    # We are loading image directly to the allocated array.
    # In this way we save memory and time.
    # Also, ``tifffile.imread`` adds one extra dimension
    # when loading the image into numpy array but in this way
    # the data is directly loaded into the exact shape.
    # Therefore we manually build the DataArray
    # instead of using ``scipp.from_dict`` function.
    # However, each coordinate and mask is loaded using ``from_dict`` function
    # since they are serialized as dictionaries.
    coords = {
        key: from_dict(value.model_dump())
        for key, value in image_metadata.coords.items()
    }
    masks = {
        key: from_dict(value.model_dump())
        for key, value in image_metadata.masks.items()
    }
    return sc.DataArray(
        data=image, coords=coords, masks=masks, name=image_metadata.name or ''
    )


def load_scitiff(
    file_path: str | pathlib.Path, *, squeeze: bool = True
) -> sc.DataGroup:
    """Load an image in SCITIFF format to a scipp data structure.

    Parameters
    ----------
    file_path:
        The path to the SCITIFF format image file.

    squeeze:
        If True, the dimensions with size 1 are squeezed out.
        You can also do it manually using ``sc.DataArray.squeeze`` method.

    Returns
    -------
    :
        The loaded image data in ``scipp.DataGroup``.
        The data group should have the same structure
        as the :class:`scitiff.SciTiffMetadataContainer` except
        the image data has values loaded from the tiff file
        not just the metadata.

    """
    with tf.TiffFile(file_path) as tif:
        if tif.imagej_metadata is None:
            raise ValueError(
                f"ImageJ metadata is not found in the tiff file: {file_path}"
            )
        loaded_metadata = {
            key: json.loads(value)
            if _is_nested_value(SciTiffMetadataContainer.model_fields, key)
            else value
            for key, value in tif.imagej_metadata.items()
        }
        container = SciTiffMetadataContainer(**loaded_metadata)

    image_da = _read_image_as_dataarray(container.scitiffmeta.image, file_path)
    if squeeze:
        image_da = image_da.squeeze()

    return sc.DataGroup(image=image_da)
