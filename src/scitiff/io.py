# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)
import json
import pathlib
import warnings
from enum import Enum
from typing import TypeVar

import numpy as np
import pydantic
import scipp as sc
import tifffile as tf
from scipp.compat.dict import from_dict

from ._schema import CHANNEL_DIMENSION_AND_COORDINATE_NAME as CHANNEL_DIM
from ._schema import (
    SCITIFF_IMAGE_STACK_DIMENSIONS,
    ImageDataArrayMetadata,
    ImageVariableMetadata,
    ScippVariable,
    SciTiffMetadata,
    SciTiffMetadataContainer,
)
from ._schema import TIME_DIMENSION_AND_COORDINATE_NAME as TIME_DIM
from ._schema import XAXIS_DIMENSION_AND_COORDINATE_NAME as X_DIM
from ._schema import YAXIS_DIMENSION_AND_COORDINATE_NAME as Y_DIM
from ._schema import ZAXIS_DIMENSION_AND_COORDINATE_NAME as Z_DIM


class IncompatibleDtypeWarning(Warning):
    """Warning for incompatible dtype."""


class UnmatchedMetadataWarning(Warning):
    """Warning for unmatched metadata."""


class ImageJMetadataNotFoundWarning(Warning):
    """Warning for missing ImageJ metadata."""


class ScitiffMetadataWarning(Warning):
    """Warning for broken scitiff metadata."""


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
    default_sizes = {X_DIM: 1, Y_DIM: 1, Z_DIM: 1, TIME_DIM: 1, CHANNEL_DIM: 1}
    final_sizes = {**default_sizes, **sizes}
    return {key: final_sizes[key] for key in order if key in final_sizes}


def _retrieve_mask_and_wrap_as_dataarray(
    da: sc.DataArray, mask_name: str | None = None
) -> sc.DataArray | None:
    """Find the matching mask and pop it out from the DataArray."""
    mask = None
    if mask_name is not None:
        mask = da.masks.pop(mask_name, None)
        if mask is not None and (mask.sizes != da.sizes):
            raise ValueError(
                f"Mask ``{mask_name}`` has unexpected size: {mask.sizes}. "
                f"Expected size is: {da.sizes}. "
                "Use `scipp.broadcast` to match the image size."
                "Or if the mask is 1D, it will be saved as metadata "
                "so you do not have to concatenate it as a separate channel."
            )

    elif len(da.masks) == 1 and (next(iter(da.masks.values()))).sizes == da.sizes:
        mask_name, mask = da.masks.popitem()
    else:  # Try to find the mask with the same size as the DataArray
        # If there is only one mask with the same size as the DataArray
        # pop it out from the DataArray and wrap it as DataArray
        matchine_masks = {
            mask_name: mask
            for mask_name, mask in da.masks.items()
            if mask.sizes == da.sizes
        }
        if len(matchine_masks) == 1:
            mask_name, mask = matchine_masks.popitem()
            da.masks.pop(mask_name)

    if mask is not None and mask_name is not None:
        mask_channel = da.copy(deep=False)
        mask_channel.data = mask.to(dtype=da.dtype)
        mask_channel.unit = da.unit  # Overwrite the unit
        mask_channel.name = mask_name
        # Assign channel coordinate
        mask_channel.coords[CHANNEL_DIM] = sc.scalar(value=Channel.mask.value)
        return mask_channel
    else:
        # If mask is not found, return None
        return None


def _retrieve_stdevs_and_wrap_as_dataarray(da: sc.DataArray) -> sc.DataArray | None:
    """Find the matching mask and pop it out from the DataArray."""
    if da.variances is not None:
        stdevs = sc.stddevs(da)
        # Assign channel coordinate
        stdevs.coords[CHANNEL_DIM] = sc.scalar(value=Channel.stdevs.value)
        return stdevs
    else:
        # If variances are not found, return None
        return None


def _concat_intensities_stdevs_and_mask(
    da: sc.DataArray,
    mask_channel: sc.DataArray | None,
    stdevs_channel: sc.DataArray | None,
) -> sc.DataArray:
    """Concatenate the intensities, stdevs and mask to a single DataArray."""
    if mask_channel is None and stdevs_channel is None:
        # Make sure ``da`` and ``c`` coordinate has consistent dimensions
        # as returned values of other cases.
        da = sc.concat([da], dim=CHANNEL_DIM)
        da.coords['c'] = sc.concat([da.coords['c']], dim=CHANNEL_DIM)
        return da

    intensities = sc.values(da)
    concatenated_channels = [intensities, stdevs_channel, mask_channel]
    return sc.concat(
        [c_da for c_da in concatenated_channels if c_da is not None],
        dim=CHANNEL_DIM,
    )


def _has_variances_or_masks(da: sc.DataArray) -> bool:
    """Check if the DataArray is multi-channel with variances or masks."""
    return da.variances is not None or bool(da.masks)


def _validate_da_and_squeeze_channel(da: sc.DataArray) -> sc.DataArray:
    """Validate the DataArray and squeeze the channel dimension.

    It will also check if the ``c`` coordinate has expected value.

    Returns
    -------
    :
        The DataArray with the channel dimension squeezed.
        If ``c`` channel is not present, it will simply return
        shallow copy of the DataArray.


    """
    # Check if ``da`` has ``c`` dimension already.
    orig_c_size = da.sizes.get(CHANNEL_DIM, 0)
    if orig_c_size > 1 and _has_variances_or_masks(da):
        raise NotImplementedError(
            f"DataArray already has c dimension with size {orig_c_size}. "
            "Multiple channel-intensities are not supported yet. "
            "It is because scitiff is mainly for high energy imaging that "
            "does not have multiple channels like optical images. "
        )
    elif orig_c_size == 1:
        da = da.squeeze(CHANNEL_DIM)
    else:
        da = da.copy(deep=False)

    # Check if ``da`` has ``c`` coordinate already, it should be a scalar.
    expected_coord = sc.scalar(Channel.intensities.value)
    c_coord = da.coords.get(CHANNEL_DIM)
    if c_coord is not None and sc.all(c_coord != expected_coord).value:
        raise ValueError(
            f"DataArray has unexpected ``c`` coordinate: {c_coord}. "
            "The ``c`` coordinate should not exist or "
            "should be a single element array in ``c`` dimension "
            "with the value of `intensities` as a string. "
        )
    # Assign the channel coordinate to the DataArray
    da.coords[CHANNEL_DIM] = expected_coord
    return da


def concat_stdevs_as_channels(da: sc.DataArray) -> sc.DataArray:
    """Concatenate intensities and stdevs into channel dimension.

    Parameters
    ----------
    da:
        The DataArray to retrieve stdevs.
        If ``variances`` property of DataArray is ``None``, it will raise an error.

    Returns
    -------
    :
        The DataArray with the stdevs concatenated in channel dimension.
        The intensities will lose ``variances``.
        as it is concatenated as a separate channel.

    Raises
    ------
    ValueError
        If the DataArray does not have ``variances``.


    .. tip::
        Use `scipp.DataArray.variances` to assign variances.


    .. tip::
        Use :func:`concat_stdevs_and_mask_as_channels`
        to concatenate only when ``variances`` are present.


    """
    if da.variances is None:
        raise ValueError(
            "DataArray does not have ``variances``. "
            "Use `scipp.DataArray.variances` to assign variances."
            "Otherwise, use ``concat_stdevs_and_mask_as_channels`` "
            "to concatenate **only** when ``variances`` are present."
        )

    da = _validate_da_and_squeeze_channel(da)
    stdevs = _retrieve_stdevs_and_wrap_as_dataarray(da)
    return _concat_intensities_stdevs_and_mask(
        da, mask_channel=None, stdevs_channel=stdevs
    )


def concat_mask_as_channels(da: sc.DataArray, mask_name: str | None) -> sc.DataArray:
    """Concatenate intensities and a mask into channel dimension.

    Parameters
    ----------
    da:
        The DataArray to retrieve a mask.
        If the mask cannot be determined, it will raise an error.

    mask_name:
        The name of the mask to be concatenated as a separate channel.
        If ``None``, it will try to find a single mask
        with the same size as the ``da``.
        If there are multiple masks with the same size, it will raise.

    Returns
    -------
    :
        The DataArray with the mask concatenated in channel dimension.
        The intensities will lose the matching mask
        as they are concatenated as channels.
        It will have `c` coordinate with the size of corresponding channels.

    Raises
    ------
    ValueError
        If the DataArray has ``variances``.


    .. tip::
        Use :func:`scipp.values` to drop ``variances``


    .. tip::
        Use :func:`concat_stdevs_and_mask_as_channels`
        to concatenate both stdevs and a mask.


    """
    if da.variances is not None:
        raise ValueError(
            "DataArray has ``variances``. "
            "Use `scipp.values` to drop ``variances``."
            "Otherwise, use ``concat_stdevs_and_mask_as_channels`` "
            "to concatenate both ``stdevs`` and ``mask``."
        )

    da = _validate_da_and_squeeze_channel(da)
    mask_channel = _retrieve_mask_and_wrap_as_dataarray(da, mask_name)

    if mask_channel is None:
        raise ValueError(
            "A mask to be concatenated cannot be determined. "
            "Use ``scipp.DataArray.assign_masks`` to assign a mask. "
        )

    return _concat_intensities_stdevs_and_mask(
        da, mask_channel=mask_channel, stdevs_channel=None
    )


def concat_stdevs_and_mask_as_channels(
    da: sc.DataArray, mask_name: str | None = None
) -> sc.DataArray:
    """Concatenate intensities, stdevs and a mask into channel dimension.

    Parameters
    ----------
    da:
        The DataArray to retrieve stdevs and a mask.
        If stdevs or a mask does not exist, it will be ignored.

    mask_name:
        The name of the mask to repack as a separate channel.
        If ``None``, it will try to find a single mask
        with the same size as the ``da``.
        If there are multiple masks with the same size, masks will be ignored.


    Returns
    -------
    :
        The DataArray with the ``stdevs`` and a ``mask`` concatenated as channels.
        The intensities will lose ``variances`` and a matching mask
        as they are repacked and concatenated as channels.
        It will have ``c`` coordinate with the size of corresponding channels.


    .. tip::
        If there are multiple masks with multi-dimensions, it cannot be saved
        in the :class:`SciTiffMetadata`
        so you will have to either make them as multiple
        1D masks or remove them before saving the image.

    """
    da = _validate_da_and_squeeze_channel(da)

    # Retrieving mask first to get rid of the mask from the DataArray
    # before concatenating the channels.
    mask_channel = _retrieve_mask_and_wrap_as_dataarray(da, mask_name)
    stdevs = _retrieve_stdevs_and_wrap_as_dataarray(da)
    # Concatenate all one-three channels

    return _concat_intensities_stdevs_and_mask(
        da, mask_channel=mask_channel, stdevs_channel=stdevs
    )


def to_scitiff_image(
    da: sc.DataArray,
    *,
    concat_stdevs_and_mask: bool = True,
    mask_name: str | None = None,
) -> sc.DataArray | sc.Dataset:
    """Modify dimnesions and shapes to match the scitiff image schema.

    The function will modify the dimensions and shapes of the DataArray.
    It also changes the order of the dimensions to match the HyperStack order.
    See :class:`SciTiffMetadata`.

    Parameters
    ----------
    da:
        The DataArray to modify as scitiff image.

    concat_stdevs_and_mask:
        If True, the function will concatenate
        ``stdevs`` and a ``mask`` to separate channels.
        If not specified, it will default to `True` and try to concatenate
        unless the DataArray is already multi-channel image and have variances
        and masks.
        In that case, it will raise an error as it is not supported yet.
        In case of the error, set it to `False`.

    mask_name:
        The name of the mask to be concatenated as a separate channel.
        It will be ignored if the ``concat_stdevs_and_mask`` is ``False``.
        If ``None``, it will try to find a single mask
        with the same size as the ``da``.
        If there are multiple masks with the same size, masks will be ignored.


    .. tip::
        You can explicitly concatenate the channels
        by using :func:`concat_stdevs_and_mask_as_channels` function.
        For example,
        if you do not want to save ``stdevs`` and ``mask`` for raw images
        but want to save them for normalized images,
        you can use :func:`concat_stdevs_and_mask_as_channels`
        function only for normalized images.


    .. warning::
        Interpretation of multi-channel images as
        ``intensities``, ``stdevs`` and ``mask``
        is not officially supported by the scitiff schema.

        It may change in the future.


    .. tip::
        If there are multiple masks with multi-dimensions, it cannot be saved
        in the scitiff format so you will have to either make them as multiple
        1D masks or remove them before saving the image.

    """
    _validate_dimensions(da)
    if concat_stdevs_and_mask and _has_variances_or_masks(da):
        da = concat_stdevs_and_mask_as_channels(da, mask_name)
    final_sizes = _ensure_hyperstack_sizes_default_order(da.sizes)
    dims = tuple(final_sizes.keys())
    shape = tuple(final_sizes.values())
    sizes: dict[str, int] = dict(zip(dims, shape, strict=True))
    # It is because ``z`` dimension and ``c`` dimension are often not present
    # but it is require by the HyperStacks and scitiffmeta schema.
    # Also, HyperStacks require specific order of dimensions.
    return sc.broadcast(da, sizes=sizes)


def _validate_dtypes(da: sc.DataArray) -> None:
    # Checking int8 and int16 as well because scipp currently does not have
    # dtype of int8 and int16, but may have in the future.
    # We are checking dtype in advance to the ``tifffile.imwrite`` function
    # raises an error, because the error message is not clear.
    # i.e. ValueError: the ImageJ format does not support data type 'd'
    # when the dtype is float64.
    if str(da.dtype) not in ("int8", "int16", "float32"):
        raise sc.DTypeError(
            f"DataArray has unexpected dtype: {da.dtype}. "
            "ImageJ only supports float32, int8, and int16 dtypes. "
            "Use `scipp.DataArray.astype` to convert the dtype. "
            "**Note that scipp currently does not support int8 and int16 dtypes.**"
        )


def _save_data_array(
    da: sc.DataArray,
    file_path: str | pathlib.Path,
    *,
    concat_stdevs_and_mask: bool = True,
    mask_name: str | None = None,
) -> None:
    final_image = to_scitiff_image(
        da, concat_stdevs_and_mask=concat_stdevs_and_mask, mask_name=mask_name
    )
    metadata = extract_metadata(final_image)
    _validate_dtypes(final_image)
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
    dg: sc.DataGroup | sc.DataArray,
    file_path: str | pathlib.Path,
    *,
    concat_stdevs_and_mask: bool = True,
    mask_name: str | None = None,
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

    concat_stdevs_and_mask:
        If `True`, ``stdevs`` and a ``mask`` will be concatenated
        into channel dimension. The default is `True`.

    mask_name:
        The name of the mask to be concatenated as a separate channel.
        It will be ignored if the ``concat_stdevs_and_mask`` is ``False``.
        If ``None`` while ``concat_stdevs_and_mask`` is ``True``,
        it will try to find a single mask with the same size as the image.
        If there are multiple masks with the same size, masks will be ignored.

    Raises
    ------
    ValueError
        If the image data has unexpected dimensions.
        The function does not understand any other names for the dimensions
        except ``x``, ``y``, ``c``, ``z``, ``t``.

    scipp.DTypeError
        If the image data has unexpected dtype.
        ImageJ only supports float32, int8, and int16 dtypes.

    """
    if isinstance(dg, sc.DataArray):
        _save_data_array(
            dg,
            file_path,
            concat_stdevs_and_mask=concat_stdevs_and_mask,
            mask_name=mask_name,
        )
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


def _wrap_as_arbitrary_variable(image_values: np.ndarray) -> sc.Variable:
    arbitrary_dims = [f"dim_{i}" for i in range(image_values.ndim)]
    try:
        return sc.array(dims=arbitrary_dims, values=image_values)
    except RuntimeError:
        # Decide the dtype to be used
        # Supported dtypes are skipped since it should have been
        # already handled in the try block.
        dtype = image_values.dtype
        match image_values.dtype:
            case np.int8 | np.int16 | np.uint8 | np.uint16:
                dtype = np.int32
            case np.uint32:
                dtype = np.int64
            case np.float16:
                dtype = np.float32
            case _:
                raise RuntimeError(
                    f"Unsupported dtype: {image_values.dtype}. "
                    "Try using ``tifffile.imread`` to load the image data. "
                ) from None

        warnings.warn(
            "The image data does not have compatible dtype. "
            f"The image has dtype of ``{image_values.dtype}``. "
            f"The dtype will be converted to ``{dtype}``.",
            stacklevel=2,
            category=IncompatibleDtypeWarning,
        )
        return sc.array(
            dims=arbitrary_dims, values=image_values.astype(dtype, copy=False)
        )


def _find_real_shape(
    meta_shape: tuple[int, ...], image_shape: tuple[int, ...]
) -> tuple[int, ...] | None:
    # We need to find the real shape of the image data
    # because tifffile sometimes adds one extra dimension
    # when loading the image into numpy array.
    if (len(meta_shape) + 1) == len(image_shape):
        if meta_shape == image_shape[1:]:
            # The first dimension is the extra dimension
            return image_shape[1:]
        elif meta_shape == image_shape[:-1]:
            # The last dimension is the extra dimension
            return image_shape[:-1]
        else:
            return image_shape
    if len(meta_shape) == len(image_shape):
        return image_shape
    else:
        return image_shape


def _read_image_as_dataarray(
    image_metadata: ImageDataArrayMetadata, file_path: str | pathlib.Path
) -> sc.DataArray:
    try:
        image = sc.zeros(
            dims=[*image_metadata.data.dims],
            shape=[*image_metadata.data.shape],
            unit=image_metadata.data.unit,
            dtype=image_metadata.data.dtype,
        )
        tf.imread(file_path, squeeze=False, out=image.values)
    except ValueError:
        image_values = tf.imread(file_path, squeeze=False)
        real_shape = _find_real_shape(image_metadata.data.shape, image_values.shape)
        if image_metadata.data.shape != real_shape:
            # tifffile has one extra dimension for some reason.
            warnings.warn(
                "Size of the image data does not match with the metadata.\n"
                f"Metadata: \n{image_metadata.data.shape}\n"
                f"Actual image data: \n{image_values.shape[1:]}\n"
                "Discarding all metadata and "
                "loading the squeezed image with arbitrary dimension names...\n",
                stacklevel=2,
                category=UnmatchedMetadataWarning,
            )
        else:
            ...
            # Size mismatches due to dtype mismatch
            # It is handled by the ``_wrap_as_arbitrary_variable`` function
        return sc.DataArray(data=_wrap_as_arbitrary_variable(image_values.squeeze()))

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


def _fall_back_loader(
    file_path: str | pathlib.Path, *, squeeze: bool = True
) -> sc.DataGroup:
    # This is a fall back loader for the image data
    # when the metadata is not found in the tiff file or it is broken.
    # The metadata is discarded and the image is loaded with arbitrary dimensions.
    image_values = tf.imread(file_path, squeeze=squeeze)
    image_da = sc.DataArray(data=_wrap_as_arbitrary_variable(image_values))
    return sc.DataGroup(image=image_da)


class Channel(Enum):
    intensities = "intensities"
    stdevs = "stdevs"
    mask = "mask"


def _resolve_channels(da: sc.DataArray) -> sc.DataArray:
    if (
        da.sizes.get(CHANNEL_DIM, 0) <= 1
        or CHANNEL_DIM not in da.coords
        or da.coords[CHANNEL_DIM].dim != CHANNEL_DIM
    ):
        # No need to resolve channels
        return da

    all_channel_names = [name.value for name in Channel]
    c_coord = da.coords[CHANNEL_DIM]
    if any(v not in all_channel_names for v in c_coord.values) or (
        Channel.intensities.value not in c_coord.values
    ):
        # If there is no intensities channel, it means we cannot resolve channels
        # automatically.
        # Therefore it returns the DataArray as is.
        return da
    # We have to copy the slice in order to assign mask and stdevs
    intensities = da[CHANNEL_DIM, sc.scalar(Channel.intensities.value)].copy(deep=True)
    # Check if there is stdevs channel and assign it
    # to the intensities variable.
    if Channel.stdevs.value in c_coord.values:
        stdevs = da[CHANNEL_DIM, sc.scalar(Channel.stdevs.value)]
        # Add stdevs as variances to the intensities variable
        intensities.variances = (stdevs.data**2).values

    # Check if there is mask channel and assign it
    # to the intensities variable.
    # There must be only one mask channel.
    # The rest of masks should all be stored as metadata as 1d array.
    if Channel.mask.value in c_coord.values:
        mask = da[CHANNEL_DIM, sc.scalar(Channel.mask.value)]
        intensities.masks['scitiff-mask'] = mask.data.astype(bool)
        intensities.masks['scitiff-mask'].unit = None  # Remove the unit as it is a mask

    return intensities


T = TypeVar("T", sc.DataArray, sc.DataGroup)


def resolve_scitiff_channels(scitiff_image: T) -> T:
    """Slice channel dimension and recombine the DataArray.

    If ``da`` is a DataGroup, it will replace the image data with the resolved
    image data. The rest of the DataGroup will be unchanged.

    Parameters
    ----------
    da:
        The DataArray or DataGroup to resolve channels.
        The DataArray should have a coordinate and dimension named 'c'
        with values of 'intensities', 'stdevs', and 'mask' (see :class:`~.Channel`).

    """
    if isinstance(scitiff_image, sc.DataGroup):
        return sc.DataGroup(
            image=_resolve_channels(scitiff_image['image']),
            **{key: value for key, value in scitiff_image.items() if key != 'image'},
        )
    else:
        return _resolve_channels(scitiff_image)


def load_scitiff(
    file_path: str | pathlib.Path,
    *,
    squeeze: bool = True,
    resolve_channels: bool = True,
) -> sc.DataGroup:
    """Load an image in SCITIFF format to a scipp data structure.

    Parameters
    ----------
    file_path:
        The path to the SCITIFF format image file.

    squeeze:
        If True, the dimensions with size 1 are squeezed out.
        You can also do it manually using ``sc.DataArray.squeeze`` method.

    resolve_channels:
        If True, the channel dimension is resolved as intensities, stdevs and mask.

        .. warning::
            Channel interpreted as intensities, variances and mask
            is not yet officially specified by ``Scitiff`` schema.
            It may change in the future.

    Returns
    -------
    :
        The loaded image data in ``scipp.DataGroup``.
        The data group should have the same structure
        as the :class:`scitiff.SciTiffMetadataContainer` except
        the image data has values loaded from the tiff file
        not just the metadata.

        .. note::
            If the metadata is not found in the tiff file or it is broken,
            the image is loaded with arbitrary dimensions
            and the metadata is discarded.
            The returned data group will have the same structure.

    Warnings
    --------
    :class:`~.IncompatibleDtypeWarning`
        If the image data has incompatible dtype.

    :class:`~UnmatchedMetadataWarning`
        If the image data has incompatible size with the metadata.
        The metadata is discarded and the image is loaded with arbitrary dimensions.

    """
    with tf.TiffFile(file_path) as tif:
        imagej_metadata = tif.imagej_metadata

    if imagej_metadata is None:
        warnings.warn(
            "ImageJ metadata not found in the tiff file.\n"
            "Loading the image with arbitrary dimensions...\n",
            stacklevel=2,
            category=ImageJMetadataNotFoundWarning,
        )
        img = _fall_back_loader(file_path, squeeze=squeeze)
    else:
        try:
            loaded_metadata = {
                key: json.loads(value)
                if _is_nested_value(SciTiffMetadataContainer.model_fields, key)
                else value
                for key, value in imagej_metadata.items()
            }
            container = SciTiffMetadataContainer(**loaded_metadata)
        except pydantic.ValidationError as e:
            warnings.warn(
                "Scitiff metadata is broken.\n"
                "Loading the image with arbitrary dimensions...\n"
                f"{e}",
                stacklevel=2,
                category=ScitiffMetadataWarning,
            )
            img = _fall_back_loader(file_path, squeeze=squeeze)
        else:
            try:
                image_da = _read_image_as_dataarray(
                    container.scitiffmeta.image, file_path
                )
                if squeeze:
                    image_da = image_da.squeeze()

                img = sc.DataGroup(image=image_da)
            except Exception as e:
                warnings.warn(
                    "Failed to reconstruct DataArray from metadata and image stack.\n"
                    "Loading the image with fall back loader...\n"
                    f"{e}",
                    stacklevel=2,
                    category=RuntimeWarning,
                )
                img = _fall_back_loader(file_path, squeeze=squeeze)

    return img if not resolve_channels else resolve_scitiff_channels(img)
