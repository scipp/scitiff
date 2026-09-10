# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 ess-maintainers (https://github.com/orgs/scipp/teams/ess-maintainers)

import scipp as sc

from .io import Channel


def _validate_scitiff_image(da: sc.DataArray) -> None:
    if da.sizes.get('c', 0) > 1 and 'c' not in da.coords:
        raise ValueError(
            "c(Channel) coordinate not found in the multichannel image. "
            "Cannot decide intensity channel."
        )


def _variances_and_masks_in_channels(da: sc.DataArray) -> bool:
    return 'c' in da.coords and da.sizes['c'] > 1


def values(
    image_container: sc.DataGroup | sc.DataArray,
    *,
    deepcopy: bool = False,
) -> sc.DataArray:
    """Retrieve values(intensities) from a scitiff image.

    Parameters
    ----------
    image_container:
        A data group or a data array that contains image values.
        If the variances and masks are concatenated in the channel
        dimension, it must have `c` coordinate (intensities, stdevs, mask),
        so that which slice contains the intensities.
        If it is a data group, it must contain 'image'.

    Returns
    -------
    :
        An image only with intensities as values as a DataArray.
        Always return a shallow copy of the DataArray by default.

    """
    if isinstance(image_container, sc.DataArray):
        image = image_container
    elif isinstance(image_container, sc.DataGroup):
        if 'image' not in image_container:
            raise KeyError(
                "'image' not found in the image container. "
                "Cannot find an image in the data group."
            )
        image = image_container['image']
    else:
        raise TypeError(f'Unsupported type {type(image_container)}')

    _validate_scitiff_image(image)

    if _variances_and_masks_in_channels(image):
        sliced = image['c', sc.scalar(Channel.intensities.value)]
    elif image.variances is not None:
        sliced = sc.values(image)

    # It drops the multidimensional mask to be consistent with the case
    # where the mask is concatnenated into channel dimensions.
    multi_dim_masks = [name for name, mask in image.masks.items() if len(mask.dims) > 1]
    sliced = sliced.drop_masks(multi_dim_masks)
    return sliced.copy(deep=deepcopy)
