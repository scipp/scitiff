# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 ess-maintainers (https://github.com/orgs/scipp/teams/ess-maintainers)
import re

import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_identical

from scitiff import SCITIFF_IMAGE_STACK_DIMENSIONS
from scitiff.io import (
    Channel,
    concat_mask_as_channels,
    concat_stdevs_and_mask_as_channels,
    concat_stdevs_as_channels,
    resolve_scitiff_channels,
    to_scitiff_image,
)


def test_to_scitiff_image_with_variances(sample_image: sc.DataArray) -> None:
    sample_image.variances = (sample_image**2).values
    final_image = to_scitiff_image(sample_image, concat_stdevs_and_mask=True)

    assert final_image.dims == SCITIFF_IMAGE_STACK_DIMENSIONS
    expected_coord = sc.array(
        dims=['c'], values=[Channel.intensities.value, Channel.stdevs.value]
    )
    assert_identical(final_image.coords['c'], expected_coord)
    squeezed = sc.squeeze(final_image)
    assert_identical(
        squeezed['c', sc.scalar(Channel.intensities.value)].data,
        sc.values(sample_image.data),
    )
    assert_identical(
        squeezed['c', sc.scalar(Channel.stdevs.value)].data,
        sc.stddevs(sample_image.data),
    )


@pytest.mark.parametrize('mask_name', ['mask', None])
def test_to_scitiff_image_with_mask(
    sample_image: sc.DataArray, mask_name: str | None
) -> None:
    mask = sc.zeros_like(sample_image.data).astype(bool)
    mask['t', 0]['y', 1]['x', 2] = True
    mask.unit = None  # Mask should not have a unit
    sample_image.masks['mask'] = mask

    final_image = to_scitiff_image(
        sample_image, concat_stdevs_and_mask=True, mask_name=mask_name
    )

    assert final_image.dims == SCITIFF_IMAGE_STACK_DIMENSIONS
    expected_coord = sc.array(
        dims=['c'], values=[Channel.intensities.value, Channel.mask.value]
    )
    assert_identical(final_image.coords['c'], expected_coord)
    squeezed = sc.squeeze(final_image)
    assert_identical(
        squeezed['c', sc.scalar(Channel.intensities.value)].data,
        sc.values(sample_image.data),
    )
    np.all(
        squeezed['c', sc.scalar(Channel.mask.value)].data.values
        == mask.astype(sample_image.dtype).values,
    )


@pytest.mark.parametrize('mask_name', ['mask', None])
def test_to_scitiff_image_variances_and_mask(
    sample_image: sc.DataArray, mask_name: str | None
) -> None:
    mask = sc.zeros_like(sample_image.data).astype(bool)
    mask['t', 0]['y', 1]['x', 2] = True
    mask.unit = None  # Mask should not have a unit
    sample_image.masks['mask'] = mask
    sample_image.variances = (sample_image**2).values

    final_image = to_scitiff_image(
        sample_image, concat_stdevs_and_mask=True, mask_name=mask_name
    )

    assert final_image.dims == SCITIFF_IMAGE_STACK_DIMENSIONS
    expected_coord = sc.array(
        dims=['c'],
        values=[Channel.intensities.value, Channel.stdevs.value, Channel.mask.value],
    )
    assert_identical(final_image.coords['c'], expected_coord)
    squeezed = sc.squeeze(final_image)
    assert_identical(
        squeezed['c', sc.scalar(Channel.intensities.value)].data,
        sc.values(sample_image.data),
    )
    assert_identical(
        squeezed['c', sc.scalar(Channel.stdevs.value)].data,
        sc.stddevs(sample_image.data),
    )
    np.all(
        squeezed['c', sc.scalar(Channel.mask.value)].data.values
        == mask.astype(sample_image.dtype).values,
    )


def test_concat_scitiff_channels_intensities_variances(
    sample_image: sc.DataArray,
) -> None:
    sample_image.variances = (sample_image**2).values
    two_channel_image = concat_stdevs_as_channels(sample_image)

    assert_identical(
        two_channel_image.coords['c'],
        sc.array(dims=['c'], values=[Channel.intensities.value, Channel.stdevs.value]),
    )
    assert_identical(
        two_channel_image['c', sc.scalar(Channel.intensities.value)].data,
        sc.values(sample_image.data),
    )
    assert_identical(
        two_channel_image['c', sc.scalar(Channel.stdevs.value)].data,
        sc.stddevs(sample_image.data),
    )


def test_concat_scitiff_channels_intensities_mask_has_variances_raises(
    sample_image: sc.DataArray,
) -> None:
    sample_image.variances = (sample_image**2).values
    with pytest.raises(ValueError, match='variances'):
        concat_mask_as_channels(sample_image, mask_name='')


@pytest.mark.parametrize('mask_name', ['mask', None])
def test_concat_scitiff_channels_intensities_mask(
    sample_image: sc.DataArray, mask_name: str | None
) -> None:
    mask = sc.zeros_like(sample_image.data).astype(bool)
    mask['t', 0]['y', 1]['x', 2] = True
    sample_image.masks['mask'] = mask

    two_channel_image = concat_mask_as_channels(sample_image, mask_name=mask_name)

    assert_identical(
        two_channel_image.coords['c'],
        sc.array(dims=['c'], values=[Channel.intensities.value, Channel.mask.value]),
    )
    assert_identical(
        two_channel_image['c', sc.scalar(Channel.intensities.value)].data,
        sc.values(sample_image.data),
    )
    np.all(
        two_channel_image['c', sc.scalar(Channel.mask.value)].data.values
        == mask.astype(sample_image.dtype).values,
    )


def test_concat_scitiff_channels_intensities_mask_too_many_masks_raises(
    sample_image: sc.DataArray,
) -> None:
    mask = sc.zeros_like(sample_image.data).astype(bool)
    sample_image.masks['mask'] = mask
    sample_image.masks['mask2'] = mask

    with pytest.raises(
        ValueError, match=re.escape("A mask to be concatenated cannot be determined.")
    ):
        concat_mask_as_channels(sample_image, mask_name=None)


def test_concat_scitiff_channels_intensities_mask_not_same_size_raises(
    sample_image: sc.DataArray,
) -> None:
    mask = sc.zeros_like(sample_image.data)['t', 0].astype(bool)
    sample_image.masks['mask'] = mask

    with pytest.raises(
        ValueError, match=re.escape("has unexpected size: {'y': 3, 'x': 4}")
    ):
        concat_mask_as_channels(sample_image, mask_name='mask')


@pytest.mark.parametrize('mask_name', ['mask', None])
def test_concat_scitiff_channels_intensities_variances_masks(
    sample_image: sc.DataArray, mask_name: str | None
) -> None:
    mask = sc.zeros_like(sample_image.data).astype(bool)
    mask['t', 0]['y', 1]['x', 2] = True
    sample_image.masks['mask'] = mask
    sample_image.variances = (sample_image**2).values

    three_channel_image = concat_stdevs_and_mask_as_channels(
        sample_image, mask_name=mask_name
    )
    expected_coord = sc.array(
        dims=['c'],
        values=[Channel.intensities.value, Channel.stdevs.value, Channel.mask.value],
    )
    assert_identical(three_channel_image.coords['c'], expected_coord)
    assert_identical(
        three_channel_image['c', sc.scalar(Channel.intensities.value)].data,
        sc.values(sample_image.data),
    )
    assert_identical(
        three_channel_image['c', sc.scalar(Channel.stdevs.value)].data,
        sc.stddevs(sample_image.data),
    )
    assert_identical(
        three_channel_image['c', sc.scalar(Channel.mask.value)].data,
        mask.astype(sample_image.dtype),
    )


@pytest.mark.parametrize('mask_name', ['mask', None])
def test_concat_scitiff_channels_intensities_variances_only_mask(
    sample_image: sc.DataArray, mask_name: str | None
) -> None:
    mask = sc.zeros_like(sample_image.data).astype(bool)
    mask['t', 0]['y', 1]['x', 2] = True
    sample_image.masks['mask'] = mask

    two_channel_image = concat_stdevs_and_mask_as_channels(
        sample_image, mask_name=mask_name
    )
    expected_coord = sc.array(
        dims=['c'],
        values=[Channel.intensities.value, Channel.mask.value],
    )
    assert_identical(two_channel_image.coords['c'], expected_coord)
    assert_identical(
        two_channel_image['c', sc.scalar(Channel.intensities.value)].data,
        sc.values(sample_image.data),
    )
    assert_identical(
        two_channel_image['c', sc.scalar(Channel.mask.value)].data,
        mask.astype(sample_image.dtype),
    )


def test_concat_scitiff_channels_intensities_variances_masks_only_variances(
    sample_image: sc.DataArray,
) -> None:
    sample_image.variances = (sample_image**2).values

    two_channel_image = concat_stdevs_and_mask_as_channels(sample_image)

    expected_coord = sc.array(
        dims=['c'],
        values=[Channel.intensities.value, Channel.stdevs.value],
    )
    assert_identical(two_channel_image.coords['c'], expected_coord)
    assert_identical(
        two_channel_image['c', sc.scalar(Channel.intensities.value)].data,
        sc.values(sample_image.data),
    )
    assert_identical(
        two_channel_image['c', sc.scalar(Channel.stdevs.value)].data,
        sc.stddevs(sample_image.data),
    )


def test_concat_scitiff_channels_intensities_variances_masks_nothing(
    sample_image: sc.DataArray,
) -> None:
    just_image = concat_stdevs_and_mask_as_channels(sample_image)
    expected_coord = sc.array(dims=['c'], values=[Channel.intensities.value])
    assert_identical(just_image.coords['c'], expected_coord)
    assert_identical(
        just_image['c', sc.scalar(Channel.intensities.value)].drop_coords('c'),
        sample_image,
    )


def test_resolve_scitiff_channels_intensities_variances(
    sample_image: sc.DataArray,
) -> None:
    sample_stdevs = sample_image.copy(deep=False)
    two_channel_image = sc.concat([sample_image, sample_stdevs], dim='c')
    two_channel_image.coords['c'] = sc.array(
        dims=['c'], values=[Channel.intensities.value, Channel.stdevs.value]
    )
    resolved_image = resolve_scitiff_channels(two_channel_image)
    assert resolved_image.dims == ('t', 'y', 'x')
    assert_identical(sample_image.data, sc.values(resolved_image.data))
    assert np.all(resolved_image.variances == (sample_image**2).values)


def test_resolve_scitiff_channels_intensities_mask(
    sample_image: sc.DataArray,
) -> None:
    sample_mask = sample_image < sc.scalar(200.0, unit=sample_image.unit)
    sample_mask.unit = sample_image.unit
    two_channel_image = sc.concat(
        [sample_image, sample_mask.astype('float32', copy=False)], dim='c'
    )
    two_channel_image.coords['c'] = sc.array(
        dims=['c'], values=[Channel.intensities.value, Channel.mask.value]
    )
    resolved_image = resolve_scitiff_channels(two_channel_image)
    assert resolved_image.dims == ('t', 'y', 'x')
    assert_identical(sample_image.data, sc.values(resolved_image.data))

    sample_mask.unit = None
    assert_identical(
        resolved_image.masks['scitiff-mask'], sample_mask.data.astype(bool, copy=False)
    )


def test_resolve_scitiff_channels_intensities_variances_mask(
    sample_image: sc.DataArray,
) -> None:
    sample_stdevs = sample_image.copy(deep=False)
    sample_mask = sample_image < sc.scalar(200.0, unit=sample_image.unit)
    sample_mask.unit = sample_image.unit
    three_channel_image = sc.concat(
        [sample_stdevs, sample_image, sample_mask.astype('float32', copy=False)],
        dim='c',
    )
    three_channel_image.coords['c'] = sc.array(
        dims=['c'],
        values=[Channel.stdevs.value, Channel.intensities.value, Channel.mask.value],
    )
    resolved_image = resolve_scitiff_channels(three_channel_image)
    assert resolved_image.dims == ('t', 'y', 'x')
    assert np.all(resolved_image.variances == (sample_image**2).values)
    assert_identical(sample_image.data, sc.values(resolved_image.data))

    sample_mask.unit = None
    assert_identical(
        resolved_image.masks['scitiff-mask'], sample_mask.data.astype(bool, copy=False)
    )


def test_resolve_scitiff_channels_intensities_variances_mask_on_datagroup(
    sample_image: sc.DataArray,
) -> None:
    sample_stdevs = sample_image.copy(deep=False)
    sample_stdevs.unit = sample_image.unit
    sample_mask = sample_image < sc.scalar(200.0, unit=sample_image.unit)
    sample_mask.unit = sample_image.unit
    three_channel_image = sc.concat(
        [sample_stdevs, sample_image, sample_mask.astype('float32', copy=False)],
        dim='c',
    )
    three_channel_image.coords['c'] = sc.array(
        dims=['c'],
        values=[Channel.stdevs.value, Channel.intensities.value, Channel.mask.value],
    )
    resolved_image = resolve_scitiff_channels(sc.DataGroup(image=three_channel_image))[
        'image'
    ]
    assert resolved_image.dims == ('t', 'y', 'x')
    assert np.all(resolved_image.variances == (sample_image.data**2).values)
    assert_identical(sample_image.data, sc.values(resolved_image.data))

    sample_mask.unit = None
    assert_identical(
        resolved_image.masks['scitiff-mask'], sample_mask.data.astype(bool, copy=False)
    )
