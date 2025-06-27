# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)
import json
import pathlib

import numpy as np
import pytest
import scipp as sc
import tifffile as tf
from scipp.testing import assert_identical

from scitiff import SCITIFF_IMAGE_STACK_DIMENSIONS
from scitiff.io import (
    Channel,
    ImageJMetadataNotFoundWarning,
    IncompatibleDtypeWarning,
    ScitiffMetadataWarning,
    UnmatchedMetadataWarning,
    concat_mask_as_channels,
    concat_stdevs_and_mask_as_channels,
    concat_stdevs_as_channels,
    extract_metadata,
    load_scitiff,
    resolve_scitiff_channels,
    save_scitiff,
    to_scitiff_image,
)


@pytest.fixture
def sample_image() -> sc.DataArray:
    pattern = [[i * 400 + j for j in range(4)] for i in range(3)]
    sample_img = sc.DataArray(
        data=sc.array(
            dims=['t', 'y', 'x'],
            values=[pattern, pattern[::-1]],
            unit='counts',
            dtype=sc.DType.float32,
        ),
        coords={
            't': sc.array(dims=['t'], values=[0, 1], unit='s'),
            'y': sc.linspace(dim='y', start=0.0, stop=300.0, num=3, unit='mm'),
            'x': sc.linspace(dim='x', start=0.0, stop=400.0, num=4, unit='mm'),
        },
    )
    return sample_img


def test_export_and_load_scitiff(sample_image, tmp_path) -> None:
    tmp_file_path = tmp_path / 'test.tiff'
    save_scitiff(sample_image, tmp_file_path)
    loaded_image = load_scitiff(tmp_file_path)
    # exported image already has right order of dimensions
    assert sc.identical(sample_image, loaded_image)


def test_export_and_load_scitiff_with_scalar_coord(sample_image, tmp_path) -> None:
    sample_image.coords['Ltotal'] = sc.scalar(0.0, unit='mm')
    tmp_file_path = tmp_path / 'test_scalar_coord.tiff'
    save_scitiff(sample_image, tmp_file_path)
    loaded_image = load_scitiff(tmp_file_path)
    # exported image already has right order of dimensions
    assert sc.identical(sample_image, loaded_image)


@pytest.fixture
def sample_image_2d_coordinate(sample_image: sc.DataArray) -> sc.DataArray:
    new_image = sample_image.copy()
    flattend_x = sc.flatten(sample_image, dims=['y', 'x'], to='pos').coords['x']
    new_image.coords['x'] = sc.fold(
        flattend_x, dim='pos', dims=['y', 'x'], shape=[3, 4]
    )
    return new_image


def test_export_illegal_dimension_raises(sample_image: sc.DataArray) -> None:
    with pytest.raises(ValueError, match='DataArray has unexpected dimensions: meh'):
        save_scitiff(sample_image.rename_dims({'x': 'meh'}), '')


def test_export_multi_dimension_coordinate_raises(
    sample_image_2d_coordinate: sc.DataArray,
) -> None:
    with pytest.raises(
        ValueError, match='Only 1-dimensional variable is allowed for metadata.'
    ):
        save_scitiff(sample_image_2d_coordinate, 'test.tiff')


def test_load_squeeze_false(sample_image, tmp_path) -> None:
    tmp_file_path = tmp_path / 'test.tiff'
    save_scitiff(sample_image, tmp_file_path)
    loaded_image = load_scitiff(tmp_file_path, squeeze=False)
    assert loaded_image.dims == SCITIFF_IMAGE_STACK_DIMENSIONS


def test_save_wrong_dtype_raises(sample_image: sc.DataArray) -> None:
    with pytest.raises(sc.DTypeError, match='DataArray has unexpected dtype: int64'):
        save_scitiff(sample_image.astype(int), 'test.tiff')
    with pytest.raises(sc.DTypeError, match='DataArray has unexpected dtype: float64'):
        save_scitiff(sample_image.astype(float), 'test.tiff')


def _save_data_array_with_wrong_dtype(
    da: sc.DataArray, file_path: str | pathlib.Path, dtype: str
) -> None:
    final_image = to_scitiff_image(da)
    metadata = extract_metadata(final_image)
    tf.imwrite(
        file_path,
        final_image.values.astype(dtype),
        imagej=True if dtype in ['uint8', 'uint16', 'float32'] else False,
        metadata={
            key: json.dumps(value)
            for key, value in metadata.model_dump(mode="json").items()
        },
        photometric='minisblack',  # int8/int16/float16 dtype can be stored as `rgb`
        # in some versions of `tifffile`
        dtype=dtype,
    )


@pytest.mark.parametrize(
    ('dtype', 'expected_dtype'),
    [('uint8', 'int32'), ('uint16', 'int32')],
)
def test_load_scipp_incompatible_dtype_fallback(
    sample_image: sc.DataArray, tmp_path, dtype, expected_dtype
) -> None:
    """Test scipp incompatible dtypes but compatible with ImageJ."""
    tmp_file_path = tmp_path / 'wrong_dtype.tiff'
    _save_data_array_with_wrong_dtype(sample_image, tmp_file_path, dtype=dtype)
    with pytest.warns(
        IncompatibleDtypeWarning,
        match=f"dtype of ``{dtype}``. "
        f"The dtype will be converted to ``<class 'numpy.{expected_dtype}'>``",
    ):
        loaded_image = load_scitiff(tmp_file_path, resolve_channels=False)['image']

    assert loaded_image.dims == tuple(f"dim_{i}" for i in range(3))  # Image is squeezed
    assert loaded_image.dtype == expected_dtype


@pytest.mark.parametrize(
    ('dtype', 'expected_dtype'),
    [('int8', 'int32'), ('int16', 'int32'), ('float16', 'float32')],
)
def test_load_imagej_scipp_incompatible_dtype_fallback(
    sample_image: sc.DataArray, tmp_path, dtype, expected_dtype
) -> None:
    """Test ImageJ and scipp incompatible dtypes."""
    tmp_file_path = tmp_path / 'wrong_dtype.tiff'
    _save_data_array_with_wrong_dtype(sample_image, tmp_file_path, dtype=dtype)
    with pytest.warns(
        IncompatibleDtypeWarning,
        match=f"dtype of ``{dtype}``. "
        f"The dtype will be converted to ``<class 'numpy.{expected_dtype}'>``",
    ):
        with pytest.warns(
            ImageJMetadataNotFoundWarning, match='ImageJ metadata not found'
        ):  # These dtypes are not supported by tifffile with `imagej=True`
            loaded_image = load_scitiff(tmp_file_path)['image']

    assert loaded_image.dims == tuple(f"dim_{i}" for i in range(3))  # Image is squeezed
    assert loaded_image.dtype == expected_dtype


@pytest.mark.parametrize(('dtype', 'expected_dtype'), [('float64', 'float64')])
def test_load_imagej_incompatible_dtype_fallback(
    sample_image: sc.DataArray, tmp_path, dtype, expected_dtype
) -> None:
    """Test ImageJ incompatible but scipp compatible dtypes.

    float64 is supported by scipp so it should not raise a dtype conversion warning.
    """
    tmp_file_path = tmp_path / 'wrong_dtype.tiff'
    _save_data_array_with_wrong_dtype(sample_image, tmp_file_path, dtype=dtype)
    with pytest.warns(
        ImageJMetadataNotFoundWarning, match='ImageJ metadata not found'
    ):  # These dtypes are not supported by tifffile with `imagej=True`
        loaded_image = load_scitiff(tmp_file_path)['image']

    assert loaded_image.dims == tuple(f"dim_{i}" for i in range(3))  # Image is squeezed
    assert loaded_image.dtype == expected_dtype


def _save_data_array_with_unmatching_shape(
    da: sc.DataArray, file_path: str | pathlib.Path
) -> None:
    final_image = to_scitiff_image(da)
    metadata = extract_metadata(final_image)
    tf.imwrite(
        file_path,
        final_image['t', 0].values,
        imagej=True,
        metadata={
            key: json.dumps(value)
            for key, value in metadata.model_dump(mode="json").items()
        },
        dtype=str(final_image.dtype),
    )


def test_load_incompatible_metadata_warns(sample_image: sc.DataArray, tmp_path) -> None:
    tmp_file_path = tmp_path / 'unmatching_size.tiff'
    _save_data_array_with_unmatching_shape(sample_image, tmp_file_path)
    with pytest.warns(
        UnmatchedMetadataWarning,
        match="Size of the image data does not match with the metadata.",
    ):
        loaded_image = load_scitiff(tmp_file_path)['image']

    assert loaded_image.dims == tuple(f"dim_{i}" for i in range(2))  # Image is squeezed


def test_load_without_metadata_warns(tmp_path) -> None:
    import numpy as np

    no_metadata_file_path = tmp_path / 'no_metadata.tiff'
    tf.imwrite(no_metadata_file_path, [[1.0, 2], [3, 4]])
    with pytest.warns(
        ImageJMetadataNotFoundWarning, match='ImageJ metadata not found.'
    ):
        loaded_image: sc.DataArray = load_scitiff(no_metadata_file_path)['image']

    assert loaded_image.dims == tuple(f"dim_{i}" for i in range(2))
    assert np.all(loaded_image.values == np.array([[1.0, 2], [3, 4]]))


def test_load_broken_metadata_warns(tmp_path) -> None:
    import numpy as np

    no_metadata_file_path = tmp_path / 'broken_metadata.tiff'
    arbitrary_image = np.array([[1.0, 2], [3, 4]], dtype='float32')
    tf.imwrite(
        no_metadata_file_path, arbitrary_image, imagej=True, metadata={"meh": "meh"}
    )
    with pytest.warns(ScitiffMetadataWarning, match='Scitiff metadata is broken.'):
        loaded_image: sc.DataArray = load_scitiff(no_metadata_file_path)['image']

    assert loaded_image.dims == tuple(f"dim_{i}" for i in range(2))
    assert np.all(loaded_image.values == arbitrary_image)


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
        ValueError, match="A mask to be concatenated cannot be determined. "
    ):
        concat_mask_as_channels(sample_image, mask_name=None)


def test_concat_scitiff_channels_intensities_mask_not_same_size_raises(
    sample_image: sc.DataArray,
) -> None:
    mask = sc.zeros_like(sample_image.data)['t', 0].astype(bool)
    sample_image.masks['mask'] = mask

    with pytest.raises(ValueError, match="has unexpected size: {'y': 3, 'x': 4}"):
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
