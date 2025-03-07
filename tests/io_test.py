# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)
import pytest
import scipp as sc

from scitiff import SCITIFF_IMAGE_STACK_DIMENSIONS
from scitiff.io import export_scitiff, load_scitiff


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
    export_scitiff(sample_image, tmp_file_path)
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
        export_scitiff(sample_image.rename_dims({'x': 'meh'}), '')


def test_export_multi_dimension_coordinate_raises(
    sample_image_2d_coordinate: sc.DataArray,
) -> None:
    with pytest.raises(
        ValueError, match='Only 1-dimensional variable is allowed for metadata.'
    ):
        export_scitiff(sample_image_2d_coordinate, 'test.tiff')


def test_load_squeeze_false(sample_image, tmp_path) -> None:
    tmp_file_path = tmp_path / 'test.tiff'
    export_scitiff(sample_image, tmp_file_path)
    loaded_image = load_scitiff(tmp_file_path, squeeze=False)
    assert loaded_image.dims == SCITIFF_IMAGE_STACK_DIMENSIONS
