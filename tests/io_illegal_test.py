# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 ess-maintainers (https://github.com/orgs/scipp/teams/ess-maintainers)
import pathlib
import re

import pytest
import scipp as sc

from scitiff.io import save_scitiff


def test_export_and_load_scitiff_dagagroup_wrong_daqmetadatatype_raises(
    sample_image_datagroup: sc.DataGroup,
) -> None:
    import uuid

    tmp_name = uuid.uuid4().hex + '.tiff'
    sample_image_datagroup['daq'] = {}
    with pytest.raises(TypeError, match='DAQMetadata'):
        save_scitiff(sample_image_datagroup, tmp_name)

    # Should not have written the file.
    assert not pathlib.Path(tmp_name).exists()


@pytest.fixture
def sample_image_3d_coordinate(sample_image: sc.DataArray) -> sc.DataArray:
    new_image = sample_image.copy()
    flattend_x = sc.flatten(sample_image, dims=['t', 'y', 'x'], to='lambda').coords['t']
    new_image.coords['lambda'] = sc.fold(
        flattend_x, dim='lambda', dims=['y', 'x', 't'], shape=[3, 4, 2]
    )
    return new_image


def test_export_illegal_dimension_raises(sample_image: sc.DataArray) -> None:
    with pytest.raises(ValueError, match='DataArray has unexpected dimensions: meh'):
        save_scitiff(sample_image.rename_dims({'x': 'meh'}), '')


def test_export_multi_dimension_coordinate_raises(
    sample_image_3d_coordinate: sc.DataArray,
) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            'Only variables with at most 2 dimensions are allowed for metadata.'
        ),
    ):
        save_scitiff(sample_image_3d_coordinate, 'test.tiff')


@pytest.fixture
def sample_image_2d_str_coordinate(sample_image: sc.DataArray) -> sc.DataArray:
    new_image = sample_image.copy()
    pixel_id = sc.array(dims=['pixel-id'], values=[str(i) for i in range(12)]).fold(
        dim='pixel-id', dims=['y', 'x'], shape=[3, 4]
    )
    new_image.coords['pixel-id-str'] = pixel_id
    return new_image


def test_export_2D_string_coordinate_raises(
    sample_image_2d_str_coordinate: sc.DataArray,
) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("Failed to construct pydantic model"),
    ):
        save_scitiff(sample_image_2d_str_coordinate, 'test.tiff')


def test_save_wrong_dtype_raises(sample_image: sc.DataArray) -> None:
    with pytest.raises(sc.DTypeError, match='DataArray has unexpected dtype: int64'):
        save_scitiff(sample_image.astype(int), 'test.tiff')
    with pytest.raises(sc.DTypeError, match='DataArray has unexpected dtype: float64'):
        save_scitiff(sample_image.astype(float), 'test.tiff')
