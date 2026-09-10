# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 ess-maintainers (https://github.com/orgs/scipp/teams/ess-maintainers)
import pytest
import scipp as sc
from scipp.testing import assert_identical

import scitiff


@pytest.fixture
def sample_image_var_mask(sample_image: sc.DataArray) -> sc.DataArray:
    sample_image.variances = (sample_image**2).values
    mask = sc.ones_like(sc.values(sample_image.data)).to(dtype=bool)
    sample_image.masks['whole'] = mask
    return sample_image


@pytest.fixture
def sample_image_channels(sample_image_var_mask: sc.DataArray) -> sc.DataArray:
    return scitiff.to_scitiff_image(
        sample_image_var_mask, concat_stdevs_and_mask=True, mask_name='whole'
    )


def test_channel_slicer_image(sample_image_channels) -> None:
    values_only = scitiff.values(sample_image_channels)
    assert_identical(values_only.coords['c'], sc.scalar('intensities'))


def test_channel_slicer_image_only_variances(sample_image) -> None:
    sample_image.variances = (sample_image**2).values
    result = scitiff.values(sample_image)
    assert result.variances is None
    assert len(result.masks) == 0


def test_channel_slicer_image_variances_and_mask(sample_image_var_mask) -> None:
    result = scitiff.values(sample_image_var_mask)
    assert result.variances is None
    assert len(result.masks) == 0


def test_channel_slicer_image_1d_mask_not_dropped(sample_image_var_mask) -> None:
    sample_image_var_mask.masks['1dmask'] = sc.array(dims=['t'], values=[True, False])
    result = scitiff.values(sample_image_var_mask)
    assert result.variances is None
    assert len(result.masks) == 1
    assert 'whole' not in result.masks
    assert '1dmask' in result.masks


def test_no_c_coord_multi_channel_image_raises(sample_image_channels) -> None:
    with pytest.raises(ValueError, match='Cannot decide intensity channel'):
        scitiff.values(sample_image_channels.drop_coords('c'))


def test_no_image_key_in_datagroup() -> None:
    with pytest.raises(KeyError, match='Cannot find an image'):
        scitiff.values(sc.DataGroup())
