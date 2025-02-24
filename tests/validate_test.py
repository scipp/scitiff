# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)
import pytest
import scipp as sc

from scitiff import validate_scitiff
from scitiff.io import extract_metadata


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


def test_validation(sample_image) -> None:
    validate_scitiff(extract_metadata(sample_image))
