# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 ess-maintainers (https://github.com/orgs/scipp/teams/ess-maintainers)
import pytest
import scipp as sc


@pytest.fixture
def sample_image() -> sc.DataArray:
    pattern = [[i * 400 + j for j in range(4)] for i in range(3)]
    pixel_id = sc.arange(dim='pixel-id', start=0, stop=12, unit='dimensionless').fold(
        dim='pixel-id', sizes={'x': 4, 'y': 3}
    )
    sample_img = sc.DataArray(
        data=sc.array(
            dims=['t', 'y', 'x'],
            values=[pattern, pattern[::-1]],
            unit='counts',
            dtype=sc.DType.float32,
        ),
        coords={
            't': sc.array(dims=['t'], values=[0, 1], unit='s'),
            'global-time': sc.datetimes(dims=['t'], values=[0, 1], unit='s'),
            'timestamp': sc.datetime('now', unit='hour'),
            'y': sc.linspace(dim='y', start=0.0, stop=300.0, num=3, unit='mm'),
            'x': sc.linspace(dim='x', start=0.0, stop=400.0, num=4, unit='mm'),
            # 2D coordinate
            'pixel-id': pixel_id,
        },
    )
    return sample_img


@pytest.fixture
def sample_image_datagroup(sample_image: sc.DataArray) -> sc.DataGroup:
    return sc.DataGroup(
        image=sample_image,
        extra={
            'writer': 'RapBear',
            'generation': 1,
            'today': sc.datetime('now', unit='s'),
        },
    )
