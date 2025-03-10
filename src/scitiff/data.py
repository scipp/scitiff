# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)
import scipp as sc


def hyperstack_example() -> sc.DataArray:
    """Create a sample image with ImageJ Hyperstack.

    The image is not aligned with the ImageJ Hyperstack default dimension order.
    It has `c`, `t`, `z`, `y`, `x` dimensions with the following sizes:

    - `c`: 3  (RBG)
    - `t`: 4
    - `z`: 2
    - `y`: 300
    - `x`: 400

    Channel 0 and 1 represents data and variances respectively.

    """

    pattern = [[i * 400 + j for j in range(400)] for i in range(300)]
    return sc.DataArray(
        data=sc.array(
            dims=['c', 't', 'z', 'y', 'x'],
            values=[
                [
                    [[*pattern[75 * i_t :], *pattern[: 75 * i_t]] for _ in range(2)]
                    for i_t in range(4)
                ]
                for _ in range(3)
            ],
            unit='counts',
            dtype=sc.DType.float32,
        ),
        coords={
            'c': sc.array(dims=['c'], values=['R', 'G', 'B']),
            't': sc.array(dims=['t'], values=[0, 1, 2, 3], unit='s'),
            'z': sc.array(dims=['z'], values=[10, 20], unit='mm'),
            'y': sc.linspace(dim='y', start=0.0, stop=300.0, num=300, unit='mm'),
            'x': sc.linspace(dim='x', start=0.0, stop=400.0, num=400, unit='mm'),
        },
    )
