# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)
import logging
import pathlib

import scipp as sc

from scitiff.data import hyperstack_example
from scitiff.io import save_scitiff


def _require_rich() -> None:
    try:
        import rich  # noqa: F401 - just for checking
    except ImportError as e:
        raise ImportError(
            "You need `rich` to run this script.\n"
            "Please install `rich` or `scitiff` with `GUI` "
            "optional dependencies.\n"
            "Recommended command: pip install scitiff[gui]."
        ) from e


def _get_rich_logger() -> logging.Logger:
    _require_rich()
    from rich.logging import RichHandler

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.addHandler(RichHandler())
        logger.setLevel(logging.INFO)
    return logger


def _get_scitiff_version() -> str:
    from scitiff import __version__

    if 'dev' in __version__ or __version__.startswith('0.'):
        raise RuntimeError(
            "Only release versions must be used for dumping an example image."
        )
    else:
        return __version__


def _example_image_after_2610() -> sc.DataGroup:
    from scitiff._schema import DAQMetadata
    from scitiff.data import hyperstack_example_with_variances_and_mask

    # Trimmed the example image
    example_image = hyperstack_example_with_variances_and_mask()['x', :10]['y', :10]
    daq_metadata = DAQMetadata(
        facility='scitiff-dev',
        instrument='computer',
        detector_type='computer',
        simulated=True,
    )
    extra = {
        'string-value': 'string-value',
        'int-value': 1,
        'float-value': 1.2,
        'scipp-scalar-number': sc.scalar(1, unit='count'),
        'scipp-scalar-datetime': sc.datetime('now'),
    }
    return sc.DataGroup(image=example_image, daq=daq_metadata, extra=extra)


def _example_image(version: str) -> sc.DataArray | sc.DataGroup:
    from packaging.version import Version

    cur_version = Version(version)
    if cur_version < Version('25.12.0'):  # When saving mask and stdev was introduced.
        return hyperstack_example()['x', :10]['y', :10]
    elif cur_version < Version('26.1.0'):  # When saving data group was introduced.
        from scitiff.data import hyperstack_example_with_variances_and_mask

        return hyperstack_example_with_variances_and_mask()['x', :10]['y', :10]
    else:
        return _example_image_after_2610()


def dump_example_scitiff():
    """Dump an example scitiff file with all possible metadata fields."""

    logger = _get_rich_logger()
    version = _get_scitiff_version()
    default_dir = pathlib.Path(__file__).parent.parent / pathlib.Path(
        'tests/_regression_test_files'
    )
    prefix = 'scitiff_'
    suffix = '.tiff'
    new_file_name = ''.join([prefix, version, suffix])
    new_file_path = default_dir / new_file_name
    logger.info("Dumping new example scitiff at %s", new_file_path.as_posix())
    image = _example_image(version=version)
    logger.info(image)
    logger.info("Dumping image for version %s", version)
    save_scitiff(dg=image, file_path=new_file_path)
    logger.info(
        "Successfully saved image for version %s in %s",
        version,
        new_file_path.as_posix(),
    )


if __name__ == "__main__":
    dump_example_scitiff()
