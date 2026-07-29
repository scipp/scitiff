# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp(ESS) contributors (https://github.com/scipp)
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


def _example_image_after_2660() -> sc.DataGroup:
    # Reuse most of the previous example.
    dg = _example_image_after_2610()
    dg['image'] = dg['image'].copy(deep=True)
    # Trimmed the example image
    example_image = dg['image']
    # From 26.6.0, 2D coordinates are allowed.
    example_image.coords['pixel-id'] = sc.arange(
        dim='pixel-id',
        start=0,
        stop=example_image.sizes['x'] * example_image.sizes['y'],
    ).fold(dim='pixel-id', sizes={xydim: example_image.sizes[xydim] for xydim in 'xy'})
    example_image.coords['pixel-distance'] = example_image.coords['pixel-id'].to(
        dtype=float
    )
    return dg


def _example_image_after_2670() -> sc.DataGroup:
    from scitiff._schema import (
        DAQMetadata,
        ExperimentIdentifier,
        ExperimentIdentifierType,
        ImageProcessMetadata,
        ImageResultType,
        NeutronMetadata,
        NeutronSourceType,
        Person,
        PhotonConvertDetectorMetadata,
        ProcessIdentifier,
        ScippVariable0D,
        SourceType,
    )

    # Reuse image of the previous example.
    dg = _example_image_after_2660()
    # From 26.7.0, more DAQ metadata and image process metadata fields were added
    # and bug was fixed for extra metadata None.
    dg['extra'] = None
    dg['daq'] = DAQMetadata(
        facility=['ess', 'esss'],
        instrument=['coda', 'nido'],
        detector_type='optic-converter',
        detector=[
            PhotonConvertDetectorMetadata(
                scintillator_type='A123', detector_identifier="ABC123"
            ),
            PhotonConvertDetectorMetadata(
                scintillator_type='A123', detector_identifier="ABD122"
            ),
        ],
        source_type=SourceType.NEUTRON,
        source=NeutronMetadata(
            neutron_type=NeutronSourceType.LONG_PULSE,
            wavelength_range=(
                ScippVariable0D(values=1.0, unit='angstrom', dtype='float'),
                ScippVariable0D(values=10.0, unit='angstrom', dtype='float'),
            ),
        ),
        simulated=True,
        principal_investigators=[
            Person(
                affiliation='ess',
                email='valid@email.com',
                name='Jeffry',
                orcid="0000-0000-0000-0001",
            )
        ],
        team=[Person(name='Herrison'), Person(name='Fuizao')],
        local_contacts=[Person(name='Some one at ESS')],
        experiment_identifiers=[
            ExperimentIdentifier(
                type=ExperimentIdentifierType.RUN_NUMBER,
                value='1234',
                description='the overnight run for the sample 1',
            )
        ],
    )
    dg['process'] = ImageProcessMetadata(
        result_type=ImageResultType.NORMALIZED,
        processing_steps=[
            'dark-current-subtraction',
            'normalized-by-proton-charge',
            'histogram-in-wavelength',
            'normalized-by-openbeam-image',
        ],
        parameters={'wav-bin-nums': 300},
        process_identifiers=[
            ProcessIdentifier(
                type='notebook',
                value='odin-data-reduction.ipynb',
                description='normalization-notebook',
            )
        ],
        coordinate_descriptions={'pixel-id': 'Pixel ID of the entire detector'},
    )
    return dg


def _example_image(version: str) -> sc.DataArray | sc.DataGroup:
    from packaging.version import Version

    cur_version = Version(version)
    if cur_version < Version('25.12.0'):  # When saving mask and stdev was introduced.
        return hyperstack_example()['x', :10]['y', :10]
    elif cur_version < Version('26.1.0'):  # When saving data group was introduced.
        from scitiff.data import hyperstack_example_with_variances_and_mask

        return hyperstack_example_with_variances_and_mask()['x', :10]['y', :10]
    elif cur_version < Version('26.6.0'):  # When saving data group was introduced.
        return _example_image_after_2610()
    elif cur_version < Version('26.7.0'):  # When DAQ/process metadata were introduced.
        return _example_image_after_2660()
    else:
        return _example_image_after_2670()


def dump_example_scitiff():
    """Dump an example scitiff file with all possible metadata fields."""

    logger = _get_rich_logger()
    version = _get_scitiff_version()
    default_dir = pathlib.Path(__file__).parent.parent / pathlib.Path(
        'tests/_regression_test_files'
    )
    prefix = 'scitiff-'
    suffix = '.tiff'
    new_file_name = ''.join([prefix, version, suffix])
    new_file_path = default_dir / new_file_name
    logger.info("Dumping new example scitiff at %s", new_file_path.as_posix())
    image = _example_image(version=version)
    logger.info(image)
    if isinstance(image, sc.DataGroup):
        for k, v in image.items():
            logger.info("%s: ", k)
            logger.info(v)
    logger.info("Dumping image for version %s", version)
    save_scitiff(dg=image, file_path=new_file_path)
    logger.info(
        "Successfully saved image for version %s in %s",
        version,
        new_file_path.as_posix(),
    )


if __name__ == "__main__":
    dump_example_scitiff()
