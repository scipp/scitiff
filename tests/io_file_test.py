# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 ess-maintainers (https://github.com/orgs/scipp/teams/ess-maintainers)
import json
import pathlib

import numpy as np
import pytest
import scipp as sc
import tifffile as tf
from scipp.testing import assert_identical

from scitiff import (
    SCITIFF_IMAGE_STACK_DIMENSIONS,
    DAQMetadata,
    ExperimentIdentifier,
    ExperimentIdentifierType,
    ImageProcessMetadata,
    Person,
    ProcessIdentifier,
)
from scitiff.io import (
    ImageJMetadataNotFoundWarning,
    IncompatibleDtypeWarning,
    ScitiffMetadataWarning,
    UnmatchedMetadataWarning,
    extract_metadata,
    load_scitiff,
    save_scitiff,
    to_scitiff_image,
)


def test_export_and_load_scitiff(
    sample_image: sc.DataArray, tmp_path: pathlib.Path
) -> None:
    tmp_file_path = tmp_path / 'test.tiff'
    save_scitiff(sample_image, tmp_file_path)
    loaded_image = load_scitiff(tmp_file_path, only_image=True)
    # exported image already has right order of dimensions
    assert_identical(sample_image, loaded_image)


def test_export_and_load_scitiff_datagroup(
    sample_image_datagroup: sc.DataGroup, tmp_path: pathlib.Path
) -> None:
    tmp_file_path = tmp_path / 'test.tiff'
    save_scitiff(sample_image_datagroup, tmp_file_path)
    loaded_image = load_scitiff(tmp_file_path)
    # exported image already has right order of dimensions
    assert_identical(sample_image_datagroup['image'], loaded_image['image'])
    assert_identical(sample_image_datagroup['extra'], loaded_image['extra'])


def test_export_and_load_and_export_scitiff_datagroup(
    sample_image_datagroup: sc.DataGroup, tmp_path: pathlib.Path
) -> None:
    tmp_file_path = tmp_path / 'test.tiff'
    save_scitiff(sample_image_datagroup, tmp_file_path)
    loaded_image = load_scitiff(tmp_file_path)
    # exported image already has right order of dimensions
    assert_identical(sample_image_datagroup['image'], loaded_image['image'])
    assert_identical(sample_image_datagroup['extra'], loaded_image['extra'])
    # save again
    save_scitiff(loaded_image, tmp_path / 'test2.tiff')


def test_export_and_load_and_export_scitiff_datagroup_with_extra_none(
    sample_image_datagroup: sc.DataGroup, tmp_path: pathlib.Path
) -> None:
    sample_image_datagroup['extra'] = None
    tmp_file_path = tmp_path / 'test.tiff'
    save_scitiff(sample_image_datagroup, tmp_file_path)
    loaded_image = load_scitiff(tmp_file_path)
    # exported image already has right order of dimensions
    assert_identical(sample_image_datagroup['image'], loaded_image['image'])
    assert_identical(sample_image_datagroup['extra'], loaded_image['extra'])
    assert loaded_image['extra'] is None
    # save again
    save_scitiff(loaded_image, tmp_path / 'test2.tiff')


def test_export_and_load_scitiff_with_scalar_coord_str(
    sample_image, tmp_path: pathlib.Path
) -> None:
    sample_image.coords['some-name-key'] = sc.scalar('some-name-value')
    tmp_file_path = tmp_path / 'test_scalar_coord_str.tiff'
    save_scitiff(sample_image, tmp_file_path)
    loaded_image = load_scitiff(tmp_file_path, only_image=True)
    # exported image already has right order of dimensions
    assert sc.identical(sample_image, loaded_image)
    tmp_file_path.unlink()


def test_export_and_load_scitiff_with_scalar_coord(
    sample_image, tmp_path: pathlib.Path
) -> None:
    sample_image.coords['Ltotal'] = sc.scalar(0.0, unit='mm')
    tmp_file_path = tmp_path / 'test_scalar_coord.tiff'
    save_scitiff(sample_image, tmp_file_path)
    loaded_image = load_scitiff(tmp_file_path, only_image=True)
    # exported image already has right order of dimensions
    assert sc.identical(sample_image, loaded_image)


def test_load_squeeze_false(sample_image, tmp_path: pathlib.Path) -> None:
    tmp_file_path = tmp_path / 'test.tiff'
    save_scitiff(sample_image, tmp_file_path)
    loaded_image = load_scitiff(tmp_file_path, squeeze=False, only_image=True)
    assert loaded_image.dims == SCITIFF_IMAGE_STACK_DIMENSIONS


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
    sample_image: sc.DataArray, tmp_path: pathlib.Path, dtype, expected_dtype
) -> None:
    """Test scipp incompatible dtypes but compatible with ImageJ."""
    tmp_file_path = tmp_path / 'wrong_dtype.tiff'
    _save_data_array_with_wrong_dtype(sample_image, tmp_file_path, dtype=dtype)
    with pytest.warns(
        IncompatibleDtypeWarning,
        match=f"dtype of ``{dtype}``. "
        f"The dtype will be converted to ``<class 'numpy.{expected_dtype}'>``",
    ):
        loaded_image = load_scitiff(
            tmp_file_path, resolve_channels=False, only_image=True
        )

    assert loaded_image.dims == tuple(f"dim_{i}" for i in range(3))  # Image is squeezed
    assert loaded_image.dtype == expected_dtype


@pytest.mark.parametrize(
    ('dtype', 'expected_dtype'),
    [('int8', 'int32'), ('int16', 'int32'), ('float16', 'float32')],
)
def test_load_imagej_scipp_incompatible_dtype_fallback(
    sample_image: sc.DataArray, tmp_path: pathlib.Path, dtype, expected_dtype
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
            loaded_image = load_scitiff(tmp_file_path, only_image=True)

    assert loaded_image.dims == tuple(f"dim_{i}" for i in range(3))  # Image is squeezed
    assert loaded_image.dtype == expected_dtype


@pytest.mark.parametrize(('dtype', 'expected_dtype'), [('float64', 'float64')])
def test_load_imagej_incompatible_dtype_fallback(
    sample_image: sc.DataArray, tmp_path: pathlib.Path, dtype, expected_dtype
) -> None:
    """Test ImageJ incompatible but scipp compatible dtypes.

    float64 is supported by scipp so it should not raise a dtype conversion warning.
    """
    tmp_file_path = tmp_path / 'wrong_dtype.tiff'
    _save_data_array_with_wrong_dtype(sample_image, tmp_file_path, dtype=dtype)
    with pytest.warns(
        ImageJMetadataNotFoundWarning, match='ImageJ metadata not found'
    ):  # These dtypes are not supported by tifffile with `imagej=True`
        loaded_image = load_scitiff(tmp_file_path, only_image=True)

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


def test_load_incompatible_metadata_warns(
    sample_image: sc.DataArray, tmp_path: pathlib.Path
) -> None:
    tmp_file_path = tmp_path / 'unmatching_size.tiff'
    _save_data_array_with_unmatching_shape(sample_image, tmp_file_path)
    with pytest.warns(
        UnmatchedMetadataWarning,
        match="Size of the image data does not match with the metadata.",
    ):
        loaded_image = load_scitiff(tmp_file_path, only_image=True)

    assert loaded_image.dims == tuple(f"dim_{i}" for i in range(2))  # Image is squeezed


def test_load_without_metadata_warns(tmp_path: pathlib.Path) -> None:
    import numpy as np

    no_metadata_file_path = tmp_path / 'no_metadata.tiff'
    tf.imwrite(no_metadata_file_path, [[1.0, 2], [3, 4]])
    with pytest.warns(
        ImageJMetadataNotFoundWarning, match='ImageJ metadata not found.'
    ):
        loaded_image = load_scitiff(no_metadata_file_path, only_image=True)

    assert loaded_image.dims == tuple(f"dim_{i}" for i in range(2))
    assert np.all(loaded_image.values == np.array([[1.0, 2], [3, 4]]))


def test_load_broken_metadata_warns(tmp_path: pathlib.Path) -> None:
    no_metadata_file_path = tmp_path / 'broken_metadata.tiff'
    arbitrary_image = np.array([[1.0, 2], [3, 4]], dtype='float32')
    tf.imwrite(
        no_metadata_file_path, arbitrary_image, imagej=True, metadata={"meh": "meh"}
    )
    with pytest.warns(ScitiffMetadataWarning, match='Scitiff metadata is broken.'):
        loaded_image = load_scitiff(no_metadata_file_path, only_image=True)

    assert loaded_image.dims == tuple(f"dim_{i}" for i in range(2))
    assert np.all(loaded_image.values == arbitrary_image)


@pytest.fixture
def example_daq_meta() -> DAQMetadata:
    return DAQMetadata(
        instrument=['nido', 'coda'],
        simulated=True,
        principal_investigators=[Person(name="Wash Bear")],
        experiment_identifiers=[
            ExperimentIdentifier(
                type=ExperimentIdentifierType.PROPOSAL_ID, value="123456"
            )
        ],
    )


def test_export_and_load_scitiff_datagroup_with_daq_meta(
    sample_image_datagroup: sc.DataGroup,
    example_daq_meta: DAQMetadata,
    tmp_path: pathlib.Path,
) -> None:
    sample_image_datagroup['daq'] = example_daq_meta
    tmp_file_path = tmp_path / 'test.tiff'
    save_scitiff(sample_image_datagroup, tmp_file_path)
    loaded_image = load_scitiff(tmp_file_path)
    assert loaded_image['daq'] == example_daq_meta


def test_export_and_load_scitiff_wrong_email_warns(
    sample_image_datagroup: sc.DataGroup,
    example_daq_meta: DAQMetadata,
    tmp_path: pathlib.Path,
) -> None:
    # Person model validates the email.
    with pytest.warns(UserWarning, match="invalid-email-address"):
        example_daq_meta.principal_investigators.append(
            Person(name="", email="invalid-email-address")
        )
    sample_image_datagroup['daq'] = example_daq_meta
    tmp_file_path = tmp_path / 'test.tiff'
    # Scitiff will not complain when it is being saved.
    save_scitiff(sample_image_datagroup, tmp_file_path)
    # Scitiff will complain when loading.
    with pytest.warns(UserWarning, match="invalid-email-address"):
        loaded_image = load_scitiff(tmp_file_path)
    assert loaded_image['daq'] == example_daq_meta


def test_export_and_load_scitiff_wrong_orcid_warns(
    sample_image_datagroup: sc.DataGroup,
    example_daq_meta: DAQMetadata,
    tmp_path: pathlib.Path,
) -> None:
    # Person model validates the orcid.
    with pytest.warns(UserWarning, match="invalid-orcid"):
        example_daq_meta.principal_investigators.append(
            Person(name="", orcid="invalid-orcid")
        )
    sample_image_datagroup['daq'] = example_daq_meta
    tmp_file_path = tmp_path / 'test.tiff'
    # Scitiff will not complain when it is being saved.
    save_scitiff(sample_image_datagroup, tmp_file_path)
    # Scitiff will complain when loading.
    with pytest.warns(UserWarning, match="invalid-orcid"):
        loaded_image = load_scitiff(tmp_file_path)

    assert loaded_image['daq'] == example_daq_meta


@pytest.fixture
def example_process_meta() -> ImageProcessMetadata:
    return ImageProcessMetadata(
        result_type="Dummy",
        processing_steps=["random-generator"],
        parameters={"seed": 42, "algorithm": "uniform"},
        process_identifiers=[
            ProcessIdentifier(type="python", value=f"{np.__name__}=={np.__version__}")
        ],
        coordinate_descriptions={"t": "time of flight, NOT-time-of-arrival."},
    )


def test_export_and_load_scitiff_datagroup_with_process_meta(
    sample_image_datagroup: sc.DataGroup,
    example_process_meta: ImageProcessMetadata,
    tmp_path: pathlib.Path,
) -> None:
    sample_image_datagroup['process'] = example_process_meta
    tmp_file_path = tmp_path / 'test.tiff'
    save_scitiff(sample_image_datagroup, tmp_file_path)
    loaded_image = load_scitiff(tmp_file_path)
    assert loaded_image['process'] == example_process_meta


def test_wrong_coordinate_name_in_description_warns(
    sample_image_datagroup: sc.DataGroup,
    example_process_meta: ImageProcessMetadata,
    tmp_path: pathlib.Path,
) -> None:
    example_process_meta.coordinate_descriptions[
        'something that probably does not exist'
    ] = ""
    tmp_file_path = tmp_path / 'test.tiff'
    sample_image_datagroup['process'] = example_process_meta
    # Should not fail but warn about the non-existing names.
    with pytest.warns(
        expected_warning=UserWarning, match="something that probably does not exist"
    ):
        save_scitiff(sample_image_datagroup, tmp_file_path)

    # Should warn about the same thing when loaded.
    with pytest.warns(
        expected_warning=UserWarning, match="something that probably does not exist"
    ):
        loaded_image = load_scitiff(tmp_file_path)

    assert loaded_image['process'] == example_process_meta
