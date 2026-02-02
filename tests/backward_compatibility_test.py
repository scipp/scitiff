# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp(ESS) contributors (https://github.com/scipp)
import json
import pathlib
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

import pydantic
import pytest
import requests
from packaging.version import Version

from scitiff.io import load_scitiff

_LOWER_BOUND_VERSION = Version('25.1.0')
_SCITIFF_TEST_CACHE = pathlib.Path.home() / '.cache' / 'scitiff-test'
_SCITIFF_TEST_CACHE.mkdir(parents=True, exist_ok=True)
_CACHED_PACKAGE_INFO_PATH = _SCITIFF_TEST_CACHE / 'scitiff-package-info.json'


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


class ScitiffPackageInfoCache(pydantic.BaseModel):
    last_updated: datetime = pydantic.Field(default_factory=_utc_now)
    versions: tuple[str, ...]

    @property
    def testing_versions(self) -> tuple[str, ...]:
        return tuple(
            _v.base_version
            for _v in sorted(
                Version(version)
                for version in self.versions
                if Version(version) >= _LOWER_BOUND_VERSION
            )
        )

    @classmethod
    def from_pypi(cls) -> 'ScitiffPackageInfoCache':
        url = "https://pypi.org/pypi/scitiff/json"
        response = requests.get(url, timeout=1)
        data = response.json()
        pacakge_info = ScitiffPackageInfoCache(versions=tuple(data['releases'].keys()))
        # Save the info if possible
        try:
            _CACHED_PACKAGE_INFO_PATH.write_text(
                data=pacakge_info.model_dump_json(indent=True)
            )
        except Exception as err:
            import warnings

            warnings.warn(
                'Could not save scitiff package info into '
                f'{_CACHED_PACKAGE_INFO_PATH.as_posix()}.\n'
                f'An error raised: {err}\n'
                'Skipping saving file... use `from_pypi` instead '
                'if you do not need to save the info.',
                RuntimeWarning,
                stacklevel=3,
            )

        return pacakge_info

    @classmethod
    def maybe_from_cache(cls) -> 'ScitiffPackageInfoCache':
        if _CACHED_PACKAGE_INFO_PATH.exists():
            try:
                latest = ScitiffPackageInfoCache(
                    **json.loads(_CACHED_PACKAGE_INFO_PATH.read_text())
                )
            except Exception:
                ...
            else:
                if (_utc_now() - latest.last_updated).seconds <= 300:
                    return latest

        return cls.from_pypi()


SCITIFF_PACKAGE_INFO = ScitiffPackageInfoCache.maybe_from_cache()


@dataclass
class KnownError:
    error_type: type
    """Type of error. i.e. RuntimeError."""
    error_match: str
    """Match description of the error for pytest."""


_KNOWN_ERRORS: dict[str, tuple[KnownError]] = {}  # No known errors yet


@contextmanager
def known_backward_compatibility_issues(_errors: tuple[KnownError, ...]):
    if len(_errors) == 1:
        with pytest.raises(_errors[0].error_type, match=_errors[0].error_match):
            yield
    elif len(_errors) >= 1:
        with pytest.raises(_errors[0].error_type, match=_errors[0].error_match):
            with known_backward_compatibility_issues(_errors[1:]):
                yield
    else:
        yield


def _get_scitiff_example_file_path(version: str) -> pathlib.Path:
    _test_files_dir = pathlib.Path(__file__).parent / '_regression_test_files'
    return _test_files_dir / f'scitiff-{version}.tiff'


@pytest.mark.parametrize(
    argnames=('scitiff_version'),
    argvalues=SCITIFF_PACKAGE_INFO.testing_versions,
)
def test_example_files_for_all_releases_exist(scitiff_version) -> None:
    cur_version_file = _get_scitiff_example_file_path(scitiff_version)
    if not cur_version_file.exists():
        raise RuntimeError(
            f"Example file for version {scitiff_version} does not exist. "
            "Use `tools/dump_scitiff_example.py` to create a new one "
            "with the missing release version.\n"
            "Creating an example file for a new release is not automated.\n"
            "Therefore a developer will have to create a new scitiff file "
            "and push it to the repo manually."
        )


@pytest.mark.parametrize(
    argnames=('scitiff_version'),
    argvalues=SCITIFF_PACKAGE_INFO.testing_versions,
)
def test_loading_old_version_files(scitiff_version) -> None:
    _known_erros = _KNOWN_ERRORS.get(scitiff_version, ())
    with known_backward_compatibility_issues(_known_erros):
        load_scitiff(_get_scitiff_example_file_path(scitiff_version))
