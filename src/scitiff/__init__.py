# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402, F401, I

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

from ._img_processors import values
from ._schema import (
    SCITIFF_IMAGE_STACK_DIMENSIONS,
    SciTiffMetadata,
    SciTiffMetadataContainer,
)
from .executables import show_metadata
from .io import (
    concat_mask_as_channels,
    concat_stdevs_and_mask_as_channels,
    concat_stdevs_as_channels,
    load_scitiff,
    resolve_scitiff_channels,
    save_scitiff,
    to_scitiff_image,
)
from .validator import validate_scitiff_metadata_container
