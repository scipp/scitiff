# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)
# ruff: noqa: E402, F401, I

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

from ._schema import (
    SCITIFF_IMAGE_STACK_DIMENSIONS,
    SciTiffMetadata,
    SciTiffMetadataContainer,
)
from .io import load_scitiff, save_scitiff
from .validator import validate_scitiff_metadata_container
