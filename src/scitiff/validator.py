# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Ess-dmsc-dram contributors (https://github.com/ess-dmsc-dram)
from jsonschema import validate

from ._resources import SCITIFF_METADATA_CONTAINER_SCHEMA

# We use the jsonschema file to validate the metadata container
# instead of the pydantic model because the json file platform independent
# and can be used in other languages as well.
# And we would like to keep python validator consistent with other languages.


def validate_scitiff_metadata_container(instance: dict) -> None:
    """Validate the metadata container.

    It validates a dictionary that carries the metadata of a scitiff file.
    It is because tiff or other image formats can carry other metadata as well,
    therefore the container schema defines under what name the scitiff metadata
    should be stored.

    Example
    -------
    A tiff image stack can have metadata like:

    .. code-block:: json

      {
        "owner": "Sun",
        "date": "2025-01-01",
        "scitiff": {"...": "..."}
      }


    This whole dictionary validates as :class:`scitiff.SciTiffMetadataContainer`.
    It implies that the dictionary under `scitiff` key is the scitiff metadata,
    should validate as the :class:`scitiff.SciTiffMetadata`.

    Parameters
    ----------
    instance : dict
        The metadata container to validate.

    """
    validate(instance, SCITIFF_METADATA_CONTAINER_SCHEMA)
